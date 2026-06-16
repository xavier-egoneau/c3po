"""
Téléchargement de modèles GGUF depuis Hugging Face, en HTTP brut (urllib stdlib),
sans dépendance externe ni CLI/token requis.

Protocole `c3po load <user>/<repo>[:QUANT]` :
  1. on liste les fichiers du repo via l'API publique HF (tailles + SHA256 LFS) ;
  2. on regroupe par quantization (en réunissant les éventuels shards) ;
  3. si la quant n'est pas précisée, on choisit automatiquement la plus grosse
     qui tient dans la VRAM/RAM dispo (taille × FIT_MARGIN ≤ mémoire) ;
  4. on télécharge dans ~/.c3po/models avec reprise, progression et vérification SHA256.

Aucun token n'est requis pour les modèles publics. Si la variable d'environnement
HF_TOKEN (ou HUGGING_FACE_HUB_TOKEN) est définie, elle est utilisée pour les modèles
gated — mais elle n'est jamais exigée.
"""

from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .models import models_dir, FIT_MARGIN

HF_BASE = "https://huggingface.co"
_USER_AGENT = "c3po-llm-runtime"
_CHUNK = 1 << 20  # 1 Mo

# Reconnaît la quantization dans un nom de fichier GGUF (ex: Q4_K_M, Q8_0, IQ3_XXS, F16).
_QUANT_RE = re.compile(r"(IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|BF16|F16|F32)", re.IGNORECASE)
# Reconnaît un suffixe de shard (ex: -00001-of-00002).
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})", re.IGNORECASE)


@dataclass
class GGUFFile:
    path: str            # chemin dans le repo (ex: "Model-Q4_K_M.gguf")
    size: int            # taille réelle en octets
    sha256: str | None   # oid LFS si disponible (sinon None → pas de vérif)


@dataclass
class QuantOption:
    quant: str
    files: list[GGUFFile]          # plusieurs fichiers si modèle splitté en shards

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)


# ---------------------------------------------------------------------------
# Parsing (pur, testable sans réseau)
# ---------------------------------------------------------------------------

def parse_ref(ref: str) -> tuple[str, str | None]:
    """
    "user/repo:Q4_K_M" → ("user/repo", "Q4_K_M")
    "user/repo"        → ("user/repo", None)

    Le repo doit être explicite (forme <user>/<repo>) : on télécharge ce qu'on
    nomme, pas d'alias opaque.
    """
    repo, _, quant = ref.partition(":")
    repo = repo.strip()
    if "/" not in repo:
        raise ValueError(
            f"Référence invalide : '{ref}'. Donnez le repo Hugging Face complet, "
            f"ex. 'bartowski/Qwen2.5-7B-Instruct-GGUF[:Q4_K_M]'."
        )
    return repo, (quant.strip() or None)


def _parse_quant(filename: str) -> str | None:
    m = _QUANT_RE.search(filename)
    return m.group(1).upper() if m else None


def _is_mmproj(filename: str) -> bool:
    return "mmproj" in filename.lower()


def has_mmproj(files: list[GGUFFile]) -> bool:
    """Le repo contient-il un projecteur multimodal (→ modèle vision/audio) ?"""
    return any(_is_mmproj(Path(f.path).name) for f in files if f.path.endswith(".gguf"))


def select_mmproj(files: list[GGUFFile]) -> GGUFFile | None:
    """
    Choisit un projecteur multimodal à télécharger.

    On préfère BF16/F16 (bonne qualité, taille raisonnable) puis Q8_0, puis F32 en dernier
    recours. Le runtime multimodal n'est pas encore branché ; ceci prépare seulement les
    assets nécessaires.
    """
    candidates = [
        f for f in files
        if f.path.endswith(".gguf") and _is_mmproj(Path(f.path).name)
    ]
    if not candidates:
        return None
    preference = {"BF16": 0, "F16": 1, "Q8_0": 2, "F32": 3}

    def key(f: GGUFFile) -> tuple[int, int, str]:
        quant = _parse_quant(Path(f.path).name) or ""
        return (preference.get(quant, 99), f.size, f.path)

    return sorted(candidates, key=key)[0]


def group_by_quant(files: list[GGUFFile]) -> dict[str, QuantOption]:
    """
    Regroupe les .gguf en options téléchargeables.

    On regroupe d'abord par *base* (nom de fichier sans le suffixe de shard), pour
    qu'un set splitté `…-00001-of-00002.gguf` compte comme un seul modèle — et ne soit
    pas additionné à une éventuelle version fichier unique de la même quant présente
    dans le repo. Si plusieurs bases donnent la même quant (ex. un `mmproj` multimodal
    qui partage le tag `BF16`), on conserve la plus grosse (le vrai modèle).
    """
    # 1) regrouper les fichiers par base
    bases: dict[str, dict] = {}
    for f in files:
        if not f.path.endswith(".gguf"):
            continue
        name = Path(f.path).name
        if _is_mmproj(name):
            continue  # projecteur multimodal : pas un modèle texte téléchargeable seul
        stem = name[: -len(".gguf")]
        m = _SHARD_RE.search(stem)
        if m:
            base = stem[: m.start()] + stem[m.end():]
            bases.setdefault(base, {"shards": [], "single": None})["shards"].append(f)
        else:
            bases.setdefault(stem, {"shards": [], "single": None})["single"] = f

    # 2) une représentation par base : le set de shards complet, sinon le fichier unique
    options: dict[str, QuantOption] = {}
    for base, parts in bases.items():
        quant = _parse_quant(base)
        if quant is None:
            continue
        if parts["shards"]:
            chosen = sorted(parts["shards"], key=lambda x: x.path)
        elif parts["single"] is not None:
            chosen = [parts["single"]]
        else:
            continue
        candidate = QuantOption(quant=quant, files=chosen)
        # 3) si collision de quant entre deux bases, garder la plus grosse
        existing = options.get(quant)
        if existing is None or candidate.total_size > existing.total_size:
            options[quant] = candidate
    return options


def choose_quant(
    options: dict[str, QuantOption],
    available_gb: float,
    requested: str | None = None,
) -> QuantOption:
    """
    Sélectionne la quantization à télécharger.

    requested fourni → on l'exige (erreur listant les quants dispo si absente).
    sinon            → la plus grosse qui tient dans `available_gb` (× FIT_MARGIN de marge).
                       Si rien ne tient, la plus petite, avec un avertissement.
    """
    if not options:
        raise ValueError("Aucun fichier .gguf trouvé dans ce repo.")

    if requested is not None:
        key = requested.upper()
        if key not in options:
            dispo = ", ".join(sorted(options))
            raise ValueError(f"Quantization '{requested}' absente. Disponibles : {dispo}")
        return options[key]

    by_size = sorted(options.values(), key=lambda o: o.total_size)
    fitting = [o for o in by_size if (o.total_size / 1024**3) * FIT_MARGIN <= available_gb]
    if fitting:
        return fitting[-1]  # la plus grosse qui tient

    smallest = by_size[0]
    print(
        f"⚠ Aucune quantization ne tient dans {available_gb:.1f} Go ; "
        f"téléchargement de la plus petite ({smallest.quant}, "
        f"{smallest.total_size / 1024**3:.1f} Go) — l'inférence débordera sur CPU.",
        file=sys.stderr,
    )
    return smallest


def best_fitting_quant(
    options: dict[str, QuantOption], available_gb: float
) -> tuple[QuantOption, bool]:
    """
    Retourne (meilleure quant, tient_dans_la_mémoire).
    Si une quant tient, la plus grosse qui tient (meilleure qualité). Sinon la plus
    petite, avec `fits=False`. Ne lève pas et n'affiche rien (usage : recherche).
    """
    by_size = sorted(options.values(), key=lambda o: o.total_size)
    fitting = [o for o in by_size if (o.total_size / 1024**3) * FIT_MARGIN <= available_gb]
    if fitting:
        return fitting[-1], True
    return by_size[0], False


# ---------------------------------------------------------------------------
# Réseau
# ---------------------------------------------------------------------------

def _auth_headers() -> dict[str, str]:
    headers = {"User-Agent": _USER_AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_gguf_files(repo: str, revision: str = "main") -> list[GGUFFile]:
    """Liste les .gguf d'un repo via l'API tree de Hugging Face (publique)."""
    url = f"{HF_BASE}/api/models/{repo}/tree/{revision}?recursive=1"
    req = urllib.request.Request(url, headers=_auth_headers())
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            entries = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise ValueError(
                f"Repo '{repo}' gated ou privé (401). Exportez HF_TOKEN pour y accéder."
            )
        if e.code == 404:
            raise ValueError(f"Repo '{repo}' introuvable sur Hugging Face (404).")
        raise

    files: list[GGUFFile] = []
    for e in entries:
        if e.get("type") != "file" or not e.get("path", "").endswith(".gguf"):
            continue
        lfs = e.get("lfs") or {}
        files.append(GGUFFile(
            path=e["path"],
            size=int(lfs.get("size") or e.get("size") or 0),
            sha256=lfs.get("oid"),  # SHA256 uniquement pour les fichiers LFS
        ))
    return files


def _download_one(repo: str, f: GGUFFile, dest: Path, revision: str = "main") -> None:
    """Télécharge un fichier avec reprise, progression et vérification SHA256."""
    if dest.exists() and dest.stat().st_size == f.size:
        print(f"  déjà présent : {dest.name}")
        return

    url = f"{HF_BASE}/{repo}/resolve/{revision}/{f.path}"
    part = dest.with_name(dest.name + ".part")
    existing = part.stat().st_size if part.exists() else 0

    headers = _auth_headers()
    if existing:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # range déjà couvert → fichier complet
            part.rename(dest)
            return
        raise

    total = f.size or (existing + int(resp.headers.get("Content-Length", 0)))
    mode = "ab" if (existing and resp.status == 206) else "wb"
    if mode == "wb":
        existing = 0  # le serveur a ignoré le Range : on repart de zéro

    downloaded = existing
    is_tty = sys.stderr.isatty()
    step = 1 if is_tty else 10   # maj à chaque % sur un terminal, tous les 10 % si capturé
    last_pct = -step
    with resp, open(part, mode) as out:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            pct = int(100 * downloaded / total) if total else 0
            if pct >= last_pct + step or downloaded == total:
                last_pct = pct
                _progress(dest.name, downloaded, total, is_tty)
    if is_tty:
        sys.stderr.write("\n")

    if f.sha256:
        actual = _sha256(part)
        if actual != f.sha256:
            part.unlink(missing_ok=True)
            raise ValueError(
                f"Intégrité invalide pour {f.path} : SHA256 attendu {f.sha256}, obtenu {actual}."
            )

    part.rename(dest)


def _progress(name: str, done: int, total: int, is_tty: bool) -> None:
    if total <= 0:
        msg = f"{name} : {done / 1024**2:.0f} Mo"
    else:
        msg = f"{name} : {done / 1024**2:.0f} / {total / 1024**2:.0f} Mo ({100 * done / total:.0f} %)"
    if is_tty:
        sys.stderr.write(f"\r  {msg}")   # ligne réécrite en place sur un terminal
        sys.stderr.flush()
    else:
        sys.stderr.write(f"  {msg}\n")   # lignes distinctes si la sortie est capturée


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load(
    ref: str,
    quant: str | None = None,
    dest_dir: str | Path | None = None,
    include_mmproj: bool = False,
) -> Path:
    """
    Télécharge un modèle GGUF depuis Hugging Face et retourne le chemin local
    du fichier principal (le 1er shard pour un modèle splitté).
    """
    from .hardware import detect_hardware

    repo, ref_quant = parse_ref(ref)
    quant = quant or ref_quant  # --quant CLI prioritaire sur la forme repo:QUANT

    files = fetch_gguf_files(repo)
    options = group_by_quant(files)

    mmproj = select_mmproj(files)
    if mmproj and include_mmproj:
        print(
            f"Projecteur multimodal : {Path(mmproj.path).name} "
            f"({mmproj.size / 1024**3:.1f} Go)",
            file=sys.stderr,
        )
    elif mmproj:
        print(
            "⚠ Modèle multimodal (mmproj détecté). Ajoutez --mmproj pour télécharger "
            "aussi le projecteur. Le runtime image/audio n'est pas encore branché.",
            file=sys.stderr,
        )

    available_gb = detect_hardware().gpu_memory_gb
    option = choose_quant(options, available_gb, requested=quant)

    dest = Path(dest_dir) if dest_dir else models_dir()
    dest.mkdir(parents=True, exist_ok=True)

    print(
        f"Repo    : {repo}\n"
        f"Quant   : {option.quant} ({option.total_size / 1024**3:.1f} Go"
        f"{', %d shards' % len(option.files) if len(option.files) > 1 else ''})\n"
        f"Vers    : {dest}"
    )

    for f in option.files:
        _download_one(repo, f, dest / Path(f.path).name)
    if include_mmproj and mmproj:
        _download_one(repo, mmproj, dest / Path(mmproj.path).name)

    main_file = dest / Path(option.files[0].path).name
    print(f"✓ {main_file}")
    return main_file


# ---------------------------------------------------------------------------
# Recherche / découverte
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    repo: str
    downloads: int
    quant: str            # meilleure quant pour ce hardware (ou la plus petite si rien ne tient)
    size_gb: float        # taille de cette quant
    fits: bool            # tient dans la mémoire dispo ?
    multimodal: bool = False  # repo avec mmproj → c3po ne charge que la partie texte


def search_repos(query: str, limit: int = 20) -> list[dict]:
    """Cherche des repos GGUF sur Hugging Face, triés par téléchargements."""
    params = urllib.parse.urlencode({
        "search": query,
        "filter": "gguf",
        "sort": "downloads",
        "direction": "-1",
        "limit": limit,
    })
    req = urllib.request.Request(f"{HF_BASE}/api/models?{params}", headers=_auth_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_eligible(
    query: str,
    available_gb: float,
    limit: int = 20,
) -> list[SearchResult]:
    """
    Cherche des modèles GGUF sur HF et, pour chacun, détermine la meilleure quant
    vis-à-vis de `available_gb`. Les appels `tree` (un par repo) sont parallélisés.
    Résultats triés : éligibles d'abord, puis par nb de téléchargements.
    """
    repos = search_repos(query, limit)

    def inspect(meta: dict) -> SearchResult | None:
        repo = meta.get("id", "")
        try:
            files = fetch_gguf_files(repo)
        except Exception:
            return None  # repo gated / sans tree accessible → on l'ignore
        options = group_by_quant(files)
        if not options:
            return None
        best, fits = best_fitting_quant(options, available_gb)
        return SearchResult(
            repo=repo,
            downloads=int(meta.get("downloads", 0)),
            quant=best.quant,
            size_gb=round(best.total_size / 1024**3, 1),
            fits=fits,
            multimodal=has_mmproj(files),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [r for r in pool.map(inspect, repos) if r is not None]

    results.sort(key=lambda r: (not r.fits, -r.downloads))
    return results

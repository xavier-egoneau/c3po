"""Solvers pour l'éval : adaptent un modèle de chat en Solver (prompt, workdir -> fichiers).

- `make_oneshot_solver(chat)` : la baseline **BRUTE**. Un seul appel modèle, zéro outil,
  zéro boucle, zéro optimisation de contexte. Le modèle doit émettre ses fichiers dans des
  blocs ```lang path=<chemin>```. On parse et on écrit. C'est le **plancher** : la couche
  agentic devra faire mieux que ça, et l'écart se mesure sur le jeu d'éval.
- `engine_chat(model_path, ...)` : adaptateur `ChatFn` au-dessus du runtime llm_runtime
  (llama.cpp). Importé paresseusement (llama_cpp est lourd et optionnel).

Tous les fichiers sont écrits **dans le workdir fourni** (sous test/), jamais ailleurs.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

Message = dict[str, str]
ChatFn = Callable[[list[Message]], str]
Solver = Callable[[str, Path], None]

_FILE_EXT = r"(?:py|js|mjs|cjs|ts|html|htm|css|json|txt|md|csv|sh)"
_BLOCK = re.compile(r"```([^\n]*)\n(.*?)```", re.S)
_PATH_IN_HEADER = re.compile(rf"(?:path|file|filename|name)\s*=\s*[\"']?([\w./-]+\.{_FILE_EXT})", re.I)
_BARE_PATH = re.compile(rf"\b([\w./-]+\.{_FILE_EXT})\b")
_BODY_MARKER = re.compile(rf"^\s*(?://|#|<!--)\s*(?:path|file)\s*:\s*([\w./-]+\.{_FILE_EXT})", re.I)

_INSTRUCTION = (
    "Tu produis le(s) fichier(s) demandés, complets et prêts à l'emploi.\n"
    "Pour CHAQUE fichier, utilise un bloc de code dont l'en-tête précise le chemin, "
    "exactement ainsi :\n"
    "```python path=solution.py\n...contenu intégral du fichier...\n```\n"
    "Donne le contenu COMPLET de chaque fichier (pas d'extrait, pas de '...'). "
    "N'écris aucune explication hors des blocs."
)


def parse_file_blocks(text: str) -> dict[str, str]:
    """Extrait {chemin: contenu} des blocs de code d'une réponse modèle."""
    files: dict[str, str] = {}
    for header, body in _BLOCK.findall(text):
        path = _path_from_header(header) or _path_from_body(body)
        if path:
            files[path] = _strip_leading_marker(body)
    return files


def _path_from_header(header: str) -> str | None:
    match = _PATH_IN_HEADER.search(header) or _BARE_PATH.search(header)
    return match.group(1) if match else None


def _path_from_body(body: str) -> str | None:
    first = body.lstrip().splitlines()[0] if body.strip() else ""
    match = _BODY_MARKER.match(first)
    return match.group(1) if match else None


def _strip_leading_marker(body: str) -> str:
    lines = body.splitlines()
    if lines and _BODY_MARKER.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip("\n") + "\n"


def _guess_filename(prompt: str) -> str | None:
    """Nom de fichier cible dans le prompt, en évitant les fichiers de test
    (on ne route jamais un livrable vers un test_*.py)."""
    names = _BARE_PATH.findall(prompt)
    non_test = [n for n in names if not Path(n).name.startswith("test_")]
    chosen = non_test or names
    return chosen[0] if chosen else None


def write_files(workdir: Path, files: dict[str, str]) -> list[str]:
    """Écrit les fichiers dans le workdir. Ignore tout chemin qui s'en échapperait."""
    workdir = Path(workdir).resolve()
    written: list[str] = []
    for raw_path, content in files.items():
        target = (workdir / raw_path).resolve()
        if target != workdir and workdir not in target.parents:
            continue  # tentative d'évasion hors workdir : ignorée
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(str(target.relative_to(workdir)))
    return written


def make_oneshot_solver(chat: ChatFn) -> Solver:
    """Baseline brute : un appel modèle, on parse les fichiers émis, on les écrit."""

    def _solve(prompt: str, workdir: Path) -> None:
        workdir = Path(workdir)
        reply = chat([
            {"role": "system", "content": _INSTRUCTION},
            {"role": "user", "content": _initial_user_message(prompt, workdir)},
        ])
        _apply_reply(reply, prompt, workdir)

    return _solve


def make_agent_solver(
    chat: ChatFn,
    max_rounds: int = 3,
    self_test: bool = False,
    vision_check: bool = False,
    vision_judge: ChatFn | None = None,
    trace: list | None = None,
) -> Solver:
    """Couche agentic GÉNÉRIQUE : émet les fichiers, les **vérifie en les exécutant**
    (compile, tests, smoke-run, navigateur headless), et **renvoie l'erreur concrète au
    modèle** pour qu'il corrige — boucle bornée. Aucune logique propre à une tâche.

    Endurance/persévérance :
    - `max_rounds` = budget de tours (1 émission + max_rounds-1 corrections).
    - **anti-blocage** : si la même erreur revient à l'identique d'un tour à l'autre, on
      arrête (persévérer aveuglément ne sert à rien — il manque un signal neuf).
    - `trace` (liste optionnelle) : on y enregistre l'état de chaque tour
      (`{round, ok, issues, stuck?, exhausted?}`) pour MESURER la trajectoire.

    `self_test` (OFF, mesuré régressif). `vision_check` (OFF, mesuré régressif aussi :
    sur gemma le juge visuel statique a fait chuter fe_markdown 100%->29% — le rendu initial
    d'un éditeur est vide par design, la vision le flagge à tort -> faux « fix »). Les seuls
    signaux fiables restent DÉTERMINISTES (crash/syntaxe/pytest/erreur navigateur).
    La vérif est indépendante des checkers de l'éval.
    """

    def _solve(prompt: str, workdir: Path) -> None:
        workdir = Path(workdir)
        messages: list[Message] = [
            {"role": "system", "content": _INSTRUCTION},
            {"role": "user", "content": _initial_user_message(prompt, workdir)},
        ]
        reply = chat(messages)
        _apply_reply(reply, prompt, workdir)
        state: dict[str, str] = {}
        last_issues: str | None = None

        try:
            for round_no in range(1, max_rounds):
                issues = _verify_workdir(workdir)
                if not issues and self_test:
                    issues = _self_test(chat, workdir, state)
                if not issues and vision_check:
                    issues = _verify_vision(workdir, prompt, chat, vision_judge)
                if trace is not None:
                    trace.append({"round": round_no, "ok": not issues, "issues": issues})
                if not issues:
                    return  # convergé
                if issues == last_issues:
                    if trace is not None:
                        trace[-1]["stuck"] = True
                    return  # blocage : même erreur, pas de progrès -> inutile de continuer
                last_issues = issues
                messages.extend(
                    [
                        {"role": "assistant", "content": reply},
                        {
                            "role": "user",
                            "content": (
                                "Des problèmes subsistent quand j'exécute ton rendu. Corrige le "
                                "CODE (pas les fichiers de test) et redonne les fichiers COMPLETS "
                                "dans le même format.\n\n"
                                f"Problèmes détectés :\n{issues}\n\n"
                                f"Fichiers actuels :\n{_render_current_files(workdir)}"
                            ),
                        },
                    ]
                )
                reply = chat(messages)
                _apply_reply(reply, prompt, workdir)
            if trace is not None:  # budget épuisé : état final
                final = _verify_workdir(workdir)
                trace.append({"round": max_rounds, "ok": not final, "issues": final, "exhausted": True})
        finally:
            _cleanup_selfcheck(workdir)

    return _solve


_SELFTEST_INSTRUCTION = (
    "Écris UN fichier selfcheck_main.py qui éprouve le comportement du code ci-dessous avec "
    "des assertions : importe le code, couvre quelques cas normaux ET surtout des cas limites "
    "représentatifs où il pourrait se tromper. Juste des `assert` + un bloc exécutable "
    "(`if __name__ == \"__main__\"`), pas de framework. Il doit lever AssertionError si le code "
    "est faux. Ne teste pas l'évident."
)


def _self_test(chat: ChatFn, workdir: Path, state: dict[str, str]) -> str:
    """Le modèle écrit ses tests-sanité (une seule fois), on les exécute. Renvoie un texte
    d'erreur seulement si SES tests échouent par AssertionError (= son code est faux selon
    lui-même). Un selfcheck cassé (import/syntaxe) est inconcluant : ignoré."""
    name = "selfcheck_main.py"
    if "generated" not in state:
        state["generated"] = "1"
        reply = chat(
            [
                {"role": "system", "content": _INSTRUCTION},
                {"role": "user", "content": _SELFTEST_INSTRUCTION + "\n\nCode actuel :\n" + _render_current_files(workdir)},
            ]
        )
        content = parse_file_blocks(reply).get(name) or _first_python_block(reply)
        if content:
            (workdir / name).write_text(content, encoding="utf-8")
            state["selfcheck"] = name
    if "selfcheck" not in state:
        return ""
    proc = subprocess.run([sys.executable, state["selfcheck"]], cwd=workdir, capture_output=True, text=True)
    if proc.returncode == 0:
        return ""
    if "AssertionError" in proc.stderr:
        return "Tes propres tests de sanité échouent (corrige le CODE, pas les tests) :\n" + proc.stderr.strip()[-500:]
    return ""


def _first_python_block(reply: str) -> str | None:
    for header, body in _BLOCK.findall(reply):
        if "python" in header.lower() or not header.strip():
            return _strip_leading_marker(body)
    return None


def _cleanup_selfcheck(workdir: Path) -> None:
    for path in Path(workdir).glob("selfcheck_*.py"):
        path.unlink(missing_ok=True)


def _initial_user_message(prompt: str, workdir: Path) -> str:
    """Prompt + contexte du dossier de travail (fichiers déjà présents). Sans ça, on
    demande au modèle de modifier des fichiers qu'il ne voit pas — baseline injuste.
    Version BRUTE : dump TOUT le dossier (sature un petit modèle si le repo est gros)."""
    existing = _render_current_files(Path(workdir))
    if existing.strip():
        return prompt + "\n\nFichiers déjà présents dans le dossier de travail :\n" + existing
    return prompt


def make_context_solver(chat: ChatFn, max_files: int = 4) -> Solver:
    """Égaliseur par le CONTEXTE : au lieu de dumper tout le dossier, on SÉLECTIONNE les
    fichiers pertinents (nommés dans le prompt + clôture de leurs imports) et on ne donne
    que ceux-là. Sur un gros dossier, ça évite de noyer un petit modèle. Générique."""

    def _solve(prompt: str, workdir: Path) -> None:
        workdir = Path(workdir)
        selected = _select_context_files(prompt, workdir, max_files)
        message = prompt
        if selected:
            message += "\n\nFichiers pertinents :\n" + _render_files(workdir, selected, keywords=_prompt_keywords(prompt))
        reply = chat([{"role": "system", "content": _INSTRUCTION}, {"role": "user", "content": message}])
        _apply_reply(reply, prompt, workdir)

    return _solve


def _select_context_files(prompt: str, workdir: Path, max_files: int) -> list[str]:
    workdir = Path(workdir)
    all_files = {
        p.relative_to(workdir).as_posix(): p
        for p in sorted(workdir.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("selfcheck_")
    }
    selected: list[str] = [rel for rel in all_files if Path(rel).name in prompt]
    # Clôture des imports : un fichier sélectionné en tire d'autres réellement utiles.
    frontier = list(selected)
    while frontier and len(selected) < max_files * 3:
        current = workdir / frontier.pop()
        if current.suffix != ".py":
            continue
        try:
            tree = ast.parse(current.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module.split(".")[0]]
            for module in modules:
                candidate = f"{module}.py"
                if candidate in all_files and candidate not in selected:
                    selected.append(candidate)
                    frontier.append(candidate)

    # Pertinence par CONTENU : si le fichier utile n'est pas nommé dans le prompt, on le
    # retrouve via les mots-clés/identifiants du prompt présents dans le CONTENU des fichiers.
    if len(selected) < max_files:
        keywords = _prompt_keywords(prompt)
        scored: list[tuple[int, int, str]] = []
        for rel, path in all_files.items():
            if rel in selected:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace").lower()
            except OSError:
                continue
            score = sum(1 for kw in keywords if kw in content)
            if score:
                scored.append((score, len(content), rel))
        scored.sort(key=lambda item: (-item[0], item[1]))  # plus pertinent, puis plus court
        for _, _, rel in scored:
            if len(selected) >= max_files:
                break
            selected.append(rel)
    return selected[:max_files]


_KW_STOP = {
    "dans", "pour", "avec", "cette", "fonction", "fichier", "resultat", "valeur", "valeurs",
    "python", "depot", "sans", "casser", "reste", "trouve", "corrige", "corriger", "milieu",
    "elements", "taille", "liste", "listes", "paire", "faux", "renvoie", "contient", "function",
    "file", "value", "values", "return", "depuis", "elle", "deux", "leur", "code", "bug",
}


def _prompt_keywords(prompt: str) -> set[str]:
    """Identifiants/mots-clés significatifs du prompt (pour la pertinence par contenu).
    Priorise les termes entre `backticks` (souvent des identifiants de code)."""
    keywords: set[str] = set()
    for span in re.findall(r"`([^`]+)`", prompt):
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", span):
            keywords.add(token.lower())
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", prompt):
        keywords.add(token.lower())
    return {k for k in keywords if len(k) >= 4 and k not in _KW_STOP}


def _render_files(workdir: Path, paths: list[str], total_budget: int = 8000, keywords: set[str] | None = None) -> str:
    """Rend les fichiers sélectionnés dans un BUDGET total (caractères). On montre chaque
    fichier ENTIER tant qu'il tient (crucial pour une tâche de réécriture : un fragment ferait
    réémettre un fichier incomplet). Seul le DÉBORDEMENT est compacté autour des mots-clés."""
    workdir = Path(workdir)
    parts: list[str] = []
    used = 0
    for rel in paths:
        path = workdir / rel
        if not path.is_file():
            continue
        remaining = total_budget - used
        if remaining <= 200:
            break
        body = path.read_text(encoding="utf-8", errors="replace")
        if len(body) > remaining:
            body = _compact_around_keywords(body, keywords or set(), remaining)
        parts.append(f"```text path={rel}\n{body}\n```")
        used += len(body)
    return "\n\n".join(parts)


def _compact_around_keywords(content: str, keywords: set[str], max_chars: int) -> str:
    """Compacte un gros fichier en gardant la TÊTE (imports/structure) + les FENÊTRES autour
    des mots-clés du prompt — au lieu de tronquer bêtement la tête (qui coupe le bug s'il est
    plus bas). Sans mots-clés : repli sur la troncature tête."""
    if not keywords:
        return content[: max_chars] + "\n[...tronqué...]"
    lines = content.splitlines()
    keep = [False] * len(lines)
    for i in range(min(len(lines), 12)):  # tête : docstring/imports
        keep[i] = True
    for i, line in enumerate(lines):
        low = line.lower()
        if any(kw in low for kw in keywords):
            for j in range(max(0, i - 3), min(len(lines), i + 14)):  # fenêtre autour du hit
                keep[j] = True

    out: list[str] = []
    total = 0
    gap = False
    for i, line in enumerate(lines):
        if keep[i]:
            if gap:
                out.append("# [...]")
                gap = False
            out.append(line)
            total += len(line) + 1
            if total >= max_chars:
                out.append("# [...tronqué...]")
                break
        else:
            gap = True
    return "\n".join(out)


def _apply_reply(reply: str, prompt: str, workdir: Path) -> None:
    files = parse_file_blocks(reply)
    if not files:
        # Repli pour modèle faible : blocs sans `path=`. On route vers le fichier que le
        # prompt demande d'écrire (jamais un test_*.py : un livrable n'écrase pas un test).
        blocks = _BLOCK.findall(reply)
        target = _guess_filename(prompt)
        if blocks and target:
            files = {target: _strip_leading_marker(blocks[0][1])}
    write_files(workdir, files)


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _uses_cli_args(src: str) -> bool:
    return "argv" in src or "argparse" in src or "input(" in src


def _verify_workdir(workdir: Path) -> str:
    """Vérifs génériques et structurelles. Renvoie un texte d'erreurs, ou "" si propre."""
    workdir = Path(workdir)
    issues: list[str] = []
    py_files = [p for p in sorted(workdir.rglob("*.py")) if "__pycache__" not in p.parts]

    for path in py_files:
        try:
            ast.parse(_src(path))
        except SyntaxError as exc:
            issues.append(f"{path.relative_to(workdir)}: SyntaxError ligne {exc.lineno}: {exc.msg}")
    if issues:
        return "\n".join(issues)

    if list(workdir.glob("test_*.py")):
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"], cwd=workdir, capture_output=True, text=True
        )
        if proc.returncode != 0:
            issues.append("pytest échoue :\n" + proc.stdout.strip()[-600:])

    mains = [
        p for p in py_files
        if "__main__" in _src(p) and not p.name.startswith("test_") and not _uses_cli_args(_src(p))
    ]
    if len(mains) == 1 and not issues:
        proc = subprocess.run([sys.executable, mains[0].name], cwd=workdir, capture_output=True, text=True)
        if proc.returncode != 0:
            issues.append(f"{mains[0].name} plante :\n" + proc.stderr.strip()[-400:])

    web = _verify_web(workdir)
    if web:
        issues.append(web)
    return "\n".join(issues)


def _verify_web(workdir: Path) -> str:
    """Vérif WEB via navigateur headless (Playwright) : charge chaque .html, clique chaque
    bouton, et remonte les erreurs JS (pageerror / console.error) — câble les crashs du
    type « mauvais id », « null.addEventListener », handler qui jette. Pendant web du check
    Python. Dépendance OPTIONNELLE : si Playwright absent, on saute proprement."""
    workdir = Path(workdir).resolve()  # as_uri() exige un chemin absolu
    html_files = [p for p in sorted(workdir.rglob("*.html")) if "__pycache__" not in p.parts]
    if not html_files:
        return ""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return ""

    issues: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for html in html_files:
                errors: list[str] = []
                page = browser.new_page()
                page.on("pageerror", lambda exc, acc=errors: acc.append(f"pageerror: {exc}"))
                page.on(
                    "console",
                    lambda msg, acc=errors: acc.append(f"console.error: {msg.text}") if msg.type == "error" else None,
                )
                try:
                    page.goto(html.as_uri(), wait_until="load", timeout=5000)
                    for button in page.query_selector_all("button"):
                        try:
                            button.click(timeout=500)
                        except Exception:  # noqa: BLE001 - un bouton non cliquable n'est pas l'erreur cherchée
                            pass
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"chargement: {exc}")
                page.close()
                if errors:
                    unique = " | ".join(dict.fromkeys(errors))
                    issues.append(f"{html.relative_to(workdir)} (navigateur) : {unique[:400]}")
        finally:
            browser.close()
    return "\n".join(issues)


def _screenshot_html(html_path: Path) -> Path | None:
    """Rend une page en navigateur headless et capture un PNG. None si Playwright absent."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    shot = html_path.parent / "_vision_shot.png"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(html_path.resolve().as_uri(), wait_until="load", timeout=5000)
            page.screenshot(path=str(shot))
        except Exception:  # noqa: BLE001
            return None
        finally:
            browser.close()
    return shot


def _drive_and_screenshot(html_path: Path) -> Path | None:
    """drive-then-look : rend la page, PILOTE l'UI génériquement (remplit textareas avec
    un échantillon markdown, inputs texte avec une valeur, clique les boutons) PUIS capture.
    Sans ça, l'état dépendant d'interaction (aperçu markdown, liste todo) reste invisible et
    le juge se trompe. Générique — aucune règle par tâche."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    sample_md = "# Grand Titre\n**texte en gras** *et italique*\n- premier point\n- second point"
    shot = html_path.parent / "_vision_shot.png"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(html_path.resolve().as_uri(), wait_until="load", timeout=5000)
            for textarea in page.query_selector_all("textarea"):
                try:
                    textarea.fill(sample_md)
                except Exception:  # noqa: BLE001
                    pass
            for field in page.query_selector_all("input[type=text], input:not([type])"):
                try:
                    field.fill("Élément de test")
                except Exception:  # noqa: BLE001
                    pass
            for button in page.query_selector_all("button"):
                try:
                    button.click(timeout=500)
                except Exception:  # noqa: BLE001
                    pass
            page.wait_for_timeout(200)
            page.screenshot(path=str(shot))
        except Exception:  # noqa: BLE001
            return None
        finally:
            browser.close()
    return shot


def _verify_vision(workdir: Path, prompt: str, chat: ChatFn, judge: ChatFn | None = None) -> str:
    """Signal SÉMANTIQUE pour le web : DRIVE-then-look (piloter l'UI puis screenshot) ->
    le sidecar vision décrit l'état rendu -> un JUGE capable dit si l'intention est satisfaite.
    Capte le « câblé mais inerte / liste vide / markdown non rendu » que le crash-check rate.

    `judge` (sinon `chat`) : un modèle CAPABLE (mesuré : Qwen3-14B/Phi-4 fiables ; 2B/coder non).
    Dépendances optionnelles (Playwright + sidecar vision) : si absentes, on saute. Tout tourne
    en SUBPROCESS (swap multimodal/multi-modèle in-process = crash CUDA)."""
    judge = judge or chat
    workdir = Path(workdir).resolve()
    html_files = [p for p in sorted(workdir.rglob("*.html")) if "__pycache__" not in p.parts]
    if not html_files:
        return ""
    try:
        from agent.structured import extract_json_object
        from agent.vision.gemma4 import observe_image_subprocess
    except ImportError:
        return ""

    shot = _drive_and_screenshot(html_files[0])
    if shot is None:
        return ""
    try:
        observation = observe_image_subprocess(shot)
    except Exception:  # noqa: BLE001 - vision indisponible/échec -> pas de signal, on ne bloque pas
        return ""
    finally:
        shot.unlink(missing_ok=True)

    description = observation.get("parsed") or observation.get("raw") or ""
    reply = judge(
        [
            {"role": "system", "content": "Tu juges si un rendu web satisfait une tâche, d'après une description visuelle neutre. Réponds uniquement en JSON."},
            {
                "role": "user",
                "content": (
                    f"Tâche demandée :\n{prompt}\n\n"
                    f"Description automatique (vision) du rendu actuel :\n{description}\n\n"
                    "Les éléments attendus sont-ils visiblement présents ET peuplés (listes non vides, "
                    "valeurs affichées) ? Sois strict : une liste vide ou un élément manquant = non satisfait.\n"
                    'Réponds : {"ok": true|false, "manque": "ce qui manque visiblement, sinon vide"}'
                ),
            },
        ]
    )
    parsed = extract_json_object(reply)
    if isinstance(parsed, dict) and parsed.get("ok") is False:
        manque = str(parsed.get("manque", "")).strip()
        if manque:
            return "Vérif visuelle du rendu : " + manque[:200]
    return ""


def _render_current_files(workdir: Path, max_chars: int = 4000) -> str:
    workdir = Path(workdir)
    exts = {".py", ".js", ".mjs", ".html", ".htm", ".css", ".json", ".txt", ".csv"}
    parts: list[str] = []
    for path in sorted(workdir.rglob("*")):
        if path.name.startswith("selfcheck_"):
            continue  # scaffolding interne : on ne le montre pas au modèle qui corrige le code
        if path.is_file() and path.suffix in exts and "__pycache__" not in path.parts:
            body = _src(path)
            if len(body) > max_chars:
                body = body[:max_chars] + "\n[...tronqué...]"
            parts.append(f"```text path={path.relative_to(workdir).as_posix()}\n{body}\n```")
    return "\n\n".join(parts)


def engine_chat(
    model_path: str | Path,
    *,
    n_ctx: int = 8192,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    **engine_kwargs,
) -> ChatFn:
    """Adaptateur ChatFn sur le runtime. `chat.close()` libère le modèle."""
    from llm_runtime.engine import Engine

    engine = Engine(model_path, n_ctx=n_ctx, **engine_kwargs)

    def chat(messages: list[Message]) -> str:
        resp = engine.chat(messages, max_tokens=max_tokens, temperature=temperature)
        return resp["choices"][0]["message"]["content"]

    chat.close = engine.close  # type: ignore[attr-defined]
    return chat


def chat_subprocess(
    model: str | Path,
    messages: list[Message],
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: float = 900,
) -> str:
    """Génère via un subprocess éphémère (un modèle par process). Évite le crash CUDA du
    swap in-process : le process appelant ne tient AUCUN modèle GPU, chaque appel est isolé."""
    import json as _json
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "req.json"
        out = Path(tmp) / "out.txt"
        req.write_text(
            _json.dumps({"model": str(model), "messages": messages, "temperature": temperature, "max_tokens": max_tokens}),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, "-m", "agent.eval.chat_worker", str(req), str(out)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"chat_worker rc={proc.returncode} : {proc.stderr.strip()[-300:]}")
        return out.read_text(encoding="utf-8") if out.exists() else ""


def make_subprocess_chat(model: str | Path, *, temperature: float = 0.2, max_tokens: int = 4096) -> ChatFn:
    """ChatFn drop-in où chaque appel tourne dans un subprocess éphémère (VRAM/CUDA-safe).
    Permet d'orchestrer plusieurs modèles (petit solver + gros juge) sans swap in-process."""

    def chat(messages: list[Message]) -> str:
        return chat_subprocess(model, messages, temperature=temperature, max_tokens=max_tokens)

    return chat

"""
Indexeur de modèles : trouve tous les GGUFs disponibles,
qu'ils viennent d'Ollama ou de fichiers locaux.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path


OLLAMA_MANIFESTS = Path.home() / ".ollama" / "models" / "manifests"
OLLAMA_BLOBS     = Path.home() / ".ollama" / "models" / "blobs"

# Type MIME du blob qui contient les poids du modèle
OLLAMA_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"


@dataclass
class ModelInfo:
    name: str           # ex: "gemma4:latest" ou "mistral-7b-q4.gguf"
    path: Path          # chemin absolu vers le fichier (blob ou .gguf)
    size_gb: float      # taille en Go
    source: str         # "ollama" ou "local"

    def fits_in(self, memory_gb: float) -> bool:
        """Est-ce que ce modèle tient dans la mémoire disponible ?"""
        # On ajoute ~10% de marge pour le KV cache
        return self.size_gb * 1.1 <= memory_gb

    def __str__(self) -> str:
        return (
            f"{self.name:<30} {self.size_gb:>5.1f} Go  [{self.source}]  {self.path}"
        )


# ---------------------------------------------------------------------------
# Indexation Ollama
# ---------------------------------------------------------------------------

def _scan_ollama() -> list[ModelInfo]:
    """Parcourt les manifests Ollama et retourne les modèles disponibles."""
    models = []

    if not OLLAMA_MANIFESTS.exists():
        return models

    # Structure : manifests/<registry>/<namespace>/<name>/<tag>
    for manifest_path in OLLAMA_MANIFESTS.rglob("*"):
        if not manifest_path.is_file():
            continue

        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        layers = manifest.get("layers", [])
        for layer in layers:
            if layer.get("mediaType") != OLLAMA_MODEL_MEDIA_TYPE:
                continue

            digest = layer.get("digest", "")          # "sha256:abc123..."
            size   = layer.get("size", 0)

            # Le fichier blob remplace ":" par "-"
            blob_name = digest.replace(":", "-")
            blob_path = OLLAMA_BLOBS / blob_name

            if not blob_path.exists():
                continue

            # Reconstruit un nom lisible depuis le chemin du manifest
            # ex: registry.ollama.ai/library/gemma4/latest → gemma4:latest
            parts = manifest_path.parts
            try:
                # On cherche "library" ou l'équivalent namespace
                lib_idx = next(
                    i for i, p in enumerate(parts) if p in ("library", "namespace")
                )
                name = f"{parts[lib_idx + 1]}:{parts[lib_idx + 2]}"
            except (StopIteration, IndexError):
                name = manifest_path.name

            models.append(ModelInfo(
                name=name,
                path=blob_path,
                size_gb=size / (1024 ** 3),
                source="ollama",
            ))

    return models


# ---------------------------------------------------------------------------
# Indexation fichiers locaux
# ---------------------------------------------------------------------------

def _scan_local(directories: list[Path]) -> list[ModelInfo]:
    """Scanne des répertoires locaux à la recherche de fichiers .gguf."""
    models = []
    for directory in directories:
        if not directory.exists():
            continue
        for gguf_path in directory.rglob("*.gguf"):
            size_gb = gguf_path.stat().st_size / (1024 ** 3)
            models.append(ModelInfo(
                name=gguf_path.stem,
                path=gguf_path,
                size_gb=size_gb,
                source="local",
            ))
    return models


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def list_models(
    local_dirs: list[str | Path] | None = None,
) -> list[ModelInfo]:
    """
    Retourne tous les modèles disponibles (Ollama + fichiers locaux).

    local_dirs : répertoires supplémentaires à scanner pour des .gguf
    """
    models = _scan_ollama()

    extra_dirs = [Path(d) for d in (local_dirs or [])]
    models += _scan_local(extra_dirs)

    # Déduplique par chemin
    seen = set()
    unique = []
    for m in models:
        if m.path not in seen:
            seen.add(m.path)
            unique.append(m)

    return sorted(unique, key=lambda m: m.size_gb)


def find_model(
    query: str,
    local_dirs: list[str | Path] | None = None,
) -> ModelInfo:
    """
    Résout un nom de modèle (chemin direct ou sous-chaîne du nom) vers un ModelInfo.

    Lève ValueError si le modèle est introuvable ou si plusieurs modèles
    correspondent à la requête (ambiguïté).
    """
    p = Path(query)
    if p.exists():
        size_gb = p.stat().st_size / (1024 ** 3)
        return ModelInfo(name=p.stem, path=p, size_gb=size_gb, source="local")

    models = list_models(local_dirs)
    matches = [m for m in models if query.lower() in m.name.lower()]

    if not matches:
        raise ValueError(
            f"Modèle '{query}' introuvable. Utilisez 'c3po list' pour voir les modèles disponibles."
        )
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise ValueError(f"Plusieurs modèles correspondent à '{query}' : {names}. Précisez le nom.")

    return matches[0]


def best_model(
    available_memory_gb: float,
    local_dirs: list[str | Path] | None = None,
) -> ModelInfo | None:
    """
    Retourne le plus grand modèle qui tient dans la mémoire disponible.
    Utile pour choisir automatiquement sans intervention manuelle.
    """
    candidates = [m for m in list_models(local_dirs) if m.fits_in(available_memory_gb)]
    return candidates[-1] if candidates else None


if __name__ == "__main__":
    print("Modèles disponibles :\n")
    for m in list_models():
        print(" ", m)

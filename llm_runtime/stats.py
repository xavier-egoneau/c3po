"""
Statistiques d'un modèle.

Rassemble en un seul endroit :
  - les métadonnées du GGUF (architecture, quantization, nb de paramètres, etc.),
  - les paramètres d'inférence qui seront appliqués sur le hardware courant,
  - un petit benchmark réel (temps de chargement, time-to-first-token, tokens/s).

Le benchmark charge réellement le modèle et génère quelques tokens : les chiffres
reflètent donc le backend effectif (CUDA, Metal ou CPU) de la machine.
"""

from __future__ import annotations
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .hardware import detect_hardware


# Prompt de benchmark : assez long pour solliciter la génération, déterministe (temp 0).
_BENCH_PROMPT = "Explique en quelques phrases ce qu'est un compilateur."
_BENCH_MAX_TOKENS = 128


@dataclass
class ModelStats:
    # Fichier
    name: str
    path: str
    size_gb: float

    # Métadonnées GGUF
    architecture: str
    quantization: str
    n_params_b: float | None      # milliards de paramètres (None si non renseigné)
    n_ctx_train: int | None       # contexte max à l'entraînement
    n_embd: int | None
    n_layers: int | None
    n_vocab: int | None

    # Hardware + paramètres appliqués
    backend: str
    device: str
    n_gpu_layers: int
    n_threads: int
    n_ctx: int
    flash_attn: bool

    # Benchmark
    load_time_s: float
    ttft_s: float | None          # time-to-first-token
    gen_tps: float | None         # tokens/s en génération (régime établi)
    gen_tokens: int               # nb de tokens générés pendant le bench
    vram_used_mb: float | None    # VRAM consommée par le chargement (CUDA only)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_mem_used_mb() -> float | None:
    """VRAM utilisée sur le premier GPU Nvidia (None si pas de nvidia-smi)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return float(out.splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def _parse_quantization(name: str, metadata: dict[str, str]) -> str:
    """Déduit la quantization du nom de fichier (robuste), sinon du file_type GGUF."""
    m = re.search(r"(IQ\d[\w]*|Q\d[\w]*|BF16|F16|F32)", name, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return metadata.get("general.file_type", "?")


def _meta_int(metadata: dict[str, str], *keys: str) -> int | None:
    """Premier des `keys` présent dans les métadonnées, converti en int."""
    for k in keys:
        if k in metadata:
            try:
                return int(metadata[k])
            except (ValueError, TypeError):
                pass
    return None


def _benchmark(engine) -> tuple[float | None, float | None, int]:
    """
    Génère quelques tokens et mesure le time-to-first-token et la vitesse
    de génération en régime établi. Retourne (ttft_s, gen_tps, gen_tokens).
    """
    messages = [{"role": "user", "content": _BENCH_PROMPT}]
    t0 = time.time()
    t_first: float | None = None
    count = 0

    for chunk in engine.chat(
        messages, max_tokens=_BENCH_MAX_TOKENS, temperature=0.0, stream=True
    ):
        delta = chunk["choices"][0].get("delta", {}).get("content", "")
        if delta:
            if t_first is None:
                t_first = time.time()
            count += 1

    t_end = time.time()
    ttft = (t_first - t0) if t_first is not None else None
    # tokens/s sur la phase de génération seule (on exclut le 1er token / prompt eval)
    gen_tps = None
    if t_first is not None and count > 1 and t_end > t_first:
        gen_tps = (count - 1) / (t_end - t_first)
    return ttft, gen_tps, count


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def collect_stats(
    model_path: str | Path,
    n_ctx: int = 4096,
    n_gpu_layers: int | None = None,
    n_threads: int | None = None,
    flash_attn: bool | None = None,
) -> ModelStats:
    """
    Charge le modèle, lit ses métadonnées et lance un benchmark de génération.
    Les leviers (n_gpu_layers/n_threads/flash_attn) sont transmis à l'Engine ;
    les stats reflètent donc la configuration réellement appliquée.
    """
    from .engine import Engine

    model_path = Path(model_path)

    profile = detect_hardware()

    vram_before = _gpu_mem_used_mb()
    t0 = time.time()
    engine = Engine(
        model_path, n_ctx=n_ctx, profile=profile,
        n_gpu_layers=n_gpu_layers, n_threads=n_threads, flash_attn=flash_attn,
    )
    load_time = time.time() - t0
    vram_after = _gpu_mem_used_mb()
    vram_used = (vram_after - vram_before) if (vram_before is not None and vram_after is not None) else None

    llm = engine._llm
    metadata = dict(getattr(llm, "metadata", {}) or {})
    arch = metadata.get("general.architecture", "?")

    # nb de paramètres : clé GGUF standard (en unités → milliards)
    n_params_raw = _meta_int(metadata, "general.parameter_count")
    n_params_b = round(n_params_raw / 1e9, 2) if n_params_raw else None

    n_ctx_train = _meta_int(metadata, f"{arch}.context_length")
    n_layers = _meta_int(metadata, f"{arch}.block_count")

    # n_embd / n_vocab : via l'API llama_cpp si dispo, sinon métadonnées
    def _call(method_name: str, *meta_keys: str) -> int | None:
        method = getattr(llm, method_name, None)
        if callable(method):
            try:
                return int(method())
            except Exception:
                pass
        return _meta_int(metadata, *meta_keys)

    n_embd = _call("n_embd", f"{arch}.embedding_length")
    n_vocab = _call("n_vocab", f"{arch}.vocab_size")

    ttft, gen_tps, gen_tokens = _benchmark(engine)

    params = engine.params  # paramètres réellement appliqués (overrides inclus)
    return ModelStats(
        name=model_path.stem,
        path=str(model_path),
        size_gb=round(engine.size_gb, 2),
        architecture=arch,
        quantization=_parse_quantization(model_path.name, metadata),
        n_params_b=n_params_b,
        n_ctx_train=n_ctx_train,
        n_embd=n_embd,
        n_layers=n_layers,
        n_vocab=n_vocab,
        backend=profile.backend.value,
        device=profile.device_name,
        n_gpu_layers=params.n_gpu_layers,
        n_threads=params.n_threads,
        n_ctx=params.n_ctx,
        flash_attn=params.use_flash_attn,
        load_time_s=round(load_time, 2),
        ttft_s=round(ttft, 3) if ttft is not None else None,
        gen_tps=round(gen_tps, 1) if gen_tps is not None else None,
        gen_tokens=gen_tokens,
        vram_used_mb=round(vram_used, 0) if vram_used is not None else None,
    )


def format_stats(s: ModelStats) -> str:
    """Rendu lisible des stats pour la CLI."""
    def line(label: str, value) -> str:
        return f"  {label:<16}: {value}"

    gpu = "toutes" if s.n_gpu_layers == -1 else str(s.n_gpu_layers)
    params = f"{s.n_params_b} G" if s.n_params_b is not None else "?"
    ttft = f"{s.ttft_s * 1000:.0f} ms" if s.ttft_s is not None else "?"
    gen = f"{s.gen_tps} tok/s" if s.gen_tps is not None else "?"
    vram = f"{s.vram_used_mb:.0f} Mo" if s.vram_used_mb is not None else "n/a (pas de GPU Nvidia)"

    return "\n".join([
        f"── Modèle : {s.name} ──",
        line("Fichier", f"{s.size_gb} Go"),
        line("Architecture", s.architecture),
        line("Quantization", s.quantization),
        line("Paramètres", params),
        line("Contexte max", f"{s.n_ctx_train} tokens" if s.n_ctx_train else "?"),
        line("Couches", s.n_layers if s.n_layers is not None else "?"),
        line("Embedding", s.n_embd if s.n_embd is not None else "?"),
        line("Vocabulaire", s.n_vocab if s.n_vocab is not None else "?"),
        "",
        "── Inférence sur ce hardware ──",
        line("Backend", s.backend),
        line("Device", s.device),
        line("n_gpu_layers", f"{gpu} couches"),
        line("n_threads", s.n_threads),
        line("n_ctx", f"{s.n_ctx} tokens"),
        line("flash_attn", s.flash_attn),
        "",
        f"── Benchmark ({s.gen_tokens} tokens générés) ──",
        line("Chargement", f"{s.load_time_s} s"),
        line("1er token", ttft),
        line("Génération", gen),
        line("VRAM modèle", vram),
    ])

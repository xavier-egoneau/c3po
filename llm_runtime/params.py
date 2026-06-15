"""
Calcul des paramètres optimaux pour llama.cpp
à partir d'un HardwareProfile.
"""

from __future__ import annotations
from dataclasses import dataclass
from .hardware import Backend, HardwareProfile


@dataclass
class InferenceParams:
    n_gpu_layers: int    # -1 = tout sur GPU, 0 = CPU only, N = N couches sur GPU
    n_threads: int       # threads CPU
    n_ctx: int           # taille du contexte en tokens
    use_flash_attn: bool # optimisation mémoire pour longs contextes

    def __str__(self) -> str:
        gpu = "toutes" if self.n_gpu_layers == -1 else str(self.n_gpu_layers)
        return (
            f"n_gpu_layers  : {gpu} couches\n"
            f"n_threads     : {self.n_threads}\n"
            f"n_ctx         : {self.n_ctx} tokens\n"
            f"flash_attn    : {self.use_flash_attn}"
        )


def compute_params(
    profile: HardwareProfile,
    n_ctx: int = 4096,
    model_size_gb: float | None = None,
    n_layers: int | None = None,
    n_ctx_train: int | None = None,
) -> InferenceParams:
    """
    Calcule les paramètres d'inférence optimaux selon le hardware.

    n_ctx         : taille de contexte souhaitée (réduite si la mémoire est insuffisante).
    model_size_gb : taille du GGUF sur disque ; si fournie (CUDA), on ajuste n_gpu_layers
                    et n_ctx pour éviter de déborder la VRAM.
    n_layers      : nb de couches du modèle (en-tête GGUF) ; permet un offload partiel
                    chiffré quand le modèle ne tient pas entièrement.
    n_ctx_train   : contexte max à l'entraînement ; on ne demande jamais plus.
    """
    # On ne demande jamais plus de contexte que le modèle n'en supporte.
    if n_ctx_train:
        n_ctx = min(n_ctx, n_ctx_train)

    if profile.backend == Backend.METAL:
        return _params_metal(profile, n_ctx)

    elif profile.backend == Backend.CUDA:
        return _params_cuda(profile, n_ctx, model_size_gb, n_layers)

    else:  # CPU fallback
        return _params_cpu(profile, n_ctx)


def _params_metal(profile: HardwareProfile, n_ctx: int) -> InferenceParams:
    """
    Apple Silicon : mémoire unifiée, tout peut aller sur le GPU.
    On met toutes les couches sur Metal (-1) et on utilise
    tous les performance cores disponibles.
    """
    # Sur M4 : 4 performance cores + 6 efficiency cores
    # Pour l'inférence on veut les perf cores → on prend physicalcpu / 2 au minimum
    threads = max(profile.cpu_cores // 2, 4)

    # Flash attention utile dès que le contexte dépasse 2048
    flash = n_ctx > 2048

    return InferenceParams(
        n_gpu_layers=-1,
        n_threads=threads,
        n_ctx=n_ctx,
        use_flash_attn=flash,
    )


# Réserve VRAM pour le contexte CUDA, les buffers de calcul et le driver.
_CUDA_RESERVE_GB = 0.8
# Marge sur la taille des poids (overhead d'allocation).
_WEIGHTS_MARGIN = 1.05


def _kv_cache_gb(n_ctx: int, model_size_gb: float) -> float:
    """
    Estimation grossière de la taille du KV cache. Sans la config d'attention
    exacte (GQA, nb de têtes), on l'approxime proportionnellement à la taille du
    modèle et au contexte. Volontairement un peu conservateur.
    """
    return (n_ctx / 4096) * model_size_gb * 0.20


def _params_cuda(
    profile: HardwareProfile,
    n_ctx: int,
    model_size_gb: float | None,
    n_layers: int | None,
) -> InferenceParams:
    """
    Nvidia : VRAM dédiée et limitée. On vise tout sur GPU quand ça tient, on réduit
    le contexte si besoin, et en dernier recours on n'offload qu'une partie des
    couches (calculée depuis `n_layers`) plutôt que de tenter `-1` et faire OOM.
    """
    threads = min(profile.cpu_cores, 8)

    # Sans info sur le modèle : ancien comportement conservateur (rétrocompat).
    if model_size_gb is None:
        n_gpu_layers = -1 if profile.gpu_memory_gb >= 8.0 else 20
        return InferenceParams(n_gpu_layers, threads, n_ctx, n_ctx > 2048)

    budget = max(0.0, profile.gpu_memory_gb - _CUDA_RESERVE_GB)
    weights = model_size_gb * _WEIGHTS_MARGIN

    # 1) Tout tient au contexte demandé → toutes les couches sur GPU.
    if weights + _kv_cache_gb(n_ctx, model_size_gb) <= budget:
        return InferenceParams(-1, threads, n_ctx, n_ctx > 2048)

    # 2) Réduire le contexte jusqu'à ce que poids + KV tiennent.
    for ctx in (3072, 2048, 1536, 1024, 512):
        if ctx < n_ctx and weights + _kv_cache_gb(ctx, model_size_gb) <= budget:
            return InferenceParams(-1, threads, ctx, ctx > 2048)

    # 3) Même au minimum les poids ne tiennent pas → offload partiel chiffré.
    min_ctx = 512
    avail_for_weights = max(0.0, budget - _kv_cache_gb(min_ctx, model_size_gb))
    if n_layers and weights > 0:
        fraction = avail_for_weights / weights
        n_gpu_layers = max(0, min(n_layers, int(n_layers * fraction)))
    else:
        n_gpu_layers = 0  # on ne connaît pas le nb de couches : tout sur CPU, sûr
    return InferenceParams(n_gpu_layers, threads, min_ctx, False)


def _params_cpu(profile: HardwareProfile, n_ctx: int) -> InferenceParams:
    """CPU only : on maximise les threads, on réduit le contexte."""
    return InferenceParams(
        n_gpu_layers=0,
        n_threads=profile.cpu_cores,
        n_ctx=min(n_ctx, 2048),  # contexte réduit pour tenir en RAM
        use_flash_attn=False,
    )

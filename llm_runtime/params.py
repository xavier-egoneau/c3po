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
) -> InferenceParams:
    """
    Calcule les paramètres d'inférence optimaux selon le hardware.

    n_ctx : taille de contexte souhaitée (peut être réduite si mémoire insuffisante)
    """

    if profile.backend == Backend.METAL:
        return _params_metal(profile, n_ctx)

    elif profile.backend == Backend.CUDA:
        return _params_cuda(profile, n_ctx)

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


def _params_cuda(profile: HardwareProfile, n_ctx: int) -> InferenceParams:
    """
    Nvidia : VRAM limitée, on calcule combien de couches rentrent.
    Sans connaître le modèle à l'avance, on utilise toute la VRAM dispo
    et llama.cpp s'arrêtera de lui-même si ça déborde.
    On affine en Phase 3 quand on connaît le modèle.
    """
    # Heuristique : si on a > 8 Go de VRAM libre, on tente tout sur GPU
    if profile.gpu_memory_gb >= 8.0:
        n_gpu_layers = -1
    else:
        # Sinon on reste conservateur, à affiner avec le modèle réel
        n_gpu_layers = 20

    threads = min(profile.cpu_cores, 8)
    flash = n_ctx > 2048

    return InferenceParams(
        n_gpu_layers=n_gpu_layers,
        n_threads=threads,
        n_ctx=n_ctx,
        use_flash_attn=flash,
    )


def _params_cpu(profile: HardwareProfile, n_ctx: int) -> InferenceParams:
    """CPU only : on maximise les threads, on réduit le contexte."""
    return InferenceParams(
        n_gpu_layers=0,
        n_threads=profile.cpu_cores,
        n_ctx=min(n_ctx, 2048),  # contexte réduit pour tenir en RAM
        use_flash_attn=False,
    )

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
    kv_type: str = "f16" # précision du KV cache : "f16" | "q8_0" | "q4_0"
    speculative: bool = False  # prompt-lookup decoding (gain sur sorties qui recopient l'entrée)
    repeat_penalty: float = 1.1  # pénalité de répétition (1.0 = aucune ; anti-boucle léger par défaut)

    def __str__(self) -> str:
        gpu = "toutes" if self.n_gpu_layers == -1 else str(self.n_gpu_layers)
        return (
            f"n_gpu_layers   : {gpu} couches\n"
            f"n_threads      : {self.n_threads}\n"
            f"n_ctx          : {self.n_ctx} tokens\n"
            f"flash_attn     : {self.use_flash_attn}\n"
            f"kv_type        : {self.kv_type}\n"
            f"speculative    : {self.speculative}\n"
            f"repeat_penalty : {self.repeat_penalty}"
        )


# Octets par élément du KV cache selon la précision (F16 = défaut llama.cpp).
# Q8_0 ≈ 34 octets / 32 éléments, Q4_0 ≈ 18 / 32.
_KV_BYTES = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}


def compute_params(
    profile: HardwareProfile,
    n_ctx: int = 4096,
    model_size_gb: float | None = None,
    n_layers: int | None = None,
    n_ctx_train: int | None = None,
    n_embd: int | None = None,
    n_heads: int | None = None,
    n_kv_heads: int | None = None,
    kv_type: str | None = None,
) -> InferenceParams:
    """
    Calcule les paramètres d'inférence optimaux selon le hardware.

    n_ctx         : taille de contexte souhaitée (réduite si la mémoire est insuffisante).
    model_size_gb : taille du GGUF sur disque ; si fournie (CUDA), on ajuste n_gpu_layers
                    et n_ctx pour éviter de déborder la VRAM.
    n_layers      : nb de couches du modèle (en-tête GGUF) ; permet un offload partiel
                    chiffré quand le modèle ne tient pas entièrement.
    n_ctx_train   : contexte max à l'entraînement ; on ne demande jamais plus.
    n_embd / n_heads / n_kv_heads : config d'attention (en-tête GGUF) → calcul exact du KV
                    cache (gère la GQA) ; à défaut on retombe sur une heuristique.
    kv_type       : précision du KV cache forcée par l'utilisateur ("f16"/"q8_0"/"q4_0").
                    None = auto (F16, puis Q8_0 si besoin pour tenir ; jamais Q4 en auto).
    """
    # On ne demande jamais plus de contexte que le modèle n'en supporte.
    if n_ctx_train:
        n_ctx = min(n_ctx, n_ctx_train)

    if profile.backend == Backend.METAL:
        return _params_metal(profile, n_ctx)

    elif profile.backend == Backend.CUDA:
        return _params_cuda(profile, n_ctx, model_size_gb, n_layers,
                            n_embd, n_heads, n_kv_heads, kv_type)

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


def _kv_cache_gb(
    n_ctx: int,
    model_size_gb: float,
    n_layers: int | None = None,
    n_embd: int | None = None,
    n_heads: int | None = None,
    n_kv_heads: int | None = None,
    kv_type: str = "f16",
) -> float:
    """
    Taille du KV cache. Calcul exact si la config d'attention est connue :
        KV = 2 (K+V) × n_layers × n_ctx × (n_kv_heads × head_dim) × octets/élément
    avec head_dim = n_embd / n_heads (gère la GQA). `kv_type` donne les octets/élément
    (F16 = 2, Q8_0 ≈ 1.06, Q4_0 ≈ 0.56). À défaut, heuristique proportionnelle.
    """
    bytes_per_elem = _KV_BYTES.get(kv_type, 2.0)
    if n_layers and n_embd and n_heads:
        kv_heads = n_kv_heads or n_heads          # MHA si head_count_kv absent
        head_dim = n_embd / n_heads
        kv_dim = kv_heads * head_dim
        kv_bytes = 2 * n_layers * n_ctx * kv_dim * bytes_per_elem  # K+V
        return kv_bytes / 1024**3
    return (n_ctx / 4096) * model_size_gb * 0.20 * (bytes_per_elem / 2.0)


def _params_cuda(
    profile: HardwareProfile,
    n_ctx: int,
    model_size_gb: float | None,
    n_layers: int | None,
    n_embd: int | None = None,
    n_heads: int | None = None,
    n_kv_heads: int | None = None,
    kv_type: str | None = None,
) -> InferenceParams:
    """
    Nvidia : VRAM dédiée et limitée. On vise tout sur GPU quand ça tient, et pour faire
    rentrer le contexte demandé on dispose de deux leviers : la précision du KV cache et
    la taille du contexte. Politique auto (kv_type None) : F16 si ça tient, sinon Q8_0
    (quasi sans perte) pour garder le contexte, sinon on réduit le contexte (en Q8_0),
    sinon offload partiel. **Jamais de Q4 en auto** (perte de qualité) — uniquement si
    l'utilisateur force `kv_type="q4_0"`.
    """
    threads = min(profile.cpu_cores, 8)

    # Sans info sur le modèle : ancien comportement conservateur (rétrocompat).
    if model_size_gb is None:
        n_gpu_layers = -1 if profile.gpu_memory_gb >= 8.0 else 20
        return InferenceParams(n_gpu_layers, threads, n_ctx, n_ctx > 2048, kv_type or "f16")

    budget = max(0.0, profile.gpu_memory_gb - _CUDA_RESERVE_GB)
    weights = model_size_gb * _WEIGHTS_MARGIN

    def fits(ctx: int, kvt: str) -> bool:
        kv = _kv_cache_gb(ctx, model_size_gb, n_layers, n_embd, n_heads, n_kv_heads, kvt)
        return weights + kv <= budget

    def offload(kvt: str) -> InferenceParams:
        min_ctx = 512
        kv = _kv_cache_gb(min_ctx, model_size_gb, n_layers, n_embd, n_heads, n_kv_heads, kvt)
        avail = max(0.0, budget - kv)
        if n_layers and weights > 0:
            ngl = max(0, min(n_layers, int(n_layers * (avail / weights))))
        else:
            ngl = 0
        return InferenceParams(ngl, threads, min_ctx, False, kvt)

    # Précisions à essayer : forcée par l'utilisateur, sinon F16 puis Q8_0 (jamais Q4 auto).
    kv_candidates = [kv_type] if kv_type else ["f16", "q8_0"]

    # 1) Contexte plein : on prend la 1ère précision qui tient (meilleure d'abord).
    for kvt in kv_candidates:
        if fits(n_ctx, kvt):
            return InferenceParams(-1, threads, n_ctx, n_ctx > 2048, kvt)

    # 2) Contexte réduit, avec la précision la plus compacte considérée.
    kvt = kv_candidates[-1]
    for ctx in (3072, 2048, 1536, 1024, 512):
        if ctx < n_ctx and fits(ctx, kvt):
            return InferenceParams(-1, threads, ctx, ctx > 2048, kvt)

    # 3) Même au minimum les poids ne tiennent pas → offload partiel chiffré.
    return offload(kvt)


def apply_overrides(
    params: InferenceParams,
    n_gpu_layers: int | None = None,
    n_threads: int | None = None,
    use_flash_attn: bool | None = None,
) -> InferenceParams:
    """
    Applique les leviers fournis explicitement par l'utilisateur par-dessus les
    paramètres calculés. Un levier à None garde la valeur auto.
    """
    if n_gpu_layers is not None:
        params.n_gpu_layers = n_gpu_layers
    if n_threads is not None:
        params.n_threads = n_threads
    if use_flash_attn is not None:
        params.use_flash_attn = use_flash_attn
    return params


def _params_cpu(profile: HardwareProfile, n_ctx: int) -> InferenceParams:
    """CPU only : on maximise les threads, on réduit le contexte."""
    return InferenceParams(
        n_gpu_layers=0,
        n_threads=profile.cpu_cores,
        n_ctx=min(n_ctx, 2048),  # contexte réduit pour tenir en RAM
        use_flash_attn=False,
    )

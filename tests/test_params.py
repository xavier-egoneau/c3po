from llm_runtime.hardware import Backend, HardwareProfile
from llm_runtime.params import compute_params, apply_overrides, _kv_cache_gb


def _profile(backend: Backend, gpu_memory_gb: float, cpu_cores: int) -> HardwareProfile:
    return HardwareProfile(
        backend=backend,
        gpu_memory_gb=gpu_memory_gb,
        cpu_memory_gb=32.0,
        cpu_cores=cpu_cores,
        device_name="test-device",
    )


def test_metal_params_small_ctx():
    profile = _profile(Backend.METAL, gpu_memory_gb=12.0, cpu_cores=10)
    params = compute_params(profile, n_ctx=2048)

    assert params.n_gpu_layers == -1
    assert params.n_threads == 5  # cpu_cores // 2
    assert params.n_ctx == 2048
    assert params.use_flash_attn is False  # flash seulement si n_ctx > 2048


def test_metal_params_large_ctx_enables_flash_attn():
    profile = _profile(Backend.METAL, gpu_memory_gb=12.0, cpu_cores=10)
    params = compute_params(profile, n_ctx=4096)

    assert params.n_gpu_layers == -1
    assert params.use_flash_attn is True


def test_cuda_params_high_vram_uses_all_layers():
    profile = _profile(Backend.CUDA, gpu_memory_gb=12.0, cpu_cores=16)
    params = compute_params(profile, n_ctx=4096)

    assert params.n_gpu_layers == -1
    assert params.n_threads == 8  # min(cpu_cores, 8)
    assert params.use_flash_attn is True


def test_cuda_params_low_vram_is_conservative():
    profile = _profile(Backend.CUDA, gpu_memory_gb=4.0, cpu_cores=16)
    params = compute_params(profile, n_ctx=2048)

    assert params.n_gpu_layers == 20
    assert params.use_flash_attn is False


def test_cuda_params_model_fits_uses_all_layers():
    profile = _profile(Backend.CUDA, gpu_memory_gb=12.0, cpu_cores=16)
    params = compute_params(profile, n_ctx=4096, model_size_gb=4.4, n_layers=28)

    assert params.n_gpu_layers == -1
    assert params.n_ctx == 4096


def test_cuda_params_reduces_ctx_when_tight():
    # 5.5 Go : à 4096 le KV déborde, mais à 2048 ça tient → contexte réduit, tout sur GPU.
    profile = _profile(Backend.CUDA, gpu_memory_gb=5.5, cpu_cores=16)
    params = compute_params(profile, n_ctx=4096, model_size_gb=4.0, n_layers=28)

    assert params.n_gpu_layers == -1
    assert params.n_ctx == 2048


def test_cuda_params_partial_offload_when_model_too_big():
    # 3 Go : le modèle (4 Go) ne tient pas même au contexte minimal → offload partiel chiffré.
    profile = _profile(Backend.CUDA, gpu_memory_gb=3.0, cpu_cores=16)
    params = compute_params(profile, n_ctx=4096, model_size_gb=4.0, n_layers=32)

    assert 0 < params.n_gpu_layers < 32
    assert params.n_ctx == 512


def test_compute_params_caps_ctx_at_training_context():
    profile = _profile(Backend.METAL, gpu_memory_gb=12.0, cpu_cores=10)
    params = compute_params(profile, n_ctx=4096, n_ctx_train=2048)

    assert params.n_ctx == 2048  # on ne demande pas plus que le modèle ne supporte


def test_kv_cache_exact_accounts_for_gqa():
    # Qwen2.5-7B : 28 couches, embd 3584, 28 têtes mais 4 têtes KV (GQA).
    # KV @4096 ≈ 2×28×4096×(4×128)×2 octets ≈ 0.22 Go.
    kv = _kv_cache_gb(4096, model_size_gb=4.4, n_layers=28, n_embd=3584,
                      n_heads=28, n_kv_heads=4)
    assert 0.20 < kv < 0.25

    # Sans GQA (autant de têtes KV que de têtes) → ~7× plus gros.
    kv_mha = _kv_cache_gb(4096, model_size_gb=4.4, n_layers=28, n_embd=3584,
                          n_heads=28, n_kv_heads=28)
    assert kv_mha > kv * 6


def test_kv_cache_falls_back_to_heuristic_without_attention_config():
    kv = _kv_cache_gb(4096, model_size_gb=5.0, n_layers=None, n_embd=None,
                      n_heads=None, n_kv_heads=None)
    assert kv == (4096 / 4096) * 5.0 * 0.20


def test_cuda_exact_kv_keeps_ctx_where_heuristic_would_reduce():
    # budget = 5.8 - 0.8 = 5.0 ; poids 4.4×1.05 = 4.62.
    # KV exact @4096 ≈ 0.22 → 4.84 ≤ 5.0 : tient. Heuristique 0.88 → 5.5 > 5.0 : réduirait.
    profile = _profile(Backend.CUDA, gpu_memory_gb=5.8, cpu_cores=16)

    exact = compute_params(profile, n_ctx=4096, model_size_gb=4.4, n_layers=28,
                           n_embd=3584, n_heads=28, n_kv_heads=4)
    heuristic = compute_params(profile, n_ctx=4096, model_size_gb=4.4, n_layers=28)

    assert exact.n_ctx == 4096        # KV exact (petit) → contexte plein conservé
    assert heuristic.n_ctx < 4096     # heuristique (surévaluée) → contexte réduit


def test_apply_overrides_replaces_only_provided_levers():
    profile = _profile(Backend.CUDA, gpu_memory_gb=12.0, cpu_cores=16)
    params = compute_params(profile, n_ctx=4096)  # auto : -1, 8 threads, flash True

    apply_overrides(params, n_gpu_layers=10, use_flash_attn=False)

    assert params.n_gpu_layers == 10        # forcé
    assert params.use_flash_attn is False   # forcé
    assert params.n_threads == 8            # inchangé (None → auto conservé)


def test_apply_overrides_noop_when_all_none():
    profile = _profile(Backend.CUDA, gpu_memory_gb=12.0, cpu_cores=16)
    params = compute_params(profile, n_ctx=4096)
    before = (params.n_gpu_layers, params.n_threads, params.use_flash_attn)

    apply_overrides(params)

    assert (params.n_gpu_layers, params.n_threads, params.use_flash_attn) == before


def test_cpu_params_caps_context_and_disables_gpu():
    profile = _profile(Backend.CPU, gpu_memory_gb=0.0, cpu_cores=8)
    params = compute_params(profile, n_ctx=4096)

    assert params.n_gpu_layers == 0
    assert params.n_threads == 8
    assert params.n_ctx == 2048  # plafonné
    assert params.use_flash_attn is False

from llm_runtime.hardware import Backend, HardwareProfile
from llm_runtime.params import compute_params


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


def test_cpu_params_caps_context_and_disables_gpu():
    profile = _profile(Backend.CPU, gpu_memory_gb=0.0, cpu_cores=8)
    params = compute_params(profile, n_ctx=4096)

    assert params.n_gpu_layers == 0
    assert params.n_threads == 8
    assert params.n_ctx == 2048  # plafonné
    assert params.use_flash_attn is False

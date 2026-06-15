from unittest.mock import patch

from llm_runtime.hardware import Backend, _detect_nvidia


from llm_runtime.hardware import _NVIDIA_RESERVE_GB

NVIDIA_SMI_OUTPUT = "NVIDIA GeForce RTX 4070, 12288\n"


def test_detect_nvidia_parses_csv_output():
    with patch("subprocess.check_output", return_value=NVIDIA_SMI_OUTPUT) as mock_check_output, \
         patch("llm_runtime.hardware._get_system_ram_gb", return_value=32.0), \
         patch("llm_runtime.hardware._get_cpu_cores", return_value=16):
        profile = _detect_nvidia()

    mock_check_output.assert_called_once()
    assert profile is not None
    assert profile.backend == Backend.CUDA
    assert profile.device_name == "NVIDIA GeForce RTX 4070"
    assert profile.gpu_memory_gb == 12288 / 1024 - _NVIDIA_RESERVE_GB
    assert profile.cpu_memory_gb == 32.0
    assert profile.cpu_cores == 16


def test_detect_nvidia_returns_none_without_gpu():
    with patch("subprocess.check_output", side_effect=FileNotFoundError()):
        profile = _detect_nvidia()

    assert profile is None


def test_detect_nvidia_returns_none_on_empty_output():
    with patch("subprocess.check_output", return_value=""):
        profile = _detect_nvidia()

    assert profile is None

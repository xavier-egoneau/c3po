import struct
import sys
import os
from pathlib import Path
from unittest.mock import patch

from llm_runtime.doctor import collect_doctor, format_doctor
from llm_runtime.hardware import Backend, HardwareProfile
from llm_runtime.models import ModelInfo

_U32, _STR = 4, 8


def _profile() -> HardwareProfile:
    return HardwareProfile(
        backend=Backend.CUDA,
        gpu_memory_gb=11.0,
        cpu_memory_gb=32.0,
        cpu_cores=8,
        device_name="test-gpu",
    )


def _str(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def _kv_str(key: bytes, val: bytes) -> bytes:
    return _str(key) + struct.pack("<I", _STR) + _str(val)


def _kv_u32(key: bytes, val: int) -> bytes:
    return _str(key) + struct.pack("<I", _U32) + struct.pack("<I", val)


def _build_gguf(kvs: list[bytes]) -> bytes:
    header = b"GGUF" + struct.pack("<I", 3)
    header += struct.pack("<Q", 0)
    header += struct.pack("<Q", len(kvs))
    return header + b"".join(kvs)


def test_collect_doctor_reports_missing_llama_cpp():
    with patch("llm_runtime.doctor.detect_hardware", return_value=_profile()), \
         patch("llm_runtime.doctor._llama_cpp_info", return_value={
             "installed": False,
             "version": None,
             "gpu_offload_supported": None,
         }), \
         patch("llm_runtime.doctor.shutil.which", return_value=None):
        report = collect_doctor()

    assert report.llama_cpp_installed is False
    assert report.python_executable == sys.executable
    assert report.errors
    expected = "setup_windows_cuda.ps1" if os.name == "nt" else "GGML_CUDA"
    assert any(expected in a for a in report.actions)
    assert "llama-cpp-python" in format_doctor(report)
    assert "Actions recommandées" in format_doctor(report)


def test_collect_doctor_recommends_rebuild_when_gpu_offload_missing():
    with patch("llm_runtime.doctor.detect_hardware", return_value=_profile()), \
         patch("llm_runtime.doctor.shutil.which", return_value=None), \
         patch("llm_runtime.doctor._llama_cpp_info", return_value={
             "installed": True,
             "version": "0.0-test",
             "gpu_offload_supported": False,
         }):
        report = collect_doctor()

    assert report.warnings
    expected = "setup_windows_cuda.ps1" if os.name == "nt" else "--force-reinstall"
    assert any(expected in a for a in report.actions)


def test_collect_doctor_inspects_model_and_mmproj(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(_build_gguf([
        _kv_str(b"general.architecture", b"qwen2"),
        _kv_u32(b"qwen2.block_count", 28),
        _kv_u32(b"qwen2.context_length", 32768),
        _kv_u32(b"qwen2.embedding_length", 3584),
        _kv_u32(b"qwen2.attention.head_count", 28),
        _kv_u32(b"qwen2.attention.head_count_kv", 4),
    ]))
    (tmp_path / "mmproj-model.gguf").write_bytes(b"placeholder")
    info = ModelInfo("model", Path(model), 4.0, "local")

    with patch("llm_runtime.doctor.detect_hardware", return_value=_profile()), \
         patch("llm_runtime.doctor.shutil.which", return_value=None), \
         patch("llm_runtime.doctor._llama_cpp_info", return_value={
             "installed": True,
             "version": "0.0-test",
             "gpu_offload_supported": True,
         }), \
         patch("llm_runtime.doctor.find_model", return_value=info):
        report = collect_doctor("model")

    assert report.llama_cpp_installed is True
    assert report.arch == "qwen2"
    assert report.n_ctx_train == 32768
    assert report.mmproj_files == ["mmproj-model.gguf"]
    assert any("multimodal" in w for w in report.warnings)
    assert any("échoue au chargement" in a for a in report.actions)


def test_collect_doctor_does_not_attach_generic_mmproj_to_text_model(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(_build_gguf([
        _kv_str(b"general.architecture", b"qwen2"),
        _kv_u32(b"qwen2.block_count", 28),
    ]))
    (tmp_path / "mmproj-BF16.gguf").write_bytes(b"placeholder")
    info = ModelInfo("model", Path(model), 4.0, "local")

    with patch("llm_runtime.doctor.detect_hardware", return_value=_profile()), \
         patch("llm_runtime.doctor.shutil.which", return_value=None), \
         patch("llm_runtime.doctor._llama_cpp_info", return_value={
             "installed": True,
             "version": "0.0-test",
             "gpu_offload_supported": True,
         }), \
         patch("llm_runtime.doctor.find_model", return_value=info):
        report = collect_doctor("model")

    assert report.mmproj_files == []


def test_collect_doctor_warns_when_c3po_uses_another_python(tmp_path):
    c3po = tmp_path / "c3po"
    c3po.write_text("#!/other/python\nprint('x')\n")

    with patch("llm_runtime.doctor.detect_hardware", return_value=_profile()), \
         patch("llm_runtime.doctor.shutil.which", return_value=str(c3po)), \
         patch("llm_runtime.doctor._llama_cpp_info", return_value={
             "installed": True,
             "version": "0.0-test",
             "gpu_offload_supported": True,
         }), \
         patch("llm_runtime.doctor.sys.executable", str(tmp_path / "python.exe")), \
         patch("llm_runtime.doctor.sys.prefix", sys.base_prefix):
        report = collect_doctor()

    assert report.c3po_executable == str(c3po)
    assert report.c3po_shebang == "/other/python"
    assert any("même Python" in w for w in report.warnings)

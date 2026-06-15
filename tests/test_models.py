from pathlib import Path
from unittest.mock import patch

import pytest

from llm_runtime.models import ModelInfo, find_model, list_models, best_model


def _make_gguf(directory: Path, name: str, size_mb: int) -> Path:
    path = directory / name
    path.write_bytes(b"\0" * (size_mb * 1024 * 1024))
    return path


def test_model_info_fits_in():
    info = ModelInfo(name="m", path=Path("m.gguf"), size_gb=4.0, source="local")
    assert info.fits_in(5.0) is True   # 4.0 * 1.1 = 4.4 <= 5.0
    assert info.fits_in(4.0) is False  # 4.0 * 1.1 = 4.4 > 4.0


def test_list_models_scans_local_dir_and_sorts_by_size(tmp_path):
    _make_gguf(tmp_path, "big.gguf", size_mb=20)
    _make_gguf(tmp_path, "small.gguf", size_mb=5)

    with patch("llm_runtime.models._scan_ollama", return_value=[]):
        models = list_models(local_dirs=[tmp_path])

    assert [m.name for m in models] == ["small", "big"]
    assert all(m.source == "local" for m in models)


def test_list_models_dedups_by_path(tmp_path):
    _make_gguf(tmp_path, "model.gguf", size_mb=10)

    with patch("llm_runtime.models._scan_ollama", return_value=[]):
        models = list_models(local_dirs=[tmp_path, tmp_path])

    assert len(models) == 1


def test_best_model_returns_largest_that_fits(tmp_path):
    _make_gguf(tmp_path, "small.gguf", size_mb=100)   # ~0.098 Go
    _make_gguf(tmp_path, "big.gguf", size_mb=2000)    # ~1.95 Go

    best = best_model(available_memory_gb=1.0, local_dirs=[tmp_path])

    assert best is not None
    assert best.name == "small"


def test_best_model_returns_none_when_nothing_fits(tmp_path):
    _make_gguf(tmp_path, "huge.gguf", size_mb=5000)

    best = best_model(available_memory_gb=1.0, local_dirs=[tmp_path])

    assert best is None


def test_find_model_by_direct_path(tmp_path):
    gguf = _make_gguf(tmp_path, "model.gguf", size_mb=10)

    info = find_model(str(gguf))

    assert info.path == gguf
    assert info.source == "local"


def test_find_model_by_substring(tmp_path):
    _make_gguf(tmp_path, "qwen2.5-7b.gguf", size_mb=10)

    with patch("llm_runtime.models._scan_ollama", return_value=[]):
        info = find_model("qwen", local_dirs=[tmp_path])

    assert info.name == "qwen2.5-7b"


def test_find_model_ambiguous_raises(tmp_path):
    _make_gguf(tmp_path, "qwen2.5-7b.gguf", size_mb=10)
    _make_gguf(tmp_path, "qwen2.5-14b.gguf", size_mb=20)

    with patch("llm_runtime.models._scan_ollama", return_value=[]):
        with pytest.raises(ValueError, match="Plusieurs modèles"):
            find_model("qwen", local_dirs=[tmp_path])


def test_find_model_not_found_raises(tmp_path):
    with patch("llm_runtime.models._scan_ollama", return_value=[]):
        with pytest.raises(ValueError, match="introuvable"):
            find_model("nope", local_dirs=[tmp_path])

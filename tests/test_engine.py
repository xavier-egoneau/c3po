"""
Tests de `Engine` avec `llama_cpp` mocké.

`Engine._load_model` est le seul point qui importe `llama-cpp-python` : on le patche
pour exercer toute la logique d'`__init__` (calcul/override des params, forçage flash
attention sur KV quantifié, garde-fou mémoire, enregistrement d'instance) sans GPU ni
GGUF réel. Comble le trou « cœur testé seulement manuellement » (point #12).
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from llm_runtime.engine import Engine
from llm_runtime.hardware import Backend, HardwareProfile


def _metal_profile():
    return HardwareProfile(
        backend=Backend.METAL,
        gpu_memory_gb=12.0,
        cpu_memory_gb=16.0,
        cpu_cores=10,
        device_name="Apple M4 (test)",
    )


# Shape d'un 7B fictif, suffisante pour compute_params / _kv_cache_gb.
_FAKE_SHAPE = {
    "n_layers": 28,
    "n_ctx_train": 32768,
    "n_embd": 3584,
    "n_heads": 28,
    "n_kv_heads": 4,
}


@contextmanager
def _patched_engine(profile=None, pressure=None):
    """
    Patche les dépendances externes d'`Engine.__init__` :
    hardware, en-tête GGUF, garde-fou mémoire, enregistrement d'instance et chargement
    du modèle (qui importerait llama_cpp). `pressure` = message renvoyé par
    check_memory_pressure (None = pas de pression).
    """
    profile = profile or _metal_profile()
    with patch("llm_runtime.engine.detect_hardware", return_value=profile), \
         patch("llm_runtime.engine.model_shape", return_value=dict(_FAKE_SHAPE)), \
         patch("llm_runtime.engine.check_memory_pressure", return_value=pressure), \
         patch("llm_runtime.engine.register_instance") as reg, \
         patch.object(Engine, "_load_model", return_value=MagicMock()) as load:
        yield reg, load


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "fake-7b.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 1024)  # contenu factice, jamais lu (model_shape mocké)
    return path


def test_init_computes_params_and_registers(model_file):
    with _patched_engine() as (reg, load):
        engine = Engine(model_file, n_ctx=4096)

    # Metal → toutes les couches sur GPU, contexte conservé.
    assert engine.params.n_gpu_layers == -1
    assert engine.params.n_ctx == 4096
    assert engine.params.kv_type == "f16"
    load.assert_called_once()
    reg.assert_called_once()  # instance enregistrée après chargement réussi


def test_n_ctx_capped_to_training_context(model_file):
    with _patched_engine():
        engine = Engine(model_file, n_ctx=999999)
    assert engine.params.n_ctx == _FAKE_SHAPE["n_ctx_train"]


def test_explicit_overrides_take_precedence(model_file):
    with _patched_engine():
        engine = Engine(
            model_file,
            n_ctx=4096,
            n_gpu_layers=7,
            n_threads=3,
            flash_attn=False,
        )
    assert engine.params.n_gpu_layers == 7
    assert engine.params.n_threads == 3
    assert engine.params.use_flash_attn is False


def test_quantized_kv_forces_flash_attn(model_file):
    # Même si on désactive explicitement flash-attn, un KV quantifié le réactive
    # (llama.cpp l'exige).
    with _patched_engine():
        engine = Engine(model_file, n_ctx=4096, kv_type="q8_0", flash_attn=False)
    assert engine.params.kv_type == "q8_0"
    assert engine.params.use_flash_attn is True


def test_speculative_and_repeat_penalty_propagated(model_file):
    with _patched_engine():
        engine = Engine(model_file, speculative=True, repeat_penalty=1.3)
    assert engine.params.speculative is True
    assert engine.params.repeat_penalty == 1.3


def test_memory_pressure_blocks_loading(model_file):
    with _patched_engine(pressure="⚠️  trop de mémoire") as (reg, load):
        with pytest.raises(RuntimeError, match="Chargement bloqué"):
            Engine(model_file)
    load.assert_not_called()
    reg.assert_not_called()


def test_memory_pressure_bypassed_with_force(model_file):
    with _patched_engine(pressure="⚠️  trop de mémoire") as (reg, load):
        engine = Engine(model_file, force=True)
    assert engine is not None
    load.assert_called_once()
    reg.assert_called_once()


def test_memory_pressure_bypassed_with_env(model_file, monkeypatch):
    monkeypatch.setenv("C3PO_FORCE", "1")
    with _patched_engine(pressure="⚠️  trop de mémoire") as (reg, load):
        Engine(model_file)
    load.assert_called_once()


def test_missing_model_file_raises(tmp_path):
    with _patched_engine():
        with pytest.raises(FileNotFoundError, match="introuvable"):
            Engine(tmp_path / "absent.gguf")

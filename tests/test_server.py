from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import llm_runtime.server as server
from llm_runtime.server import ChatMessage, ChatCompletionRequest, get_engine


def test_chat_message_model():
    msg = ChatMessage(role="user", content="bonjour")
    assert msg.role == "user"
    assert msg.content == "bonjour"


def test_chat_completion_request_defaults():
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "salut"}])

    assert request.model is None
    assert request.max_tokens is None  # None → jusqu'à l'EOS / la limite de contexte
    assert request.temperature == 0.7
    assert request.stream is False
    assert request.messages[0].role == "user"


def test_chat_completion_request_with_model():
    request = ChatCompletionRequest(
        model="qwen2.5-7b-instruct",
        messages=[{"role": "user", "content": "salut"}],
        stream=True,
    )

    assert request.model == "qwen2.5-7b-instruct"
    assert request.stream is True


@pytest.fixture(autouse=True)
def _reset_engine_state():
    server._engine = None
    server._model_path = None
    yield
    server._engine = None
    server._model_path = None


def _fake_model_info(name, path):
    info = MagicMock()
    info.path = Path(path)
    info.name = name
    return info


def test_get_engine_loads_model_on_first_call():
    with patch("llm_runtime.server.find_model", return_value=_fake_model_info("qwen", "models/qwen.gguf")), \
         patch("llm_runtime.server.Engine") as MockEngine:
        MockEngine.return_value = MagicMock()

        engine = get_engine("qwen")

        MockEngine.assert_called_once_with(Path("models/qwen.gguf"))
        assert engine is MockEngine.return_value
        assert server._model_path == Path("models/qwen.gguf")


def test_get_engine_cache_hit_same_model():
    with patch("llm_runtime.server.find_model", return_value=_fake_model_info("qwen", "models/qwen.gguf")), \
         patch("llm_runtime.server.Engine") as MockEngine:
        MockEngine.return_value = MagicMock()

        first = get_engine("qwen")
        second = get_engine("qwen")

        MockEngine.assert_called_once()
        assert first is second


def test_get_engine_swaps_on_different_model():
    with patch("llm_runtime.server.find_model") as mock_find, \
         patch("llm_runtime.server.Engine") as MockEngine:
        mock_find.side_effect = [
            _fake_model_info("qwen", "models/qwen.gguf"),
            _fake_model_info("gemma4", "models/gemma4.gguf"),
        ]
        MockEngine.side_effect = [MagicMock(), MagicMock()]

        first = get_engine("qwen")
        second = get_engine("gemma4")

        assert MockEngine.call_count == 2
        assert first is not second
        assert server._model_path == Path("models/gemma4.gguf")


def test_get_engine_unknown_model_raises_value_error():
    with patch("llm_runtime.server.find_model", side_effect=ValueError("Modèle 'nope' introuvable")):
        with pytest.raises(ValueError, match="introuvable"):
            get_engine("nope")

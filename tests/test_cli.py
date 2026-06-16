from types import SimpleNamespace

from llm_runtime.cli import _maybe_compact_history, _safe_generation_max_tokens


class FakeEngine:
    def __init__(self):
        self.params = SimpleNamespace(n_ctx=20)

    def count_messages_tokens(self, messages):
        return sum(len(m.get("content", "").split()) for m in messages)

    def chat(self, messages, max_tokens=None, temperature=0.7, stream=False):
        return {"choices": [{"message": {"content": "## Objectif courant\nrésumé utile"}}]}


def test_maybe_compact_history_replaces_old_messages_with_summary(capsys):
    engine = FakeEngine()
    history = [
        {"role": "user", "content": "un deux trois quatre cinq"},
        {"role": "assistant", "content": "six sept huit neuf dix"},
        {"role": "user", "content": "onze douze treize quatorze quinze"},
        {"role": "assistant", "content": "seize dix-sept dix-huit dix-neuf vingt"},
        {"role": "user", "content": "question finale"},
    ]

    compacted = _maybe_compact_history(engine, history)

    assert compacted is not history
    assert compacted[0]["role"] == "system"
    assert "Mémoire structurée compactée" in compacted[0]["content"]
    assert "## Objectif courant" in compacted[0]["content"]
    assert "résumé utile" in compacted[0]["content"]
    assert compacted[-1] == {"role": "user", "content": "question finale"}
    assert "Historique compacté" in capsys.readouterr().out


def test_maybe_compact_history_keeps_short_history_unchanged():
    engine = FakeEngine()
    history = [{"role": "user", "content": "petit message"}]

    assert _maybe_compact_history(engine, history) is history


def test_safe_generation_max_tokens_uses_remaining_context():
    engine = FakeEngine()
    engine.params.n_ctx = 100
    history = [{"role": "user", "content": "un deux trois quatre cinq"}]

    assert _safe_generation_max_tokens(engine, history, None) == 31


def test_safe_generation_max_tokens_caps_explicit_request():
    engine = FakeEngine()
    engine.params.n_ctx = 100
    history = [{"role": "user", "content": "un deux trois quatre cinq"}]

    assert _safe_generation_max_tokens(engine, history, 200) == 31
    assert _safe_generation_max_tokens(engine, history, 12) == 12


def test_safe_generation_max_tokens_returns_zero_when_context_full():
    engine = FakeEngine()
    engine.params.n_ctx = 70
    history = [{"role": "user", "content": "un deux trois quatre cinq six"}]

    assert _safe_generation_max_tokens(engine, history, None) == 0

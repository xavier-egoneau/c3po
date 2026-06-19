"""Compensation transparente de la fenêtre de contexte pour le serveur (mode `--equalize`).

But (voie B) : qu'un petit modèle servi derrière l'API ne CRASHE jamais sur un prompt trop
gros — l'appelant (un agent au-dessus) l'utilise sans se soucier de la fenêtre. On garantit
que la requête tient dans `n_ctx` : on compacte les messages (système + tours récents, on
élague le milieu, on tronque le dernier en dernier recours) et on borne la génération.

Déterministe et SANS appel LLM supplémentaire — distinct de la compaction par résumé du chat
interactif (`cli.py`), qui vise les longues sessions et accepte le coût d'un résumé.
"""

from __future__ import annotations

from typing import Any

_OUTPUT_RESERVE_RATIO = 0.35  # part de n_ctx visée pour la réponse (le reste = budget d'entrée)
_SAFETY_TOKENS = 64


def equalize_messages(
    engine: Any,
    messages: list[dict],
    requested_max_tokens: int | None,
) -> tuple[list[dict], int, bool]:
    """Renvoie (messages compactés pour tenir, max_tokens borné, a-t-on compacté)."""
    n_ctx = engine.params.n_ctx
    input_budget = max(1, n_ctx - int(n_ctx * _OUTPUT_RESERVE_RATIO) - _SAFETY_TOKENS)
    messages, compacted = _fit_messages(engine, list(messages), input_budget)
    max_tokens = _safe_max_tokens(engine, messages, requested_max_tokens, n_ctx)
    return messages, max_tokens, compacted


def _count(engine: Any, messages: list[dict]) -> int:
    try:
        return engine.count_messages_tokens(messages)
    except Exception:  # noqa: BLE001 - repli grossier si le tokenizer échoue
        return sum(len(m.get("content", "")) for m in messages) // 4


def _fit_messages(engine: Any, messages: list[dict], budget: int) -> tuple[list[dict], bool]:
    if not messages or _count(engine, messages) <= budget:
        return messages, False

    system = messages[:1] if messages[0].get("role") == "system" else []
    body = messages[len(system):]
    last = body[-1:] if body else []

    kept = system + last
    if _count(engine, kept) > budget:
        # Même système + dernier message dépasse : on tronque le contenu du dernier
        # (on garde la FIN — la question/instruction y est généralement).
        return _truncate_last(engine, system, last, budget), True

    # On rajoute les tours récents tant que ça tient (contexte le plus frais d'abord).
    for message in reversed(body[:-1]):
        trial = system + [message] + kept[len(system):]
        if _count(engine, trial) <= budget:
            kept = trial
        else:
            break
    return kept, True


def _truncate_last(engine: Any, system: list[dict], last: list[dict], budget: int) -> list[dict]:
    if not last:
        return system
    message = dict(last[0])
    overhead = _count(engine, system + [{**message, "content": ""}])
    available_tokens = max(1, budget - overhead)
    available_chars = available_tokens * 3  # ~conservateur (3 char/token)
    content = message.get("content", "")
    if len(content) > available_chars:
        message["content"] = "[...début tronqué pour tenir dans la fenêtre de contexte...]\n" + content[-available_chars:]
    return system + [message]


def _safe_max_tokens(engine: Any, messages: list[dict], requested: int | None, n_ctx: int) -> int:
    """Borne la génération pour ne jamais atteindre le bord du buffer llama.cpp."""
    available = n_ctx - _count(engine, messages) - _SAFETY_TOKENS
    if available <= 0:
        return 1
    if requested is None or requested <= 0:
        return available
    return min(requested, available)

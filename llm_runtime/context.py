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


def has_content(result: dict) -> bool:
    """Une complétion a-t-elle un contenu non vide ? (pour le retry transparent)"""
    try:
        return bool(result["choices"][0]["message"]["content"].strip())
    except Exception:  # noqa: BLE001
        return True  # dans le doute, ne pas re-générer


def route_vision_messages(messages: list[dict]) -> tuple[list[dict], bool]:
    """Routage vision (voie B) : si un message contient une image (format multimodal OpenAI :
    `content` = liste avec un part `image_url`), on la décrit via le **sidecar vision** et on
    remplace l'image par sa description texte. → un modèle TEXTE 'voit' les images, de façon
    transparente pour l'appelant. Sidecar absent/échec → l'image devient une note, pas un crash."""
    if not any(isinstance(m.get("content"), list) for m in messages):
        return messages, False

    describe = _get_image_describer()
    routed: list[dict] = []
    used = False
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            routed.append(message)
            continue
        parts: list[str] = []
        for part in content:
            kind = part.get("type")
            if kind == "text":
                parts.append(str(part.get("text", "")))
            elif kind in ("image_url", "input_image"):
                url = (part.get("image_url") or {}).get("url", "") if kind == "image_url" else part.get("image_url", "")
                desc = describe(url) if describe else None
                used = used or bool(desc)
                parts.append(f"[Image (décrite automatiquement par le sidecar vision) : {desc}]"
                             if desc else "[Image fournie mais non décrite (sidecar vision indisponible)]")
        routed.append({**message, "content": "\n".join(p for p in parts if p)})
    return routed, used


def _get_image_describer():
    """Renvoie une fonction url->description (texte), ou None si le sidecar est indisponible."""
    try:
        from agent.vision.gemma4 import observe_image_subprocess
    except Exception:  # noqa: BLE001 - sidecar/agent absent -> routage dégradé proprement
        return None

    import base64
    import os
    import tempfile
    import urllib.request

    def describe(url: str) -> str | None:
        try:
            if url.startswith("data:"):
                data = base64.b64decode(url.split(",", 1)[1])
            elif url.startswith(("http://", "https://")):
                data = urllib.request.urlopen(url, timeout=10).read()  # noqa: S310
            elif url.startswith("file://"):
                data = open(url[7:], "rb").read()
            else:
                return None
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                handle.write(data)
                path = handle.name
            try:
                observation = observe_image_subprocess(path)
            finally:
                os.unlink(path)
            parsed = observation.get("parsed") or {}
            caption = str(parsed.get("caption", "")).strip()
            ocr = parsed.get("ocr") or []
            text = caption + (" | texte visible : " + ", ".join(map(str, ocr)) if ocr else "")
            return text.strip() or (observation.get("raw") or None)
        except Exception:  # noqa: BLE001
            return None

    return describe

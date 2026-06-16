"""
Helpers expérimentaux pour sorties JSON validées.

Cette brique appartient à c3po-agent : elle encadre un modèle local sans déplacer
la logique produit ou l'exécution d'outils dans c3po-core.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


_MISSING = object()


@dataclass(frozen=True)
class FieldSpec:
    expected: type | tuple[type, ...]
    required: bool = True
    default: Any = _MISSING


@dataclass
class StructuredResult:
    ok: bool
    raw: str
    parsed: dict | None
    errors: list[str]
    attempts: int


Schema = dict[str, FieldSpec | type | tuple[type, ...]]


def extract_json_object(text: str) -> dict | None:
    """Retourne le premier objet JSON valide trouvé dans une réponse modèle."""
    stripped = text.strip()
    if not stripped:
        return None

    direct = _loads_object(stripped)
    if direct is not None:
        return direct

    for fenced in _fenced_blocks(stripped):
        value = _loads_object(fenced)
        if value is not None:
            return value

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def validate_object(value: dict | None, schema: Schema) -> list[str]:
    if value is None:
        return ["response did not contain a JSON object"]
    if not isinstance(value, dict):
        return [f"expected JSON object, got {type(value).__name__}"]

    errors: list[str] = []
    for name, raw_spec in schema.items():
        spec = _field_spec(raw_spec)
        if name not in value:
            if spec.required and spec.default is _MISSING:
                errors.append(f"missing required field: {name}")
            continue
        if not _matches_type(value[name], spec.expected):
            errors.append(
                f"field {name} expected {_type_name(spec.expected)}, "
                f"got {type(value[name]).__name__}"
            )
    return errors


def apply_defaults(value: dict | None, schema: Schema) -> dict | None:
    if value is None:
        return None
    result = dict(value)
    for name, raw_spec in schema.items():
        spec = _field_spec(raw_spec)
        if name not in result and spec.default is not _MISSING:
            result[name] = spec.default
    return result


def parse_validated(text: str, schema: Schema) -> StructuredResult:
    parsed = extract_json_object(text)
    parsed = apply_defaults(parsed, schema)
    errors = validate_object(parsed, schema)
    return StructuredResult(
        ok=not errors,
        raw=text,
        parsed=parsed,
        errors=errors,
        attempts=1,
    )


def generate_structured(
    call_model: Callable[[str], str],
    prompt: str,
    schema: Schema,
    max_retries: int = 1,
) -> StructuredResult:
    current_prompt = prompt
    last_result: StructuredResult | None = None

    for attempt in range(1, max_retries + 2):
        raw = call_model(current_prompt)
        parsed = extract_json_object(raw)
        parsed = apply_defaults(parsed, schema)
        errors = validate_object(parsed, schema)
        result = StructuredResult(
            ok=not errors,
            raw=raw,
            parsed=parsed,
            errors=errors,
            attempts=attempt,
        )
        if result.ok:
            return result

        last_result = result
        current_prompt = retry_prompt(prompt, schema, errors)

    return last_result or StructuredResult(
        ok=False,
        raw="",
        parsed=None,
        errors=["model was not called"],
        attempts=0,
    )


def retry_prompt(prompt: str, schema: Schema, errors: list[str]) -> str:
    schema_lines = [f"- {name}: {_type_name(_field_spec(spec).expected)}" for name, spec in schema.items()]
    error_lines = [f"- {error}" for error in errors]
    return (
        f"{prompt}\n\n"
        "La réponse précédente ne respecte pas le contrat JSON.\n"
        "Erreurs détectées :\n"
        f"{chr(10).join(error_lines)}\n\n"
        "Schéma attendu :\n"
        f"{chr(10).join(schema_lines)}\n\n"
        "Réponds à nouveau uniquement avec un objet JSON valide."
    )


def _loads_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _fenced_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    parts = text.split("```")
    for index in range(1, len(parts), 2):
        block = parts[index].strip()
        if block.lower().startswith("json"):
            block = block[4:].strip()
        blocks.append(block)
    return blocks


def _field_spec(value: FieldSpec | type | tuple[type, ...]) -> FieldSpec:
    if isinstance(value, FieldSpec):
        return value
    return FieldSpec(expected=value)


def _matches_type(value: Any, expected: type | tuple[type, ...]) -> bool:
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    for expected_type in expected_types:
        if expected_type is bool:
            if type(value) is bool:
                return True
            continue
        if expected_type is int:
            if type(value) is int:
                return True
            continue
        if expected_type is float:
            if type(value) in (float, int) and type(value) is not bool:
                return True
            continue
        if isinstance(value, expected_type):
            return True
    return False


def _type_name(expected: type | tuple[type, ...]) -> str:
    if isinstance(expected, tuple):
        return " | ".join(t.__name__ for t in expected)
    return expected.__name__

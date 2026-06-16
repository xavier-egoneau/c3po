from agent.structured import (
    FieldSpec,
    extract_json_object,
    generate_structured,
    parse_validated,
    validate_object,
)


def test_extract_json_object_accepts_plain_fenced_and_surrounded_json():
    assert extract_json_object('{"action": "answer"}') == {"action": "answer"}
    assert extract_json_object('```json\n{"action": "answer"}\n```') == {
        "action": "answer"
    }
    assert extract_json_object('Avant {"action": "answer"} après') == {
        "action": "answer"
    }
    assert extract_json_object("[1, 2, 3]") is None


def test_validate_object_reports_missing_and_wrong_types():
    schema = {
        "action": str,
        "args": dict,
        "confidence": FieldSpec((int, float), required=False),
    }

    assert validate_object({"action": "tool_call", "args": {}}, schema) == []

    errors = validate_object({"action": 42}, schema)

    assert "field action expected str, got int" in errors
    assert "missing required field: args" in errors


def test_parse_validated_applies_defaults():
    result = parse_validated(
        '{"action": "answer"}',
        {
            "action": str,
            "args": FieldSpec(dict, required=False, default={}),
        },
    )

    assert result.ok is True
    assert result.parsed == {"action": "answer", "args": {}}


def test_generate_structured_retries_with_validation_errors():
    prompts = []
    responses = iter(
        [
            "je pense qu'il faut appeler un outil",
            '{"action": "tool_call", "args": {"path": "README.md"}}',
        ]
    )

    def call_model(prompt):
        prompts.append(prompt)
        return next(responses)

    result = generate_structured(
        call_model,
        "Réponds en JSON.",
        {"action": str, "args": dict},
        max_retries=1,
    )

    assert result.ok is True
    assert result.attempts == 2
    assert result.parsed == {"action": "tool_call", "args": {"path": "README.md"}}
    assert "La réponse précédente ne respecte pas le contrat JSON." in prompts[1]

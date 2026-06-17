import json

from agent.tool_loop import parse_tool_action, run_tool_loop
from agent.tools import create_repo_tool_registry


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return next(self.responses)


def test_run_tool_loop_lets_model_call_tools_then_answer(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")
    trace_path = tmp_path / "tool_calls.jsonl"
    tools = create_repo_tool_registry(repo, trace_path=trace_path)
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "repo_graph",
                    "args": {"max_files": 20},
                    "reason": "cartographier le repo",
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "read_selected_context",
                    "args": {"paths": ["README.md"]},
                    "reason": "lire la doc",
                }
            ),
            json.dumps(
                {
                    "action": "answer",
                    "content": "Le repo contient une doc Demo.",
                    "memory_after": "README.md lu",
                }
            ),
        ]
    )

    result = run_tool_loop("Comprends le repo.", model, tools, max_steps=4)

    assert result.stopped_reason == "answer"
    assert result.answer == "Le repo contient une doc Demo."
    assert result.memory_after == "README.md lu"
    assert [step.action for step in result.steps] == ["tool_call", "tool_call", "answer"]
    assert result.steps[0].tool == "repo_graph"
    assert result.steps[0].observation["data"]["total_files"] == 1
    assert result.steps[1].tool == "read_selected_context"
    assert "# Demo" in result.steps[1].observation["data"]["rendered"]

    trace_lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["tool"] for line in trace_lines] == [
        "repo_graph",
        "read_selected_context",
    ]


def test_run_tool_loop_retries_invalid_json_action(tmp_path):
    tools = create_repo_tool_registry(tmp_path)
    model = ScriptedModel(
        [
            "je devrais regarder les fichiers",
            json.dumps({"action": "answer", "content": "Réponse après correction."}),
        ]
    )

    result = run_tool_loop("Réponds.", model, tools, max_steps=3)

    assert result.stopped_reason == "answer"
    assert result.steps[0].action == "invalid"
    assert "response did not contain a JSON object" in result.steps[0].error
    assert "Action JSON invalide" in model.calls[1][-1]["content"]


def test_run_tool_loop_rejects_future_retry_answer_after_tool_error(tmp_path):
    tools = create_repo_tool_registry(tmp_path)
    bad_patch = """*** Begin Patch
--- bad.py
+++ bad.py
*** End Patch"""
    good_patch = """*** Begin Patch
*** Add File: ok.py
+print('ok')
*** End Patch"""
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "action": "plan",
                    "files": ["ok.py"],
                    "steps": ["tester un mauvais patch", "corriger", "relire"],
                    "acceptance_criteria": ["ok.py existe"],
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "apply_patch",
                    "args": {"patch": bad_patch},
                }
            ),
            json.dumps(
                {
                    "action": "answer",
                    "content": "Je vais corriger cela et réessayer.",
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "apply_patch",
                    "args": {"patch": good_patch},
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "read_file",
                    "args": {"path": "ok.py"},
                }
            ),
            json.dumps({"action": "answer", "content": "Patch appliqué."}),
        ]
    )

    result = run_tool_loop("Crée un fichier.", model, tools, max_steps=6)

    assert result.stopped_reason == "answer"
    assert result.answer == "Patch appliqué."
    assert [step.action for step in result.steps] == [
        "plan",
        "tool_call",
        "invalid",
        "tool_call",
        "tool_call",
        "answer",
    ]
    assert "answer must be final" in result.steps[2].error
    assert (tmp_path / "ok.py").read_text(encoding="utf-8") == "print('ok')\n"


def test_run_tool_loop_rejects_claiming_dry_run_patch_was_applied(tmp_path):
    tools = create_repo_tool_registry(tmp_path)
    patch = """*** Begin Patch
*** Add File: ok.py
+print('ok')
*** End Patch"""
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "action": "plan",
                    "files": ["ok.py"],
                    "steps": ["valider le patch", "appliquer", "relire"],
                    "acceptance_criteria": ["ok.py existe"],
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "apply_patch",
                    "args": {"patch": patch, "dry_run": True},
                }
            ),
            json.dumps({"action": "answer", "content": "Le patch a été appliqué."}),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "apply_patch",
                    "args": {"patch": patch, "dry_run": False},
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "read_file",
                    "args": {"path": "ok.py"},
                }
            ),
            json.dumps({"action": "answer", "content": "Le patch a été appliqué."}),
        ]
    )

    result = run_tool_loop("Crée un fichier.", model, tools, max_steps=6)

    assert result.stopped_reason == "answer"
    assert [step.action for step in result.steps] == [
        "plan",
        "tool_call",
        "invalid",
        "tool_call",
        "tool_call",
        "answer",
    ]
    assert "dry_run=true" in result.steps[2].error
    assert (tmp_path / "ok.py").read_text(encoding="utf-8") == "print('ok')\n"


def test_run_tool_loop_requires_plan_before_write_file(tmp_path):
    tools = create_repo_tool_registry(tmp_path)
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "write_file",
                    "args": {"path": "index.html", "content": "<h1>Demo</h1>"},
                }
            ),
            json.dumps(
                {
                    "action": "plan",
                    "files": ["index.html"],
                    "steps": ["créer la page", "relire"],
                    "acceptance_criteria": ["index.html contient Demo"],
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "write_file",
                    "args": {"path": "index.html", "content": "<h1>Demo</h1>"},
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "read_file",
                    "args": {"path": "index.html"},
                }
            ),
            json.dumps({"action": "answer", "content": "Page créée."}),
        ]
    )

    result = run_tool_loop("Crée une page.", model, tools, max_steps=6)

    assert [step.action for step in result.steps] == [
        "invalid",
        "plan",
        "tool_call",
        "tool_call",
        "answer",
    ]
    assert "require an accepted plan" in result.steps[0].error
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "<h1>Demo</h1>"


def test_run_tool_loop_accepts_direct_tool_action_alias(tmp_path):
    tools = create_repo_tool_registry(tmp_path)
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "action": "plan",
                    "files": ["index.html"],
                    "steps": ["créer", "relire"],
                    "acceptance_criteria": ["index.html existe"],
                }
            ),
            json.dumps(
                {
                    "action": "write_file",
                    "args": {"path": "index.html", "content": "<h1>Demo</h1>"},
                    "reason": "écrire la page",
                }
            ),
            json.dumps({"action": "read_file", "args": {"path": "index.html"}}),
            json.dumps({"action": "answer", "content": "Page créée."}),
        ]
    )

    result = run_tool_loop("Crée une page.", model, tools, max_steps=4)

    assert result.stopped_reason == "answer"
    assert [step.action for step in result.steps] == [
        "plan",
        "tool_call",
        "tool_call",
        "answer",
    ]
    assert result.steps[1].tool == "write_file"
    assert result.steps[2].tool == "read_file"
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "<h1>Demo</h1>"


def test_run_tool_loop_stops_at_max_steps(tmp_path):
    tools = create_repo_tool_registry(tmp_path)
    model = ScriptedModel(
        [
            json.dumps({"action": "tool_call", "tool": "repo_graph", "args": {}}),
            json.dumps({"action": "tool_call", "tool": "repo_graph", "args": {}}),
        ]
    )

    result = run_tool_loop("Boucle courte.", model, tools, max_steps=2, memory="déjà là")

    assert result.stopped_reason == "max_steps"
    assert result.answer == ""
    assert result.memory_after == "déjà là"
    assert len(result.steps) == 2


def test_run_tool_loop_hints_available_tools_after_unknown_tool(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "script.js").write_text("function demo() {}\n", encoding="utf-8")
    tools = create_repo_tool_registry(repo)
    model = ScriptedModel(
        [
            json.dumps({"action": "tool_call", "tool": "script.js", "args": {}}),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "read_file",
                    "args": {"path": "script.js"},
                }
            ),
            json.dumps({"action": "answer", "content": "script.js lu."}),
        ]
    )

    result = run_tool_loop("Lis le script.", model, tools, max_steps=3)

    assert result.stopped_reason == "answer"
    assert result.steps[0].observation["error"] == "unknown tool: script.js"
    assert "read_file" in result.steps[0].observation["available_tools"]
    assert result.steps[0].observation["hint"] == (
        "Utilise uniquement un nom présent dans available_tools."
    )


def test_run_tool_loop_compacts_large_tool_history_before_next_call(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "big.txt").write_text("x" * 20_000, encoding="utf-8")
    tools = create_repo_tool_registry(repo)
    message_sizes = []

    def model(messages):
        message_sizes.append(sum(len(message.get("content", "")) for message in messages))
        if len(message_sizes) == 1:
            return json.dumps(
                {
                    "action": "tool_call",
                    "tool": "read_file",
                    "args": {"path": "big.txt", "max_chars": 20_000},
                }
            )
        return json.dumps({"action": "answer", "content": "big.txt lu."})

    result = run_tool_loop("Lis le gros fichier.", model, tools, max_steps=2, max_context_chars=3_000)

    assert result.stopped_reason == "answer"
    assert len(message_sizes) == 2
    assert message_sizes[1] < 5_000


def test_run_tool_loop_accepts_blocked_answer_after_failed_artifact_audit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    tools = create_repo_tool_registry(repo)
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "action": "plan",
                    "files": ["index.html"],
                    "steps": ["créer une page", "auditer"],
                    "acceptance_criteria": ["audit artefacts exécuté"],
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "write_file",
                    "args": {
                        "path": "index.html",
                        "content": "<!doctype html><div id='editor'></div>",
                    },
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "artifact_audit",
                    "args": {"prompt": "Crée un éditeur wysiwyg html/css/js."},
                }
            ),
            json.dumps(
                {
                    "action": "answer",
                    "content": "Bloqué: l'audit artefacts signale une page non exploitable.",
                }
            ),
        ]
    )

    result = run_tool_loop("Crée un éditeur.", model, tools, max_steps=4)

    assert result.stopped_reason == "answer"
    assert result.steps[2].tool == "artifact_audit"
    assert result.steps[2].observation["ok"] is False
    assert result.answer.startswith("Bloqué")


def test_run_tool_loop_requires_artifact_audit_when_requested(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    tools = create_repo_tool_registry(repo)
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "action": "plan",
                    "files": ["index.html"],
                    "steps": ["créer une page", "répondre"],
                    "acceptance_criteria": ["page créée"],
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "write_file",
                    "args": {
                        "path": "index.html",
                        "content": "<!doctype html><button>Ok</button>",
                    },
                }
            ),
            json.dumps({"action": "answer", "content": "Page créée."}),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "artifact_audit",
                    "args": {},
                }
            ),
            json.dumps(
                {
                    "action": "answer",
                    "content": "Bloqué: l'audit signale encore des problèmes.",
                }
            ),
        ]
    )

    result = run_tool_loop(
        "Crée un éditeur wysiwyg html/css/js.",
        model,
        tools,
        max_steps=5,
        require_artifact_audit=True,
    )

    assert result.steps[2].action == "invalid"
    assert "artifact_audit has not been called" in result.steps[2].error
    assert result.steps[3].tool == "artifact_audit"
    assert result.steps[3].args["prompt"] == "Crée un éditeur wysiwyg html/css/js."
    assert result.steps[3].observation["ok"] is False
    assert result.stopped_reason == "answer"


def test_parse_tool_action_validates_contract():
    parsed, errors = parse_tool_action(
        json.dumps({"action": "tool_call", "args": {}})
    )
    assert parsed["action"] == "tool_call"
    assert errors == ["tool_call requires tool"]

    parsed, errors = parse_tool_action(json.dumps({"action": "answer"}))
    assert parsed["action"] == "answer"
    assert errors == ["answer requires content"]

    parsed, errors = parse_tool_action(json.dumps({"action": "dance"}))
    assert parsed["action"] == "dance"
    assert errors == ["action must be plan, tool_call or answer"]

    parsed, errors = parse_tool_action(
        json.dumps({"action": "write_file", "args": {"path": "x.txt"}}),
        known_tools=["write_file"],
    )
    assert parsed["action"] == "tool_call"
    assert parsed["tool"] == "write_file"
    assert errors == []

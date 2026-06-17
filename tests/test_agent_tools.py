import json

from agent.tools import create_repo_tool_registry


def test_repo_tool_registry_exposes_specs_and_traces_calls(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")
    trace_path = tmp_path / "tool_calls.jsonl"

    tools = create_repo_tool_registry(repo, trace_path=trace_path)

    assert [spec.name for spec in tools.specs()] == [
        "apply_patch",
        "artifact_audit",
        "delete_file",
        "read_file",
        "read_selected_context",
        "repo_graph",
        "run_tests",
        "search_text",
        "write_file",
    ]

    graph = tools.call("repo_graph", {"max_files": 5})
    assert graph.ok is True
    assert graph.data["total_files"] == 1
    assert '"README.md"' in graph.data["prompt_json"]

    context = tools.call("read_selected_context", {"paths": ["README.md"]})
    assert context.ok is True
    assert context.data["files"][0]["path"] == "README.md"
    assert "# Demo" in context.data["rendered"]

    trace_lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["tool"] for line in trace_lines] == [
        "repo_graph",
        "read_selected_context",
    ]


def test_tool_registry_reports_unknown_tool(tmp_path):
    tools = create_repo_tool_registry(tmp_path)

    result = tools.call("missing_tool", {})

    assert result.ok is False
    assert result.error == "unknown tool: missing_tool"


def test_read_selected_context_requires_list_paths(tmp_path):
    tools = create_repo_tool_registry(tmp_path)

    result = tools.call("read_selected_context", {"paths": "README.md"})

    assert result.ok is False
    assert result.error == "paths must be a list"


def test_search_text_finds_bounded_matches_and_filters_glob(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def target():\n    return 'target'\n", encoding="utf-8")
    (repo / "README.md").write_text("target in docs\n", encoding="utf-8")
    tools = create_repo_tool_registry(repo)

    result = tools.call("search_text", {"query": "target", "glob": "*.py", "max_results": 1})

    assert result.ok is True
    assert result.data["truncated"] is True
    assert result.data["matches"] == [
        {"path": "app.py", "line": 1, "text": "def target():", "truncated": False}
    ]


def test_artifact_audit_tool_flags_blank_interactive_page(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "index.html").write_text(
        "<!doctype html><html><body><div id='editor'></div></body></html>",
        encoding="utf-8",
    )
    tools = create_repo_tool_registry(repo)

    result = tools.call(
        "artifact_audit",
        {"prompt": "Crée un éditeur wysiwyg html/css/js sur une même page."},
    )

    assert result.ok is False
    assert "artifact issue" in result.error
    assert result.data["editor_expected"] is True
    assert any("aucune surface editable" in note for note in result.data["notes"])


def test_read_file_reads_window_and_rejects_escape(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    tools = create_repo_tool_registry(repo)

    result = tools.call("read_file", {"path": "app.py", "start": 2, "max_lines": 2})

    assert result.ok is True
    assert result.data["content"] == "two\nthree"
    assert result.data["start"] == 2
    assert result.data["end"] == 3
    assert result.data["truncated"] is True

    escaped = tools.call("read_file", {"path": "../outside.py"})
    assert escaped.ok is False
    assert escaped.error == "path must stay inside repo"


def test_read_file_accepts_absolute_path_inside_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    app = repo / "app.py"
    app.write_text("print('ok')\n", encoding="utf-8")
    tools = create_repo_tool_registry(repo)

    result = tools.call("read_file", {"path": str(app)})

    assert result.ok is True
    assert result.data["path"] == "app.py"

    outside = tmp_path / "outside.py"
    outside.write_text("print('no')\n", encoding="utf-8")
    escaped = tools.call("read_file", {"path": str(outside)})
    assert escaped.ok is False
    assert escaped.error == "path must stay inside repo"


def test_file_tools_accept_common_path_aliases(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    tools = create_repo_tool_registry(repo)

    written = tools.call(
        "write_file",
        {"file_path": "index.html", "content": "<h1>Demo</h1>"},
    )

    assert written.ok is True
    read = tools.call("read_file", {"file": "index.html"})
    assert read.ok is True
    assert "<h1>Demo</h1>" in read.data["content"]
    deleted = tools.call("delete_file", {"filename": "index.html"})
    assert deleted.ok is True
    assert not (repo / "index.html").exists()


def test_run_tests_executes_pytest_on_relative_target(tmp_path):
    repo = tmp_path / "repo"
    tests_dir = repo / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_demo.py").write_text(
        "def test_demo():\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    tools = create_repo_tool_registry(repo)

    result = tools.call("run_tests", {"target": "tests/test_demo.py", "timeout_s": 20})

    assert result.ok is True
    assert result.data["exit_code"] == 0
    assert result.data["target"] == "tests/test_demo.py"
    assert "passed" in result.data["stdout"]


def test_run_tests_rejects_target_outside_repo(tmp_path):
    tools = create_repo_tool_registry(tmp_path)

    result = tools.call("run_tests", {"target": "../tests"})

    assert result.ok is False
    assert result.error == "target must stay inside repo"


def test_apply_patch_tool_supports_dry_run_and_apply(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("one\ntwo\n", encoding="utf-8")
    tools = create_repo_tool_registry(repo)
    patch = """*** Begin Patch
*** Update File: app.py
@@
 one
-two
+TWO
*** End Patch"""

    dry_run = tools.call("apply_patch", {"patch": patch, "dry_run": True})

    assert dry_run.ok is True
    assert dry_run.data == {"files_changed": ["app.py"], "dry_run": True}
    assert (repo / "app.py").read_text(encoding="utf-8") == "one\ntwo\n"

    applied = tools.call("apply_patch", {"patch": patch})

    assert applied.ok is True
    assert applied.data == {"files_changed": ["app.py"], "dry_run": False}
    assert (repo / "app.py").read_text(encoding="utf-8") == "one\nTWO\n"


def test_apply_patch_tool_rejects_excluded_paths(tmp_path):
    tools = create_repo_tool_registry(tmp_path)
    patch = """*** Begin Patch
*** Add File: runs/output.txt
+nope
*** End Patch"""

    result = tools.call("apply_patch", {"patch": patch})

    assert result.ok is False
    assert "excluded directory" in result.error


def test_write_file_tool_creates_text_file_and_respects_overwrite(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    tools = create_repo_tool_registry(repo)

    result = tools.call(
        "write_file",
        {"path": "demo/index.html", "content": "<h1>Demo</h1>\n"},
    )

    assert result.ok is True
    assert result.data == {
        "path": "demo/index.html",
        "chars": 14,
        "overwrite": False,
    }
    assert (repo / "demo" / "index.html").read_text(encoding="utf-8") == "<h1>Demo</h1>\n"

    blocked = tools.call("write_file", {"path": "demo/index.html", "content": "x"})
    assert blocked.ok is False
    assert blocked.error == "file exists and overwrite is false"

    overwritten = tools.call(
        "write_file",
        {"path": "demo/index.html", "content": "x", "overwrite": True},
    )
    assert overwritten.ok is True
    assert (repo / "demo" / "index.html").read_text(encoding="utf-8") == "x"


def test_write_file_tool_accepts_absolute_path_inside_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "demo" / "index.html"
    tools = create_repo_tool_registry(repo)

    result = tools.call("write_file", {"path": str(target), "content": "<h1>Demo</h1>"})

    assert result.ok is True
    assert result.data["path"] == "demo/index.html"
    assert target.read_text(encoding="utf-8") == "<h1>Demo</h1>"


def test_write_file_tool_rejects_unsafe_or_non_text_paths(tmp_path):
    tools = create_repo_tool_registry(tmp_path)

    escaped = tools.call("write_file", {"path": "../x.html", "content": "x"})
    assert escaped.ok is False
    assert escaped.error == "path must stay inside repo"

    binary = tools.call("write_file", {"path": "image.bin", "content": "x"})
    assert binary.ok is False
    assert binary.error == "path is not a supported text file"


def test_delete_file_tool_removes_text_file_and_rejects_unsafe_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "orphan.css"
    target.write_text("body {}\n", encoding="utf-8")
    tools = create_repo_tool_registry(repo)

    result = tools.call("delete_file", {"path": "orphan.css"})

    assert result.ok is True
    assert result.data == {"path": "orphan.css"}
    assert not target.exists()

    escaped = tools.call("delete_file", {"path": "../outside.css"})
    assert escaped.ok is False
    assert escaped.error == "path must stay inside repo"

    missing = tools.call("delete_file", {"path": "missing.css"})
    assert missing.ok is False
    assert missing.error == "path is not a file"

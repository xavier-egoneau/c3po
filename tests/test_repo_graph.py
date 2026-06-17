from agent.repo_graph import (
    build_repo_graph,
    load_selected_context,
    render_repo_graph_for_prompt,
)


def test_build_repo_graph_extracts_python_shape_and_skips_excluded_dirs(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text(
        "import json\nfrom pathlib import Path\n\nclass Engine:\n    pass\n\ndef run():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_core.py").write_text("def test_run():\n    pass\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "ignored.py").write_text("def hidden():\n    pass\n", encoding="utf-8")

    graph = build_repo_graph(tmp_path)
    files = {item.path: item for item in graph.files}

    assert "pkg/core.py" in files
    assert "tests/test_core.py" in files
    assert ".git/ignored.py" not in files
    assert files["pkg/core.py"].symbols == ["Engine", "run"]
    assert files["pkg/core.py"].imports == ["json", "pathlib"]
    assert files["tests/test_core.py"].role == "test"
    assert any(directory.path == "pkg" for directory in graph.directories)


def test_render_repo_graph_for_prompt_is_compact_json(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")

    rendered = render_repo_graph_for_prompt(build_repo_graph(tmp_path))

    assert '"total_files":1' in rendered
    assert "\n" not in rendered


def test_load_selected_context_filters_paths_and_truncates(tmp_path):
    (tmp_path / "README.md").write_text("abcdef", encoding="utf-8")
    (tmp_path / "data.bin").write_bytes(b"\x00\x01")

    context = load_selected_context(
        tmp_path,
        ["README.md", "../outside.txt", "data.bin"],
        max_chars_per_file=3,
    )

    assert len(context) == 1
    assert context[0]["path"] == "README.md"
    assert context[0]["content"] == "abc"
    assert context[0]["truncated"] is True


def test_load_selected_context_accepts_directories(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("print('b')\n", encoding="utf-8")

    context = load_selected_context(tmp_path, ["pkg"], max_files=1)

    assert len(context) == 1
    assert context[0]["path"] == "pkg/a.py"

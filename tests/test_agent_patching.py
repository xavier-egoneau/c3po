from agent.patching import apply_patch_text, parse_patch, validate_patch_paths


def test_apply_patch_text_adds_updates_and_deletes_files(tmp_path):
    (tmp_path / "app.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (tmp_path / "old.txt").write_text("bye\n", encoding="utf-8")
    patch = """*** Begin Patch
*** Add File: docs/new.md
+# New
+content
*** Update File: app.py
@@
 one
-two
+TWO
 three
*** Delete File: old.txt
*** End Patch"""

    result = apply_patch_text(patch, tmp_path)

    assert result.ok is True
    assert result.files_changed == ["docs/new.md", "app.py", "old.txt"]
    assert (tmp_path / "docs" / "new.md").read_text(encoding="utf-8") == "# New\ncontent\n"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "one\nTWO\nthree\n"
    assert not (tmp_path / "old.txt").exists()


def test_apply_patch_text_dry_run_does_not_write(tmp_path):
    (tmp_path / "app.py").write_text("one\ntwo\n", encoding="utf-8")
    patch = """*** Begin Patch
*** Update File: app.py
@@
 one
-two
+TWO
*** End Patch"""

    result = apply_patch_text(patch, tmp_path, dry_run=True)

    assert result.ok is True
    assert result.files_changed == ["app.py"]
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "one\ntwo\n"


def test_apply_patch_text_rejects_non_matching_hunk_without_partial_write(tmp_path):
    (tmp_path / "app.py").write_text("one\ntwo\n", encoding="utf-8")
    patch = """*** Begin Patch
*** Add File: created.txt
+created
*** Update File: app.py
@@
 missing
-two
+TWO
*** End Patch"""

    result = apply_patch_text(patch, tmp_path)

    assert result.ok is False
    assert "did not match" in result.error
    assert not (tmp_path / "created.txt").exists()
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "one\ntwo\n"


def test_validate_patch_paths_rejects_unsafe_paths(tmp_path):
    patch = """*** Begin Patch
*** Add File: ../escape.txt
+nope
*** End Patch"""

    try:
        validate_patch_paths(patch, tmp_path)
    except ValueError as exc:
        assert "parent traversal" in str(exc)
    else:
        raise AssertionError("expected unsafe path to fail")


def test_parse_patch_requires_patch_markers():
    try:
        parse_patch("*** Add File: nope\n+x")
    except ValueError as exc:
        assert "must start" in str(exc)
    else:
        raise AssertionError("expected invalid patch to fail")

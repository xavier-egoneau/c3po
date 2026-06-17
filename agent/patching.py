"""Application de patchs contrôlés pour outils agentiques."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agent.repo_graph import DEFAULT_EXCLUDED_DIRS


PatchKind = Literal["add", "update", "delete"]


@dataclass
class PatchHunk:
    lines: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class PatchOperation:
    kind: PatchKind
    path: str
    lines: list[str] = field(default_factory=list)
    hunks: list[PatchHunk] = field(default_factory=list)


@dataclass
class PatchResult:
    ok: bool
    files_changed: list[str] = field(default_factory=list)
    error: str | None = None


class PatchError(ValueError):
    pass


def validate_patch_paths(patch: str, root: str | Path) -> list[str]:
    operations = parse_patch(patch)
    root_path = Path(root).resolve()
    paths: list[str] = []
    for operation in operations:
        _resolve_patch_path(root_path, operation.path)
        paths.append(operation.path)
    return paths


def apply_patch_text(patch: str, root: str | Path, dry_run: bool = False) -> PatchResult:
    root_path = Path(root).resolve()
    try:
        operations = parse_patch(patch)
        virtual: dict[str, str | None] = {}
        changed: list[str] = []
        for operation in operations:
            path = _resolve_patch_path(root_path, operation.path)
            rel = path.relative_to(root_path).as_posix()
            current = virtual.get(rel)
            if current is None and rel not in virtual and path.exists():
                current = path.read_text(encoding="utf-8", errors="replace")

            if operation.kind == "add":
                if path.exists() or rel in virtual:
                    raise PatchError(f"file already exists: {rel}")
                virtual[rel] = "\n".join(operation.lines) + "\n"
            elif operation.kind == "delete":
                if current is None and not path.exists():
                    raise PatchError(f"file does not exist: {rel}")
                virtual[rel] = None
            elif operation.kind == "update":
                if current is None:
                    raise PatchError(f"file does not exist: {rel}")
                virtual[rel] = _apply_update(current, operation.hunks)

            if rel not in changed:
                changed.append(rel)

        if not dry_run:
            for rel, content in virtual.items():
                path = root_path / rel
                if content is None:
                    if path.exists():
                        path.unlink()
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

        return PatchResult(ok=True, files_changed=changed)
    except (OSError, PatchError, ValueError) as exc:
        return PatchResult(ok=False, error=str(exc))


def parse_patch(patch: str) -> list[PatchOperation]:
    lines = patch.splitlines()
    if not lines or lines[0] != "*** Begin Patch":
        raise PatchError("patch must start with *** Begin Patch")
    if lines[-1] != "*** End Patch":
        raise PatchError("patch must end with *** End Patch")

    operations: list[PatchOperation] = []
    index = 1
    while index < len(lines) - 1:
        line = lines[index]
        if line.startswith("*** Add File: "):
            operation, index = _parse_add(lines, index)
        elif line.startswith("*** Update File: "):
            operation, index = _parse_update(lines, index)
        elif line.startswith("*** Delete File: "):
            operation = PatchOperation(kind="delete", path=line.removeprefix("*** Delete File: ").strip())
            index += 1
        else:
            raise PatchError(f"unexpected patch line: {line}")
        operations.append(operation)

    if not operations:
        raise PatchError("patch does not contain operations")
    return operations


def _parse_add(lines: list[str], index: int) -> tuple[PatchOperation, int]:
    path = lines[index].removeprefix("*** Add File: ").strip()
    index += 1
    content: list[str] = []
    while index < len(lines) and not _is_operation_or_end(lines[index]):
        line = lines[index]
        if not line.startswith("+"):
            raise PatchError("add file lines must start with +")
        content.append(line[1:])
        index += 1
    return PatchOperation(kind="add", path=path, lines=content), index


def _parse_update(lines: list[str], index: int) -> tuple[PatchOperation, int]:
    path = lines[index].removeprefix("*** Update File: ").strip()
    index += 1
    hunks: list[PatchHunk] = []
    current: PatchHunk | None = None

    while index < len(lines) and not _is_operation_or_end(lines[index]):
        line = lines[index]
        if line.startswith("@@"):
            if current is not None:
                hunks.append(current)
            current = PatchHunk()
            index += 1
            continue
        if current is None:
            raise PatchError("update file requires a @@ hunk before changes")
        if not line:
            raise PatchError("hunk lines must start with space, +, or -")
        op = line[0]
        if op not in {" ", "+", "-"}:
            raise PatchError("hunk lines must start with space, +, or -")
        current.lines.append((op, line[1:]))
        index += 1

    if current is not None:
        hunks.append(current)
    if not hunks:
        raise PatchError("update file requires at least one hunk")
    return PatchOperation(kind="update", path=path, hunks=hunks), index


def _is_operation_or_end(line: str) -> bool:
    return (
        line.startswith("*** Add File: ")
        or line.startswith("*** Update File: ")
        or line.startswith("*** Delete File: ")
        or line == "*** End Patch"
    )


def _apply_update(content: str, hunks: list[PatchHunk]) -> str:
    had_trailing_newline = content.endswith("\n")
    current = content.splitlines()
    cursor = 0
    for hunk in hunks:
        old_block = [text for op, text in hunk.lines if op in {" ", "-"}]
        new_block = [text for op, text in hunk.lines if op in {" ", "+"}]
        match_at = _find_block(current, old_block, cursor)
        if match_at is None:
            raise PatchError("update hunk did not match file content")
        current = current[:match_at] + new_block + current[match_at + len(old_block):]
        cursor = match_at + len(new_block)
    result = "\n".join(current)
    if had_trailing_newline:
        result += "\n"
    return result


def _find_block(lines: list[str], block: list[str], start: int) -> int | None:
    if not block:
        return start
    max_start = len(lines) - len(block)
    for index in range(start, max_start + 1):
        if lines[index:index + len(block)] == block:
            return index
    return None


def _resolve_patch_path(root: Path, raw_path: str) -> Path:
    if not raw_path:
        raise PatchError("patch path is empty")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise PatchError(f"absolute patch path is forbidden: {raw_path}")
    if any(part == ".." for part in candidate.parts):
        raise PatchError(f"parent traversal is forbidden: {raw_path}")
    if any(part in DEFAULT_EXCLUDED_DIRS for part in candidate.parts):
        raise PatchError(f"patch path uses an excluded directory: {raw_path}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PatchError(f"patch path escapes repo: {raw_path}") from exc
    return resolved

"""Carte compacte d'un repo pour orienter un petit modèle sans saturer son contexte."""

from __future__ import annotations

import ast
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDED_DIRS = {
    ".claude",
    ".codex",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "dist",
    "htmlcov",
    "models",
    "node_modules",
    "runs",
    "temp",
    "tmp",
}

TEXT_EXTENSIONS = {
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass
class RepoFile:
    path: str
    kind: str
    size: int
    role: str
    symbols: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


@dataclass
class RepoDirectory:
    path: str
    files: int
    bytes: int
    roles: dict[str, int] = field(default_factory=dict)


@dataclass
class RepoGraph:
    root: str
    files: list[RepoFile]
    directories: list[RepoDirectory]
    excluded_dirs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repo_graph_from_dict(data: dict[str, Any]) -> RepoGraph:
    return RepoGraph(
        root=str(data.get("root", "")),
        files=[
            RepoFile(
                path=str(item.get("path", "")),
                kind=str(item.get("kind", "")),
                size=int(item.get("size", 0)),
                role=str(item.get("role", "")),
                symbols=[str(value) for value in item.get("symbols", [])],
                imports=[str(value) for value in item.get("imports", [])],
            )
            for item in data.get("files", [])
            if isinstance(item, dict)
        ],
        directories=[
            RepoDirectory(
                path=str(item.get("path", "")),
                files=int(item.get("files", 0)),
                bytes=int(item.get("bytes", 0)),
                roles={
                    str(key): int(value)
                    for key, value in item.get("roles", {}).items()
                },
            )
            for item in data.get("directories", [])
            if isinstance(item, dict)
        ],
        excluded_dirs=[str(value) for value in data.get("excluded_dirs", [])],
    )


def build_repo_graph(
    root: str | Path,
    excluded_dirs: set[str] | None = None,
    max_parse_bytes: int = 300_000,
) -> RepoGraph:
    root_path = Path(root).resolve()
    excluded = excluded_dirs or DEFAULT_EXCLUDED_DIRS
    files: list[RepoFile] = []
    directories: dict[str, RepoDirectory] = {}

    for current_root, dirnames, filenames in os.walk(root_path):
        dirnames[:] = sorted(dirname for dirname in dirnames if dirname not in excluded)
        for filename in sorted(filenames):
            path = Path(current_root) / filename
            if _is_excluded(path, root_path, excluded):
                continue
            if not path.is_file():
                continue
            rel = path.relative_to(root_path).as_posix()
            size = _safe_size(path)
            role = _role_for(rel)
            kind = _kind_for(path)
            symbols: list[str] = []
            imports: list[str] = []
            if path.suffix == ".py" and size <= max_parse_bytes:
                symbols, imports = _python_shape(path)
            files.append(
                RepoFile(
                    path=rel,
                    kind=kind,
                    size=size,
                    role=role,
                    symbols=symbols,
                    imports=imports,
                )
            )
            _add_directory_stats(directories, rel, size, role)

    return RepoGraph(
        root=str(root_path),
        files=files,
        directories=sorted(directories.values(), key=lambda item: item.path),
        excluded_dirs=sorted(excluded),
    )


def render_repo_graph_for_prompt(graph: RepoGraph, max_files: int = 220) -> str:
    """Rend le graph en JSON compact et borné pour un prompt de sélection."""
    data = graph.to_dict()
    data["files"] = data["files"][:max_files]
    data["truncated"] = len(graph.files) > max_files
    data["total_files"] = len(graph.files)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def load_selected_context(
    root: str | Path,
    selected_paths: list[str],
    max_files: int = 8,
    max_chars_per_file: int = 12_000,
) -> list[dict[str, Any]]:
    root_path = Path(root).resolve()
    context: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in selected_paths:
        if len(context) >= max_files:
            break
        rel = str(raw_path).strip()
        if not rel or rel in seen:
            continue
        seen.add(rel)
        path = (root_path / rel).resolve()
        try:
            path.relative_to(root_path)
        except ValueError:
            continue
        if path.is_dir():
            for child in _iter_context_files(path, root_path):
                if len(context) >= max_files:
                    break
                _append_context_file(context, child, root_path, max_chars_per_file)
            continue
        if not path.is_file() or path.suffix not in TEXT_EXTENSIONS:
            continue
        _append_context_file(context, path, root_path, max_chars_per_file)
    return context


def render_selected_context(context: list[dict[str, Any]]) -> str:
    if not context:
        return "Aucun fichier sélectionné."
    blocks = []
    for item in context:
        if "error" in item:
            blocks.append(f"## {item['path']}\n[Erreur lecture: {item['error']}]")
            continue
        suffix = "\n[TRONQUÉ]" if item.get("truncated") else ""
        blocks.append(f"## {item['path']}\n```text\n{item.get('content', '')}{suffix}\n```")
    return "\n\n".join(blocks)


def _is_excluded(path: Path, root: Path, excluded_dirs: set[str]) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in excluded_dirs for part in parts[:-1])


def _iter_context_files(path: Path, root: Path):
    for child in sorted(path.rglob("*")):
        try:
            parts = child.relative_to(root).parts
        except ValueError:
            continue
        if any(part in DEFAULT_EXCLUDED_DIRS for part in parts[:-1]):
            continue
        if child.is_file() and child.suffix in TEXT_EXTENSIONS:
            yield child


def _append_context_file(
    context: list[dict[str, Any]],
    path: Path,
    root: Path,
    max_chars_per_file: int,
) -> None:
    rel = path.relative_to(root).as_posix()
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        context.append({"path": rel, "error": str(exc)})
        return
    truncated = len(content) > max_chars_per_file
    context.append(
        {
            "path": rel,
            "chars": len(content),
            "truncated": truncated,
            "content": content[:max_chars_per_file],
        }
    )


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _kind_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in {".md", ".txt"}:
        return "doc"
    if suffix in {".toml", ".yaml", ".yml", ".json", ".ini", ".cfg"}:
        return "config"
    if suffix in {".sh"}:
        return "script"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    return "binary_or_other"


def _role_for(rel_path: str) -> str:
    parts = rel_path.split("/")
    name = parts[-1].lower()
    if parts[0] == "tests" or name.startswith("test_"):
        return "test"
    if parts[0] in {"docs", "experiments", "scripts"}:
        return parts[0]
    if name.startswith("readme") or name in {"architecture.md", "context.md"}:
        return "doc"
    if name in {"pyproject.toml", "package.json", "requirements.txt"}:
        return "project_config"
    return "source"


def _python_shape(path: Path) -> tuple[list[str], list[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return [], []

    symbols: list[str] = []
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
    return _unique_limited(symbols, 32), _unique_limited(imports, 32)


def _unique_limited(values: list[str], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def _add_directory_stats(
    directories: dict[str, RepoDirectory],
    rel_path: str,
    size: int,
    role: str,
) -> None:
    parts = rel_path.split("/")[:-1]
    for depth in range(1, len(parts) + 1):
        rel_dir = "/".join(parts[:depth])
        entry = directories.setdefault(
            rel_dir,
            RepoDirectory(path=rel_dir, files=0, bytes=0, roles={}),
        )
        entry.files += 1
        entry.bytes += size
        entry.roles[role] = entry.roles.get(role, 0) + 1

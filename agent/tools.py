"""Registry d'outils agentiques réutilisables.

Cette couche exécute des outils locaux contrôlés et retourne des observations
structurées. Elle reste indépendante du modèle : un orchestrateur peut décider
quand exposer un outil, comment tracer l'appel, et quoi remettre dans le contexte.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent.artifact_audit import audit_repo_artifacts
from agent.patching import apply_patch_text
from agent.repo_graph import (
    DEFAULT_EXCLUDED_DIRS,
    TEXT_EXTENSIONS,
    build_repo_graph,
    load_selected_context,
    render_repo_graph_for_prompt,
    render_selected_context,
)


ToolHandler = Callable[[dict[str, Any]], "ToolResult"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    summary: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolTraceWriter:
    def __init__(self, path: str | Path, live: bool = False):
        self.path = Path(path)
        self.live = live
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, result: ToolResult, args: dict[str, Any]) -> None:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool": result.tool,
            "args": args,
            **result.to_dict(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        if self.live:
            print(_format_live_tool_event(event), file=sys.stderr, flush=True)


class AgentToolRegistry:
    def __init__(self, tracer: ToolTraceWriter | None = None):
        self._handlers: dict[str, ToolHandler] = {}
        self._specs: dict[str, ToolSpec] = {}
        self.tracer = tracer

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.name in self._handlers:
            raise ValueError(f"tool already registered: {spec.name}")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in sorted(self._specs)]

    def call(self, tool: str, args: dict[str, Any] | None = None) -> ToolResult:
        args = dict(args or {})
        start = time.time()
        handler = self._handlers.get(tool)
        if handler is None:
            result = ToolResult(
                tool=tool,
                ok=False,
                error=f"unknown tool: {tool}",
                duration_s=round(time.time() - start, 4),
            )
        else:
            try:
                result = handler(args)
                result.duration_s = round(time.time() - start, 4)
            except Exception as exc:
                result = ToolResult(
                    tool=tool,
                    ok=False,
                    error=str(exc),
                    duration_s=round(time.time() - start, 4),
                )
        if self.tracer is not None:
            self.tracer.write(result, args)
        return result


def create_repo_tool_registry(
    repo_root: str | Path,
    trace_path: str | Path | None = None,
    live: bool = False,
) -> AgentToolRegistry:
    tracer = ToolTraceWriter(trace_path, live=live) if trace_path is not None else None
    registry = AgentToolRegistry(tracer=tracer)
    root = Path(repo_root).resolve()

    registry.register(
        ToolSpec(
            name="repo_graph",
            description=(
                "Construit une carte compacte du repo : fichiers, dossiers, tailles, "
                "rôles, symboles et imports Python."
            ),
            args_schema={
                "max_files": "int optional, nombre de fichiers inclus dans prompt_json",
                "max_parse_bytes": "int optional, taille max des fichiers Python parsés",
            },
        ),
        lambda args: _tool_repo_graph(root, args),
    )
    registry.register(
        ToolSpec(
            name="artifact_audit",
            description=(
                "Audite les artefacts produits dans le repo: imports, références, "
                "dépendances, fichiers reliés, symboles appelés, liens HTML/CSS/JS, "
                "éléments DOM absents et placeholders."
            ),
            args_schema={
                "prompt": "str optional, intention utilisateur pour deduire les criteres",
            },
        ),
        lambda args: _tool_artifact_audit(root, args),
    )
    registry.register(
        ToolSpec(
            name="read_selected_context",
            description=(
                "Lit un paquet borné de fichiers sélectionnés dans le repo et retourne "
                "à la fois les extraits structurés et un rendu prêt pour prompt."
            ),
            args_schema={
                "paths": "list[str], chemins repo relatifs",
                "max_files": "int optional, défaut 8",
                "max_chars_per_file": "int optional, défaut 12000",
            },
        ),
        lambda args: _tool_read_selected_context(root, args),
    )
    registry.register(
        ToolSpec(
            name="search_text",
            description=(
                "Cherche une chaîne dans les fichiers texte du repo, sans charger "
                "tout le contenu dans le contexte."
            ),
            args_schema={
                "query": "str, texte à chercher",
                "glob": "str optional, suffixe ou motif simple comme *.py",
                "max_results": "int optional, défaut 40",
                "max_line_chars": "int optional, défaut 240",
            },
        ),
        lambda args: _tool_search_text(root, args),
    )
    registry.register(
        ToolSpec(
            name="read_file",
            description=(
                "Lit un fichier texte précis avec fenêtre de lignes et limite de "
                "caractères."
            ),
            args_schema={
                "path": "str, chemin repo relatif",
                "start": "int optional, ligne 1-based",
                "max_lines": "int optional, défaut 160",
                "max_chars": "int optional, défaut 20000",
            },
        ),
        lambda args: _tool_read_file(root, args),
    )
    registry.register(
        ToolSpec(
            name="delete_file",
            description=(
                "Supprime un fichier texte relatif au repo. Utile pour nettoyer "
                "un artefact généré mais orphelin ou contradictoire."
            ),
            args_schema={
                "path": "str, chemin repo relatif du fichier texte à supprimer",
            },
        ),
        lambda args: _tool_delete_file(root, args),
    )
    registry.register(
        ToolSpec(
            name="run_tests",
            description=(
                "Lance python -m pytest sur une cible relative contrôlée et retourne "
                "stdout/stderr bornés."
            ),
            args_schema={
                "target": "str optional, fichier/dossier de tests relatif",
                "timeout_s": "int optional, défaut 60",
            },
        ),
        lambda args: _tool_run_tests(root, args),
    )
    registry.register(
        ToolSpec(
            name="apply_patch",
            description=(
                "Applique un patch contrôlé au format *** Begin Patch. Les chemins "
                "doivent rester relatifs au repo et hors dossiers exclus. Format : "
                "*** Add File: path avec lignes +..., ou *** Update File: path, @@, "
                "puis lignes préfixées par espace/+/- ; pas de unified diff ---/+++."
            ),
            args_schema={
                "patch": "str, patch au format *** Begin Patch",
                "dry_run": "bool optional, valide sans écrire si true",
            },
        ),
        lambda args: _tool_apply_patch(root, args),
    )
    registry.register(
        ToolSpec(
            name="write_file",
            description=(
                "Crée ou remplace un fichier texte entier. Plus simple que apply_patch "
                "pour générer de nouveaux fichiers HTML/CSS/JS."
            ),
            args_schema={
                "path": "str, chemin repo relatif",
                "content": "str, contenu complet du fichier",
                "overwrite": "bool optional, défaut false",
            },
        ),
        lambda args: _tool_write_file(root, args),
    )
    return registry


def _tool_repo_graph(root: Path, args: dict[str, Any]) -> ToolResult:
    max_files = _positive_int(args.get("max_files"), default=220, maximum=1000)
    max_parse_bytes = _positive_int(
        args.get("max_parse_bytes"),
        default=300_000,
        maximum=2_000_000,
    )
    graph = build_repo_graph(root, max_parse_bytes=max_parse_bytes)
    prompt_json = render_repo_graph_for_prompt(graph, max_files=max_files)
    return ToolResult(
        tool="repo_graph",
        ok=True,
        data={
            "root": str(root),
            "graph": graph.to_dict(),
            "prompt_json": prompt_json,
            "total_files": len(graph.files),
            "total_directories": len(graph.directories),
        },
        summary=f"{len(graph.files)} fichiers, {len(graph.directories)} dossiers",
    )


def _tool_artifact_audit(root: Path, args: dict[str, Any]) -> ToolResult:
    prompt = str(args.get("prompt", "") or "")
    audit = audit_repo_artifacts(root, prompt=prompt)
    notes = audit.get("notes", [])
    note_count = len(notes) if isinstance(notes, list) else 0
    return ToolResult(
        tool="artifact_audit",
        ok=note_count == 0,
        error=None if note_count == 0 else f"{note_count} artifact issue(s)",
        data=audit,
        summary=f"{note_count} issue(s), {len(audit.get('html_files', []))} html file(s)",
    )


def _tool_read_selected_context(root: Path, args: dict[str, Any]) -> ToolResult:
    paths = args.get("paths", [])
    if not isinstance(paths, list):
        return ToolResult(
            tool="read_selected_context",
            ok=False,
            error="paths must be a list",
        )
    max_files = _positive_int(args.get("max_files"), default=8, maximum=50)
    max_chars_per_file = _positive_int(
        args.get("max_chars_per_file"),
        default=12_000,
        maximum=100_000,
    )
    context = load_selected_context(
        root,
        [str(path) for path in paths],
        max_files=max_files,
        max_chars_per_file=max_chars_per_file,
    )
    rendered = render_selected_context(context)
    return ToolResult(
        tool="read_selected_context",
        ok=True,
        data={
            "root": str(root),
            "files": context,
            "rendered": rendered,
        },
        summary=f"{len(context)} fichier(s) retenu(s)",
    )


def _tool_search_text(root: Path, args: dict[str, Any]) -> ToolResult:
    query = str(args.get("query", ""))
    if not query:
        return ToolResult(tool="search_text", ok=False, error="query is required")
    glob = str(args.get("glob", "") or "")
    max_results = _positive_int(args.get("max_results"), default=40, maximum=500)
    max_line_chars = _positive_int(args.get("max_line_chars"), default=240, maximum=2000)

    matches: list[dict[str, Any]] = []
    for path in _iter_text_files(root):
        rel = path.relative_to(root).as_posix()
        if glob and not _matches_simple_glob(rel, glob):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            if query not in line:
                continue
            matches.append(
                {
                    "path": rel,
                    "line": line_no,
                    "text": line[:max_line_chars],
                    "truncated": len(line) > max_line_chars,
                }
            )
            if len(matches) >= max_results:
                return ToolResult(
                    tool="search_text",
                    ok=True,
                    data={"query": query, "matches": matches, "truncated": True},
                    summary=f"{len(matches)} résultat(s), tronqué",
                )
    return ToolResult(
        tool="search_text",
        ok=True,
        data={"query": query, "matches": matches, "truncated": False},
        summary=f"{len(matches)} résultat(s)",
    )


def _tool_read_file(root: Path, args: dict[str, Any]) -> ToolResult:
    raw_path = _path_arg(args)
    path = _safe_repo_path(root, raw_path)
    if path is None:
        return ToolResult(tool="read_file", ok=False, error="path must stay inside repo")
    if not path.is_file():
        return ToolResult(tool="read_file", ok=False, error="path is not a file")
    if path.suffix not in TEXT_EXTENSIONS:
        return ToolResult(tool="read_file", ok=False, error="path is not a supported text file")

    start = _positive_int(args.get("start"), default=1, maximum=1_000_000)
    max_lines = _positive_int(args.get("max_lines"), default=160, maximum=2000)
    max_chars = _positive_int(args.get("max_chars"), default=20_000, maximum=200_000)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return ToolResult(tool="read_file", ok=False, error=str(exc))

    start_index = max(0, start - 1)
    selected = lines[start_index:start_index + max_lines]
    content = "\n".join(selected)
    truncated_chars = len(content) > max_chars
    content = content[:max_chars]
    truncated_lines = start_index + max_lines < len(lines)
    rel = path.relative_to(root).as_posix()
    return ToolResult(
        tool="read_file",
        ok=True,
        data={
            "path": rel,
            "start": start,
            "end": start + len(selected) - 1 if selected else start,
            "total_lines": len(lines),
            "content": content,
            "truncated": truncated_lines or truncated_chars,
        },
        summary=f"{rel}:{start}-{start + len(selected) - 1 if selected else start}",
    )


def _tool_run_tests(root: Path, args: dict[str, Any]) -> ToolResult:
    raw_target = str(args.get("target", "tests") or "tests")
    target = _safe_repo_path(root, raw_target)
    if target is None:
        return ToolResult(tool="run_tests", ok=False, error="target must stay inside repo")
    if not target.exists():
        return ToolResult(tool="run_tests", ok=False, error="target does not exist")
    timeout_s = _positive_int(args.get("timeout_s"), default=60, maximum=600)
    rel_target = target.relative_to(root).as_posix()
    cmd = [sys.executable, "-m", "pytest", rel_target, "-q"]
    try:
        completed = subprocess.run(
            cmd,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return ToolResult(
            tool="run_tests",
            ok=False,
            error=f"pytest timed out after {timeout_s}s",
            data={
                "target": rel_target,
                "stdout": _tail(exc.stdout or ""),
                "stderr": _tail(exc.stderr or ""),
                "timeout_s": timeout_s,
            },
            summary="timeout",
        )
    return ToolResult(
        tool="run_tests",
        ok=completed.returncode == 0,
        error=None if completed.returncode == 0 else f"pytest exited {completed.returncode}",
        data={
            "target": rel_target,
            "exit_code": completed.returncode,
            "stdout": _tail(completed.stdout),
            "stderr": _tail(completed.stderr),
            "timeout_s": timeout_s,
        },
        summary=f"pytest {rel_target} exit={completed.returncode}",
    )


def _tool_apply_patch(root: Path, args: dict[str, Any]) -> ToolResult:
    patch = str(args.get("patch", ""))
    if not patch:
        return ToolResult(tool="apply_patch", ok=False, error="patch is required")
    dry_run = bool(args.get("dry_run", False))
    result = apply_patch_text(patch, root, dry_run=dry_run)
    return ToolResult(
        tool="apply_patch",
        ok=result.ok,
        error=result.error,
        data={
            "files_changed": result.files_changed,
            "dry_run": dry_run,
        },
        summary=(
            f"{'validated' if dry_run else 'applied'} "
            f"{len(result.files_changed)} file(s)"
            if result.ok
            else "patch failed"
        ),
    )


def _tool_write_file(root: Path, args: dict[str, Any]) -> ToolResult:
    raw_path = _path_arg(args)
    content = args.get("content", "")
    if not isinstance(content, str):
        return ToolResult(tool="write_file", ok=False, error="content must be a string")
    path = _safe_repo_path(root, raw_path)
    if path is None:
        return ToolResult(tool="write_file", ok=False, error="path must stay inside repo")
    if any(part in DEFAULT_EXCLUDED_DIRS for part in path.relative_to(root).parts):
        return ToolResult(tool="write_file", ok=False, error="path uses an excluded directory")
    if path.suffix not in TEXT_EXTENSIONS:
        return ToolResult(tool="write_file", ok=False, error="path is not a supported text file")
    overwrite = bool(args.get("overwrite", False))
    if path.exists() and not overwrite:
        return ToolResult(tool="write_file", ok=False, error="file exists and overwrite is false")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    rel = path.relative_to(root).as_posix()
    return ToolResult(
        tool="write_file",
        ok=True,
        data={"path": rel, "chars": len(content), "overwrite": overwrite},
        summary=f"wrote {rel} ({len(content)} chars)",
    )


def _tool_delete_file(root: Path, args: dict[str, Any]) -> ToolResult:
    raw_path = _path_arg(args)
    path = _safe_repo_path(root, raw_path)
    if path is None:
        return ToolResult(tool="delete_file", ok=False, error="path must stay inside repo")
    rel_path = path.relative_to(root)
    if any(part in DEFAULT_EXCLUDED_DIRS for part in rel_path.parts):
        return ToolResult(tool="delete_file", ok=False, error="path uses an excluded directory")
    if not path.is_file():
        return ToolResult(tool="delete_file", ok=False, error="path is not a file")
    if path.suffix not in TEXT_EXTENSIONS:
        return ToolResult(tool="delete_file", ok=False, error="path is not a supported text file")
    path.unlink()
    rel = rel_path.as_posix()
    return ToolResult(
        tool="delete_file",
        ok=True,
        data={"path": rel},
        summary=f"deleted {rel}",
    )


def _positive_int(value: Any, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return min(parsed, maximum)


def _path_arg(args: dict[str, Any]) -> str:
    for key in ("path", "file_path", "file", "filename"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _safe_repo_path(root: Path, raw_path: str) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _format_live_tool_event(event: dict[str, Any]) -> str:
    tool = event.get("tool", "?")
    ok = event.get("ok")
    status = "ok" if ok else "ERR"
    args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
    target = _tool_target(args)
    target_text = f" {target}" if target else ""
    detail = event.get("summary") if ok else event.get("error")
    detail_text = f" {_short_text(str(detail), 120)}" if detail else ""
    return f"[tool] {tool}{target_text} {status}{detail_text} ({event.get('duration_s', '?')}s)"


def _tool_target(args: dict[str, Any]) -> str:
    for key in ("path", "target", "query"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return f"{key}={_short_text(value, 80)}"
    paths = args.get("paths")
    if isinstance(paths, list):
        return f"paths={len(paths)}"
    return ""


def _short_text(text: str, max_chars: int) -> str:
    single_line = " ".join(text.split())
    if len(single_line) <= max_chars:
        return single_line
    return single_line[: max_chars - 3].rstrip() + "..."


def _iter_text_files(root: Path):
    for path in sorted(root.rglob("*")):
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in DEFAULT_EXCLUDED_DIRS for part in rel_parts[:-1]):
            continue
        if path.is_file() and path.suffix in TEXT_EXTENSIONS:
            yield path


def _matches_simple_glob(path: str, glob: str) -> bool:
    if glob.startswith("*."):
        return path.endswith(glob[1:])
    if glob.endswith("/"):
        return path.startswith(glob)
    return glob in path


def _tail(text: str, max_chars: int = 12_000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]

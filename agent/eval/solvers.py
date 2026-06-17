"""Solvers pour l'éval : adaptent un modèle de chat en Solver (prompt, workdir -> fichiers).

- `make_oneshot_solver(chat)` : la baseline **BRUTE**. Un seul appel modèle, zéro outil,
  zéro boucle, zéro optimisation de contexte. Le modèle doit émettre ses fichiers dans des
  blocs ```lang path=<chemin>```. On parse et on écrit. C'est le **plancher** : la couche
  agentic devra faire mieux que ça, et l'écart se mesure sur le jeu d'éval.
- `engine_chat(model_path, ...)` : adaptateur `ChatFn` au-dessus du runtime llm_runtime
  (llama.cpp). Importé paresseusement (llama_cpp est lourd et optionnel).

Tous les fichiers sont écrits **dans le workdir fourni** (sous test/), jamais ailleurs.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

Message = dict[str, str]
ChatFn = Callable[[list[Message]], str]
Solver = Callable[[str, Path], None]

_FILE_EXT = r"(?:py|js|mjs|cjs|ts|html|htm|css|json|txt|md|csv|sh)"
_BLOCK = re.compile(r"```([^\n]*)\n(.*?)```", re.S)
_PATH_IN_HEADER = re.compile(rf"(?:path|file|filename|name)\s*=\s*[\"']?([\w./-]+\.{_FILE_EXT})", re.I)
_BARE_PATH = re.compile(rf"\b([\w./-]+\.{_FILE_EXT})\b")
_BODY_MARKER = re.compile(rf"^\s*(?://|#|<!--)\s*(?:path|file)\s*:\s*([\w./-]+\.{_FILE_EXT})", re.I)

_INSTRUCTION = (
    "Tu produis le(s) fichier(s) demandés, complets et prêts à l'emploi.\n"
    "Pour CHAQUE fichier, utilise un bloc de code dont l'en-tête précise le chemin, "
    "exactement ainsi :\n"
    "```python path=solution.py\n...contenu intégral du fichier...\n```\n"
    "Donne le contenu COMPLET de chaque fichier (pas d'extrait, pas de '...'). "
    "N'écris aucune explication hors des blocs."
)


def parse_file_blocks(text: str) -> dict[str, str]:
    """Extrait {chemin: contenu} des blocs de code d'une réponse modèle."""
    files: dict[str, str] = {}
    for header, body in _BLOCK.findall(text):
        path = _path_from_header(header) or _path_from_body(body)
        if path:
            files[path] = _strip_leading_marker(body)
    return files


def _path_from_header(header: str) -> str | None:
    match = _PATH_IN_HEADER.search(header) or _BARE_PATH.search(header)
    return match.group(1) if match else None


def _path_from_body(body: str) -> str | None:
    first = body.lstrip().splitlines()[0] if body.strip() else ""
    match = _BODY_MARKER.match(first)
    return match.group(1) if match else None


def _strip_leading_marker(body: str) -> str:
    lines = body.splitlines()
    if lines and _BODY_MARKER.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip("\n") + "\n"


def _guess_filename(prompt: str) -> str | None:
    """Nom de fichier cible dans le prompt, en évitant les fichiers de test
    (on ne route jamais un livrable vers un test_*.py)."""
    names = _BARE_PATH.findall(prompt)
    non_test = [n for n in names if not Path(n).name.startswith("test_")]
    chosen = non_test or names
    return chosen[0] if chosen else None


def write_files(workdir: Path, files: dict[str, str]) -> list[str]:
    """Écrit les fichiers dans le workdir. Ignore tout chemin qui s'en échapperait."""
    workdir = Path(workdir).resolve()
    written: list[str] = []
    for raw_path, content in files.items():
        target = (workdir / raw_path).resolve()
        if target != workdir and workdir not in target.parents:
            continue  # tentative d'évasion hors workdir : ignorée
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(str(target.relative_to(workdir)))
    return written


def make_oneshot_solver(chat: ChatFn) -> Solver:
    """Baseline brute : un appel modèle, on parse les fichiers émis, on les écrit."""

    def _solve(prompt: str, workdir: Path) -> None:
        reply = chat([{"role": "system", "content": _INSTRUCTION}, {"role": "user", "content": prompt}])
        _apply_reply(reply, prompt, Path(workdir))

    return _solve


def make_agent_solver(chat: ChatFn, max_rounds: int = 3) -> Solver:
    """Couche agentic GÉNÉRIQUE : émet les fichiers, les **vérifie en les exécutant**
    (compile, tests, smoke-run), et **renvoie l'erreur concrète au modèle** pour qu'il
    corrige — boucle bornée. Aucune logique propre à une tâche ni à un domaine.

    La vérification est la sienne, indépendante des checkers de l'éval (ne pas tricher).
    """

    def _solve(prompt: str, workdir: Path) -> None:
        workdir = Path(workdir)
        messages: list[Message] = [
            {"role": "system", "content": _INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        reply = chat(messages)
        _apply_reply(reply, prompt, workdir)

        for _ in range(max_rounds - 1):
            issues = _verify_workdir(workdir)
            if not issues:
                return
            messages.extend(
                [
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": (
                            "Des problèmes subsistent quand j'exécute ton rendu. Corrige le "
                            "CODE (pas les fichiers de test) et redonne les fichiers COMPLETS "
                            "dans le même format.\n\n"
                            f"Problèmes détectés :\n{issues}\n\n"
                            f"Fichiers actuels :\n{_render_current_files(workdir)}"
                        ),
                    },
                ]
            )
            reply = chat(messages)
            _apply_reply(reply, prompt, workdir)

    return _solve


def _apply_reply(reply: str, prompt: str, workdir: Path) -> None:
    files = parse_file_blocks(reply)
    if not files:
        # Repli pour modèle faible : blocs sans `path=`. On route vers le fichier que le
        # prompt demande d'écrire (jamais un test_*.py : un livrable n'écrase pas un test).
        blocks = _BLOCK.findall(reply)
        target = _guess_filename(prompt)
        if blocks and target:
            files = {target: _strip_leading_marker(blocks[0][1])}
    write_files(workdir, files)


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _uses_cli_args(src: str) -> bool:
    return "argv" in src or "argparse" in src or "input(" in src


def _verify_workdir(workdir: Path) -> str:
    """Vérifs génériques et structurelles. Renvoie un texte d'erreurs, ou "" si propre."""
    workdir = Path(workdir)
    issues: list[str] = []
    py_files = [p for p in sorted(workdir.rglob("*.py")) if "__pycache__" not in p.parts]

    for path in py_files:
        try:
            ast.parse(_src(path))
        except SyntaxError as exc:
            issues.append(f"{path.relative_to(workdir)}: SyntaxError ligne {exc.lineno}: {exc.msg}")
    if issues:
        return "\n".join(issues)

    if list(workdir.glob("test_*.py")):
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"], cwd=workdir, capture_output=True, text=True
        )
        if proc.returncode != 0:
            issues.append("pytest échoue :\n" + proc.stdout.strip()[-600:])

    mains = [
        p for p in py_files
        if "__main__" in _src(p) and not p.name.startswith("test_") and not _uses_cli_args(_src(p))
    ]
    if len(mains) == 1 and not issues:
        proc = subprocess.run([sys.executable, mains[0].name], cwd=workdir, capture_output=True, text=True)
        if proc.returncode != 0:
            issues.append(f"{mains[0].name} plante :\n" + proc.stderr.strip()[-400:])
    return "\n".join(issues)


def _render_current_files(workdir: Path, max_chars: int = 4000) -> str:
    workdir = Path(workdir)
    exts = {".py", ".js", ".mjs", ".html", ".htm", ".css", ".json", ".txt", ".csv"}
    parts: list[str] = []
    for path in sorted(workdir.rglob("*")):
        if path.is_file() and path.suffix in exts and "__pycache__" not in path.parts:
            body = _src(path)
            if len(body) > max_chars:
                body = body[:max_chars] + "\n[...tronqué...]"
            parts.append(f"```text path={path.relative_to(workdir).as_posix()}\n{body}\n```")
    return "\n\n".join(parts)


def engine_chat(
    model_path: str | Path,
    *,
    n_ctx: int = 8192,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    **engine_kwargs,
) -> ChatFn:
    """Adaptateur ChatFn sur le runtime. `chat.close()` libère le modèle."""
    from llm_runtime.engine import Engine

    engine = Engine(model_path, n_ctx=n_ctx, **engine_kwargs)

    def chat(messages: list[Message]) -> str:
        resp = engine.chat(messages, max_tokens=max_tokens, temperature=temperature)
        return resp["choices"][0]["message"]["content"]

    chat.close = engine.close  # type: ignore[attr-defined]
    return chat

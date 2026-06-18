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
        workdir = Path(workdir)
        reply = chat([
            {"role": "system", "content": _INSTRUCTION},
            {"role": "user", "content": _initial_user_message(prompt, workdir)},
        ])
        _apply_reply(reply, prompt, workdir)

    return _solve


def make_agent_solver(
    chat: ChatFn,
    max_rounds: int = 3,
    self_test: bool = False,
    trace: list | None = None,
) -> Solver:
    """Couche agentic GÉNÉRIQUE : émet les fichiers, les **vérifie en les exécutant**
    (compile, tests, smoke-run, navigateur headless), et **renvoie l'erreur concrète au
    modèle** pour qu'il corrige — boucle bornée. Aucune logique propre à une tâche.

    Endurance/persévérance :
    - `max_rounds` = budget de tours (1 émission + max_rounds-1 corrections).
    - **anti-blocage** : si la même erreur revient à l'identique d'un tour à l'autre, on
      arrête (persévérer aveuglément ne sert à rien — il manque un signal neuf).
    - `trace` (liste optionnelle) : on y enregistre l'état de chaque tour
      (`{round, ok, issues, stuck?, exhausted?}`) pour MESURER la trajectoire.

    `self_test` (OFF, mesuré régressif). La vérif est indépendante des checkers de l'éval.
    """

    def _solve(prompt: str, workdir: Path) -> None:
        workdir = Path(workdir)
        messages: list[Message] = [
            {"role": "system", "content": _INSTRUCTION},
            {"role": "user", "content": _initial_user_message(prompt, workdir)},
        ]
        reply = chat(messages)
        _apply_reply(reply, prompt, workdir)
        state: dict[str, str] = {}
        last_issues: str | None = None

        try:
            for round_no in range(1, max_rounds):
                issues = _verify_workdir(workdir)
                if not issues and self_test:
                    issues = _self_test(chat, workdir, state)
                if trace is not None:
                    trace.append({"round": round_no, "ok": not issues, "issues": issues})
                if not issues:
                    return  # convergé
                if issues == last_issues:
                    if trace is not None:
                        trace[-1]["stuck"] = True
                    return  # blocage : même erreur, pas de progrès -> inutile de continuer
                last_issues = issues
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
            if trace is not None:  # budget épuisé : état final
                final = _verify_workdir(workdir)
                trace.append({"round": max_rounds, "ok": not final, "issues": final, "exhausted": True})
        finally:
            _cleanup_selfcheck(workdir)

    return _solve


_SELFTEST_INSTRUCTION = (
    "Écris UN fichier selfcheck_main.py qui éprouve le comportement du code ci-dessous avec "
    "des assertions : importe le code, couvre quelques cas normaux ET surtout des cas limites "
    "représentatifs où il pourrait se tromper. Juste des `assert` + un bloc exécutable "
    "(`if __name__ == \"__main__\"`), pas de framework. Il doit lever AssertionError si le code "
    "est faux. Ne teste pas l'évident."
)


def _self_test(chat: ChatFn, workdir: Path, state: dict[str, str]) -> str:
    """Le modèle écrit ses tests-sanité (une seule fois), on les exécute. Renvoie un texte
    d'erreur seulement si SES tests échouent par AssertionError (= son code est faux selon
    lui-même). Un selfcheck cassé (import/syntaxe) est inconcluant : ignoré."""
    name = "selfcheck_main.py"
    if "generated" not in state:
        state["generated"] = "1"
        reply = chat(
            [
                {"role": "system", "content": _INSTRUCTION},
                {"role": "user", "content": _SELFTEST_INSTRUCTION + "\n\nCode actuel :\n" + _render_current_files(workdir)},
            ]
        )
        content = parse_file_blocks(reply).get(name) or _first_python_block(reply)
        if content:
            (workdir / name).write_text(content, encoding="utf-8")
            state["selfcheck"] = name
    if "selfcheck" not in state:
        return ""
    proc = subprocess.run([sys.executable, state["selfcheck"]], cwd=workdir, capture_output=True, text=True)
    if proc.returncode == 0:
        return ""
    if "AssertionError" in proc.stderr:
        return "Tes propres tests de sanité échouent (corrige le CODE, pas les tests) :\n" + proc.stderr.strip()[-500:]
    return ""


def _first_python_block(reply: str) -> str | None:
    for header, body in _BLOCK.findall(reply):
        if "python" in header.lower() or not header.strip():
            return _strip_leading_marker(body)
    return None


def _cleanup_selfcheck(workdir: Path) -> None:
    for path in Path(workdir).glob("selfcheck_*.py"):
        path.unlink(missing_ok=True)


def _initial_user_message(prompt: str, workdir: Path) -> str:
    """Prompt + contexte du dossier de travail (fichiers déjà présents). Sans ça, on
    demande au modèle de modifier des fichiers qu'il ne voit pas — baseline injuste.
    Version BRUTE : dump TOUT le dossier (sature un petit modèle si le repo est gros)."""
    existing = _render_current_files(Path(workdir))
    if existing.strip():
        return prompt + "\n\nFichiers déjà présents dans le dossier de travail :\n" + existing
    return prompt


def make_context_solver(chat: ChatFn, max_files: int = 4) -> Solver:
    """Égaliseur par le CONTEXTE : au lieu de dumper tout le dossier, on SÉLECTIONNE les
    fichiers pertinents (nommés dans le prompt + clôture de leurs imports) et on ne donne
    que ceux-là. Sur un gros dossier, ça évite de noyer un petit modèle. Générique."""

    def _solve(prompt: str, workdir: Path) -> None:
        workdir = Path(workdir)
        selected = _select_context_files(prompt, workdir, max_files)
        message = prompt
        if selected:
            message += "\n\nFichiers pertinents :\n" + _render_files(workdir, selected)
        reply = chat([{"role": "system", "content": _INSTRUCTION}, {"role": "user", "content": message}])
        _apply_reply(reply, prompt, workdir)

    return _solve


def _select_context_files(prompt: str, workdir: Path, max_files: int) -> list[str]:
    workdir = Path(workdir)
    all_files = {
        p.relative_to(workdir).as_posix(): p
        for p in sorted(workdir.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("selfcheck_")
    }
    selected: list[str] = [rel for rel in all_files if Path(rel).name in prompt]
    # Clôture des imports : un fichier sélectionné en tire d'autres réellement utiles.
    frontier = list(selected)
    while frontier and len(selected) < max_files * 3:
        current = workdir / frontier.pop()
        if current.suffix != ".py":
            continue
        try:
            tree = ast.parse(current.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module.split(".")[0]]
            for module in modules:
                candidate = f"{module}.py"
                if candidate in all_files and candidate not in selected:
                    selected.append(candidate)
                    frontier.append(candidate)
    return selected[:max_files]


def _render_files(workdir: Path, paths: list[str], max_chars: int = 4000) -> str:
    workdir = Path(workdir)
    parts: list[str] = []
    for rel in paths:
        path = workdir / rel
        if path.is_file():
            body = path.read_text(encoding="utf-8", errors="replace")
            if len(body) > max_chars:
                body = body[:max_chars] + "\n[...tronqué...]"
            parts.append(f"```text path={rel}\n{body}\n```")
    return "\n\n".join(parts)


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

    web = _verify_web(workdir)
    if web:
        issues.append(web)
    return "\n".join(issues)


def _verify_web(workdir: Path) -> str:
    """Vérif WEB via navigateur headless (Playwright) : charge chaque .html, clique chaque
    bouton, et remonte les erreurs JS (pageerror / console.error) — câble les crashs du
    type « mauvais id », « null.addEventListener », handler qui jette. Pendant web du check
    Python. Dépendance OPTIONNELLE : si Playwright absent, on saute proprement."""
    workdir = Path(workdir).resolve()  # as_uri() exige un chemin absolu
    html_files = [p for p in sorted(workdir.rglob("*.html")) if "__pycache__" not in p.parts]
    if not html_files:
        return ""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return ""

    issues: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for html in html_files:
                errors: list[str] = []
                page = browser.new_page()
                page.on("pageerror", lambda exc, acc=errors: acc.append(f"pageerror: {exc}"))
                page.on(
                    "console",
                    lambda msg, acc=errors: acc.append(f"console.error: {msg.text}") if msg.type == "error" else None,
                )
                try:
                    page.goto(html.as_uri(), wait_until="load", timeout=5000)
                    for button in page.query_selector_all("button"):
                        try:
                            button.click(timeout=500)
                        except Exception:  # noqa: BLE001 - un bouton non cliquable n'est pas l'erreur cherchée
                            pass
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"chargement: {exc}")
                page.close()
                if errors:
                    unique = " | ".join(dict.fromkeys(errors))
                    issues.append(f"{html.relative_to(workdir)} (navigateur) : {unique[:400]}")
        finally:
            browser.close()
    return "\n".join(issues)


def _render_current_files(workdir: Path, max_chars: int = 4000) -> str:
    workdir = Path(workdir)
    exts = {".py", ".js", ".mjs", ".html", ".htm", ".css", ".json", ".txt", ".csv"}
    parts: list[str] = []
    for path in sorted(workdir.rglob("*")):
        if path.name.startswith("selfcheck_"):
            continue  # scaffolding interne : on ne le montre pas au modèle qui corrige le code
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

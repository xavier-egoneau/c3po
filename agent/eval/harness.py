"""Harnais d'éval : mesure l'écart petit-modèle vs gros-modèle sur des tâches variées.

Principe directeur (anti-overfit) :
- une *tâche* = un prompt + un checker **exécutable et déterministe** ;
- un *solver* (ce qu'on teste : gros modèle baseline, petit modèle brut, ou petit
  modèle + couche agentic) reçoit le prompt et écrit des artefacts dans un workdir ;
- le checker vérifie les artefacts en les **exécutant** (import, subprocess, parse),
  jamais via une checklist de domaine ni un juge LLM.

Métrique : par tâche, score = fraction de checks passés (0..1). On compare les
solvers ; "gap closed" = (small_agent - small_raw) / (big - small_raw) quand les
trois labels sont présents.

Sécurité : les checkers exécutent du code produit par un modèle. À ne lancer que
localement, sur des tâches de confiance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
REPO_ROOT = Path(__file__).resolve().parents[2]
# Bac à sable unique : TOUTE exécution de test/éval se fait ici, jamais sur le repo.
SANDBOX_ROOT = REPO_ROOT / "test"

# Un solver reçoit (prompt, workdir) et écrit ses artefacts dans workdir.
Solver = Callable[[str, Path], None]
# Un checker reçoit le workdir et renvoie une liste de (nom, passé, détail).
Checker = Callable[[Path], "list[tuple[str, bool, str]]"]


@dataclass
class Task:
    name: str
    prompt: str
    dir: Path
    requires: list[str] = field(default_factory=list)
    check: Checker | None = None

    @property
    def fixture_dir(self) -> Path:
        return self.dir / "fixture"

    @property
    def solution_ref_dir(self) -> Path:
        return self.dir / "solution_ref"


@dataclass
class Outcome:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class TaskResult:
    task: str
    score: float
    outcomes: list[Outcome]
    duration_s: float
    error: str = ""


@dataclass
class EvalReport:
    label: str
    results: list[TaskResult]

    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.results) / len(self.results) if self.results else 0.0


def load_tasks(tasks_dir: Path = TASKS_DIR, only: list[str] | None = None) -> list[Task]:
    tasks: list[Task] = []
    for task_json in sorted(tasks_dir.glob("*/task.json")):
        meta = json.loads(task_json.read_text(encoding="utf-8"))
        name = meta["name"]
        if only and name not in only:
            continue
        tasks.append(
            Task(
                name=name,
                prompt=meta["prompt"],
                dir=task_json.parent,
                requires=list(meta.get("requires", [])),
                check=_load_checker(task_json.parent / "check.py"),
            )
        )
    return tasks


def _load_checker(path: Path) -> Checker:
    mod_name = f"_eval_check_{path.parent.name}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"checker introuvable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checker = getattr(module, "check", None)
    if not callable(checker):
        raise AttributeError(f"{path} doit exposer une fonction check(workdir)")
    return checker


def oracle_solver(task: Task) -> Solver:
    """Solver de référence : copie solution_ref/ dans le workdir.

    Sert à valider les checkers eux-mêmes (avec l'oracle, tout doit être vert).
    """

    def _solve(_prompt: str, workdir: Path) -> None:
        ref = task.solution_ref_dir
        if not ref.is_dir():
            raise FileNotFoundError(f"pas de solution_ref pour {task.name}")
        _copy_tree(ref, workdir)

    return _solve


def _assert_sandboxed(path: Path) -> None:
    """Garde-fou : refuse tout chemin hors du bac à sable test/. Empêche un
    solver ou un checker de toucher au repo c3po."""
    resolved = path.resolve()
    sandbox = SANDBOX_ROOT.resolve()
    if resolved != sandbox and sandbox not in resolved.parents:
        raise ValueError(f"refus: chemin hors du bac à sable test/ : {resolved}")


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]", "_", label) or "run"


def run_task(task: Task, solver: Solver, work_root: Path) -> TaskResult:
    workdir = work_root / task.name
    _assert_sandboxed(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if task.fixture_dir.is_dir():
        _copy_tree(task.fixture_dir, workdir)

    start = time.time()
    try:
        solver(task.prompt, workdir)
    except Exception as exc:  # noqa: BLE001 - un solver qui crash = tâche ratée, pas un crash global
        return TaskResult(task.name, 0.0, [], round(time.time() - start, 3), f"solver: {exc!r}")

    try:
        raw = task.check(workdir) if task.check else []
        outcomes = [Outcome(str(n), bool(p), str(d)) for n, p, d in raw]
    except Exception as exc:  # noqa: BLE001
        return TaskResult(task.name, 0.0, [], round(time.time() - start, 3), f"checker: {exc!r}")

    score = sum(1 for o in outcomes if o.passed) / len(outcomes) if outcomes else 0.0
    return TaskResult(task.name, score, outcomes, round(time.time() - start, 3))


def run_eval(
    solver_for: Callable[[Task], Solver],
    label: str,
    tasks: list[Task],
    work_root: Path | None = None,
) -> EvalReport:
    root = Path(work_root) if work_root is not None else (SANDBOX_ROOT / _safe_label(label))
    _assert_sandboxed(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    results = [run_task(task, solver_for(task), root) for task in tasks]
    return EvalReport(label=label, results=results)


def _copy_tree(src: Path, dst: Path) -> None:
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def render_report(report: EvalReport) -> str:
    lines = [f"# Éval : {report.label}", ""]
    for r in report.results:
        head = f"- {r.task}: {r.score:.0%} ({r.duration_s}s)"
        if r.error:
            head += f"  ERREUR {r.error}"
        lines.append(head)
        for o in r.outcomes:
            mark = "ok " if o.passed else "FAIL"
            detail = f" — {o.detail}" if o.detail else ""
            lines.append(f"    [{mark}] {o.name}{detail}")
    lines.append("")
    lines.append(f"Score moyen : {report.mean_score:.1%}")
    return "\n".join(lines)


def render_comparison(reports: dict[str, EvalReport]) -> str:
    """Table comparative + gap closed si labels big/small/small+agent présents."""
    labels = list(reports)
    task_names = [r.task for r in next(iter(reports.values())).results]
    width = max((len(n) for n in task_names), default=4) + 2
    header = "tâche".ljust(width) + "".join(f"{lab:>14}" for lab in labels)
    rows = [header, "-" * len(header)]
    by_task = {lab: {r.task: r.score for r in rep.results} for lab, rep in reports.items()}
    for name in task_names:
        row = name.ljust(width) + "".join(f"{by_task[lab].get(name, 0):>13.0%} " for lab in labels)
        rows.append(row)
    rows.append("-" * len(header))
    means = "moyenne".ljust(width) + "".join(f"{reports[lab].mean_score:>13.0%} " for lab in labels)
    rows.append(means)

    big = _find(reports, ("big", "gros", "baseline"))
    raw = _find(reports, ("small", "small_raw", "petit"))
    agent = _find(reports, ("small_agent", "small+agent", "agentic"))
    if big and raw and agent:
        denom = reports[big].mean_score - reports[raw].mean_score
        gain = reports[agent].mean_score - reports[raw].mean_score
        closed = (gain / denom) if denom > 1e-9 else float("nan")
        rows += ["", f"gap closed = (agent - raw)/(big - raw) = {closed:.0%}"]
    return "\n".join(rows)


def _find(reports: dict[str, EvalReport], candidates: tuple[str, ...]) -> str | None:
    for lab in reports:
        if lab.lower() in candidates:
            return lab
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harnais d'éval c3po")
    parser.add_argument("--oracle", action="store_true", help="lance le solver de référence (valide les checkers)")
    parser.add_argument("--list", action="store_true", help="liste les tâches")
    parser.add_argument("--only", nargs="*", help="restreint à ces tâches")
    args = parser.parse_args(argv)

    tasks = load_tasks(only=args.only)
    if args.list:
        for t in tasks:
            req = f" requires={t.requires}" if t.requires else ""
            print(f"{t.name}{req}: {t.prompt[:70]}")
        return 0

    if args.oracle:
        report = run_eval(lambda task: oracle_solver(task), "oracle", tasks)
        print(render_report(report))
        return 0 if report.mean_score > 0.999 else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

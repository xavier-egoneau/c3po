"""Harnais d'éval c3po : mesure l'écart petit-modèle vs gros-modèle.

Voir agent/eval/README.md.
"""

from agent.eval.harness import (
    EvalReport,
    Task,
    TaskResult,
    load_tasks,
    oracle_solver,
    render_comparison,
    render_report,
    run_eval,
    run_task,
)

__all__ = [
    "EvalReport",
    "Task",
    "TaskResult",
    "load_tasks",
    "oracle_solver",
    "render_comparison",
    "render_report",
    "run_eval",
    "run_task",
]

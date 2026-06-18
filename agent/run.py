"""Lance un solver sur une tâche RÉELLE (hors éval), pour voir ce qu'il produit.

Exemples :
  python -m agent.run --model ~/.c3po/models/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf \
      "Écris fibonacci.py avec fib(n) et un petit CLI"

  # sur un vrai dossier (contexte lu + fichiers écrits dedans) :
  python -m agent.run --model <gguf> --solver context --dir ./mon_projet \
      "Corrige le bug dans calc.py"

Solvers : context (défaut, sélection de contexte = l'égaliseur), oneshot (dump tout),
loop (add-on expérimental). Par défaut on écrit dans test/adhoc/ (scratch).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from agent.eval.solvers import (
    _INSTRUCTION,
    _apply_reply,
    _render_current_files,
    engine_chat,
    make_agent_solver,
    make_context_solver,
    make_oneshot_solver,
)

SOLVERS = {
    "context": make_context_solver,
    "oneshot": make_oneshot_solver,
    "loop": make_agent_solver,
}
SCRATCH = Path(__file__).resolve().parent.parent / "test" / "adhoc"


def _snapshot(root: Path) -> dict[str, float]:
    return {
        p.relative_to(root).as_posix(): p.stat().st_mtime
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }


def _report_changes(workdir: Path, before: dict[str, float], show_body: bool = True) -> None:
    after = _snapshot(workdir)
    changed = sorted(rel for rel, mtime in after.items() if before.get(rel) != mtime)
    print(f"=== {len(changed)} fichier(s) écrit(s) dans {workdir} ===")
    for rel in changed:
        body = (workdir / rel).read_text(encoding="utf-8", errors="replace")
        if show_body:
            print(f"\n----- {rel} ({len(body)} car) -----\n{body}")
        else:
            print(f"  {rel} ({len(body)} car)")
    if not changed:
        print("(aucun fichier produit — le modèle n'a peut-être pas suivi le format de sortie)")


def _agent_turn(chat, workdir: Path, task: str, feedback: str) -> None:
    """Un tour : le modèle voit la tâche, les fichiers ACTUELS (l'état vit dans les
    fichiers, pas dans un historique qui déborde) et le retour, puis révise."""
    files_view = _render_current_files(workdir)
    parts = [f"Tâche : {task}"]
    if feedback and feedback != task:
        parts.append(f"Retour utilisateur à corriger : {feedback}")
    if files_view.strip():
        parts.append("Fichiers actuels (révise-les, ne repars pas de zéro) :\n" + files_view)
    parts.append("Produis ou révise les fichiers nécessaires, COMPLETS, au format ```lang path=...```.")
    reply = chat([{"role": "system", "content": _INSTRUCTION}, {"role": "user", "content": "\n\n".join(parts)}])
    _apply_reply(reply, task, workdir)


def run_interactive(chat, workdir: Path, task: str) -> None:
    """Boucle d'itération : tu vois le rendu, tu donnes un retour, le modèle révise.
    Entrée vide ou 'q' pour finir. C'est TOI le vérificateur du visuel/interactif."""
    feedback = task
    while True:
        before = _snapshot(workdir)
        _agent_turn(chat, workdir, task, feedback)
        _report_changes(workdir, before, show_body=False)
        print(f"  (fichiers dans {workdir})")
        try:
            feedback = input("\nRetour à corriger (entrée vide ou 'q' = fini) > ").strip()
        except EOFError:
            break
        if not feedback or feedback.lower() in {"q", "quit", "exit"}:
            break


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lance un solver sur une tâche réelle.")
    ap.add_argument("task", help="la tâche en langage naturel")
    ap.add_argument("--model", required=True, help="chemin GGUF")
    ap.add_argument("--solver", choices=list(SOLVERS), default="context")
    ap.add_argument("--dir", help="dossier de travail (contexte lu + fichiers écrits). défaut : test/adhoc")
    ap.add_argument("--chat", action="store_true", help="mode itératif : tu donnes des retours, le modèle révise (idéal pour HTML/JS visuel)")
    ap.add_argument("--rounds", type=int, default=3, help="budget de tours pour --solver loop (endurance)")
    args = ap.parse_args(argv)

    workdir = Path(args.dir).resolve() if args.dir else SCRATCH
    workdir.mkdir(parents=True, exist_ok=True)
    if args.dir:
        print(f"⚠  écriture DIRECTE dans {workdir} (vrai dossier, pas un scratch).")

    print(f"Chargement {Path(args.model).name} …")
    t0 = time.time()
    chat = engine_chat(args.model)
    mode = "itératif" if args.chat else f"solver={args.solver}"
    print(f"chargé en {time.time() - t0:.1f}s — {mode}\n")
    try:
        if args.chat:
            run_interactive(chat, workdir, args.task)
        else:
            before = _snapshot(workdir)
            trace: list = []
            solver = make_agent_solver(chat, max_rounds=args.rounds, trace=trace) if args.solver == "loop" else SOLVERS[args.solver](chat)
            solver(args.task, workdir)
            _report_changes(workdir, before)
            for step in trace:
                tag = " BLOQUÉ" if step.get("stuck") else (" (budget épuisé)" if step.get("exhausted") else "")
                print(f"  tour {step['round']}: {'ok' if step['ok'] else 'à corriger'}{tag}"
                      + (f" — {step['issues'][:80]}" if step.get("issues") else ""))
    finally:
        close = getattr(chat, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

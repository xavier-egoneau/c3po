"""
CLI c3po — interface en ligne de commande pour llm-runtime.

Usage:
  c3po list
  c3po info
  c3po run <model>
  c3po serve [<model>] [--port 8000]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Commandes
# ---------------------------------------------------------------------------

def cmd_list(args):
    """Liste tous les modèles disponibles."""
    from .models import list_models
    from .hardware import detect_hardware

    profile = detect_hardware()
    models = list_models(local_dirs=["models"])

    if not models:
        print("Aucun modèle trouvé.")
        print("Ajoutez des fichiers .gguf dans ./models/ ou installez Ollama.")
        return

    print(f"{'NOM':<35} {'TAILLE':>8}  {'FIT':>4}  SOURCE    CHEMIN")
    print("─" * 100)
    for m in models:
        fits = "✓" if m.fits_in(profile.gpu_memory_gb) else "✗"
        name = m.name[:34]
        path_str = str(m.path)
        if len(path_str) > 45:
            path_str = "..." + path_str[-42:]
        print(f"{name:<35} {m.size_gb:>7.1f}G  {fits:>4}  {m.source:<8}  {path_str}")

    print()
    print(f"Hardware : {profile.device_name} — {profile.gpu_memory_gb:.1f} Go disponibles")

    from .instances import active_instances
    instances = active_instances()
    if instances:
        print()
        print("Instances actives :")
        for i in instances:
            print(f"  - PID {i['pid']:<8} {i['model']:<35} ~{i['size_gb']:.1f} Go")


def cmd_info(args):
    """Affiche le profil hardware et les paramètres qui seraient appliqués."""
    from .hardware import detect_hardware
    from .params import compute_params

    profile = detect_hardware()
    params = compute_params(profile)

    print("── Hardware ──────────────────────────────")
    print(profile)
    print()
    print("── Paramètres d'inférence calculés ──────")
    print(params)


def cmd_run(args):
    """Lance une session de chat interactive avec un modèle."""
    from .models import list_models
    from .engine import Engine

    model_path = _resolve_model(args.model)
    if model_path is None:
        return

    print(f"Chargement de {model_path.name}…")
    engine = Engine(model_path, n_ctx=args.ctx)
    print(engine)
    print()
    print("Session de chat (Ctrl+C ou 'exit' pour quitter)")
    print("─" * 50)

    history = []

    try:
        while True:
            try:
                user_input = input("Vous : ").strip()
            except EOFError:
                break

            if not user_input or user_input.lower() in ("exit", "quit", "q"):
                break

            history.append({"role": "user", "content": user_input})

            print("Assistant : ", end="", flush=True)
            full_response = ""

            for chunk in engine.chat(
                history,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    print(delta, end="", flush=True)
                    full_response += delta

            print()  # newline après la réponse
            history.append({"role": "assistant", "content": full_response})

    except KeyboardInterrupt:
        print("\nAu revoir.")


def cmd_batch(args):
    """Traitement batch parallèle sur fichiers ou prompts."""
    from .batch import (
        tasks_from_files, tasks_from_prompts,
        run_batch, save_results, optimal_jobs,
    )

    model_path = _resolve_model(args.model)
    if model_path is None:
        return

    # Construction des tâches
    tasks = []

    if args.input:
        import glob
        paths = []
        for pattern in args.input:
            paths.extend(Path(p) for p in glob.glob(pattern))
        if not paths:
            print(f"Aucun fichier trouvé pour : {args.input}")
            return
        template = args.prompt or "Voici un texte :\n{content}\n\nQue peux-tu me dire sur ce texte ?"
        tasks = tasks_from_files(paths, prompt_template=template)

    elif args.prompts:
        prompts_path = Path(args.prompts)
        if not prompts_path.exists():
            print(f"Fichier de prompts introuvable : {prompts_path}")
            return
        prompts = [l.strip() for l in prompts_path.read_text().splitlines() if l.strip()]
        tasks = tasks_from_prompts(prompts)

    else:
        print("Spécifiez --input <fichiers> ou --prompts <fichier>")
        return

    if not tasks:
        print("Aucune tâche à traiter.")
        return

    jobs = args.jobs or optimal_jobs(model_path, n_ctx=args.ctx)
    summary = run_batch(tasks, model_path, jobs=jobs, n_ctx=args.ctx)

    if args.output:
        save_results(summary, args.output)
    else:
        # Affichage compact si pas d'output
        print()
        for r in summary.results:
            label = Path(r.source).name if r.source else f"#{r.id}"
            print(f"── {label}")
            print(r.response.strip() if not r.error else f"[ERREUR] {r.error}")
            print()


def cmd_serve(args):
    """Lance le serveur HTTP compatible OpenAI."""
    import os
    import subprocess

    model_path = _resolve_model(args.model) if args.model else None

    env = os.environ.copy()
    if model_path:
        env["LLM_RUNTIME_MODEL"] = str(model_path)

    print(f"Démarrage du serveur sur http://localhost:{args.port}")
    if model_path:
        print(f"Modèle : {model_path.name}")
    else:
        print("Modèle : sélection automatique (best_model)")
    print("─" * 50)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "llm_runtime.server:app",
        "--host", "0.0.0.0",
        "--port", str(args.port),
    ]

    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nServeur arrêté.")


# ---------------------------------------------------------------------------
# Résolution du modèle (nom partiel, chemin, ou best_model)
# ---------------------------------------------------------------------------

def _resolve_model(query: str | None) -> Path | None:
    from .models import find_model, best_model
    from .hardware import detect_hardware

    if query is None:
        profile = detect_hardware()
        best = best_model(profile.gpu_memory_gb, local_dirs=["models"])
        if best is None:
            print("Aucun modèle disponible.")
            return None
        print(f"Modèle sélectionné automatiquement : {best.name}")
        return best.path

    try:
        return find_model(query, local_dirs=["models"]).path
    except ValueError as e:
        print(str(e))
        return None


# ---------------------------------------------------------------------------
# Parser CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="c3po",
        description="Orchestrateur llama.cpp avec routing hardware automatique",
    )
    sub = parser.add_subparsers(dest="command", metavar="<commande>")
    sub.required = True

    # list
    sub.add_parser("list", help="Liste les modèles disponibles")

    # info
    sub.add_parser("info", help="Affiche le profil hardware et les paramètres calculés")

    # run
    p_run = sub.add_parser("run", help="Lance une session de chat interactive")
    p_run.add_argument("model", nargs="?", default=None,
                       help="Nom ou chemin du modèle (auto si omis)")
    p_run.add_argument("--ctx", type=int, default=4096, help="Taille du contexte (défaut: 4096)")
    p_run.add_argument("--max-tokens", type=int, default=512, dest="max_tokens")
    p_run.add_argument("--temperature", type=float, default=0.7)

    # serve
    p_serve = sub.add_parser("serve", help="Lance le serveur HTTP compatible OpenAI")
    p_serve.add_argument("model", nargs="?", default=None,
                         help="Nom ou chemin du modèle (auto si omis)")
    p_serve.add_argument("--port", type=int, default=8000)

    # batch
    p_batch = sub.add_parser("batch", help="Traitement batch parallèle")
    p_batch.add_argument("model", nargs="?", default=None,
                         help="Nom ou chemin du modèle (auto si omis)")
    p_batch.add_argument("--input", nargs="+", metavar="FICHIER",
                         help="Fichiers à traiter (glob supporté, ex: docs/*.txt)")
    p_batch.add_argument("--prompts", metavar="FICHIER",
                         help="Fichier texte avec un prompt par ligne")
    p_batch.add_argument("--prompt", metavar="TEMPLATE",
                         help="Template de prompt pour --input (utilise {content} pour le texte du fichier)")
    p_batch.add_argument("--output", metavar="FICHIER",
                         help="Fichier JSON de sortie (ex: results.json)")
    p_batch.add_argument("--jobs", type=int, default=None,
                         help="Nombre de workers parallèles (auto si omis)")
    p_batch.add_argument("--ctx", type=int, default=2048,
                         help="Taille du contexte par worker (défaut: 2048)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "list":  cmd_list,
        "info":  cmd_info,
        "run":   cmd_run,
        "serve": cmd_serve,
        "batch": cmd_batch,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()

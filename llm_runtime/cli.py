"""
CLI c3po — interface en ligne de commande pour llm-runtime.

Usage:
  c3po list
  c3po info
  c3po search <query>
  c3po load <user/repo>[:QUANT]
  c3po run <model>
  c3po stats <model>
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
    models = list_models()

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
    engine = Engine(model_path, n_ctx=args.ctx, **_engine_overrides(args))
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
    summary = run_batch(tasks, model_path, jobs=jobs, n_ctx=args.ctx,
                        engine_overrides=_engine_overrides(args))

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


def cmd_load(args):
    """Télécharge un modèle GGUF depuis Hugging Face dans le dossier modèles."""
    from .download import load

    try:
        load(args.ref, quant=args.quant, dest_dir=args.dir)
    except ValueError as e:
        print(str(e))
    except Exception as e:
        print(f"Échec du téléchargement : {e}")


def cmd_search(args):
    """Cherche sur Hugging Face les modèles GGUF qui tiennent dans le hardware courant."""
    from .download import search_eligible
    from .hardware import detect_hardware

    profile = detect_hardware()
    print(f"Recherche « {args.query} » sur Hugging Face… ({profile.gpu_memory_gb:.1f} Go dispo)")
    try:
        results = search_eligible(args.query, profile.gpu_memory_gb, limit=args.limit)
    except Exception as e:
        print(f"Échec de la recherche : {e}")
        return

    if not args.all:
        results = [r for r in results if r.fits]

    if not results:
        print("Aucun modèle GGUF éligible trouvé. Essayez --all ou une autre requête.")
        return

    print(f"\n{'REPO':<55} {'QUANT':>10} {'TAILLE':>8}  {'FIT':>4}  {'DL':>8}")
    print("─" * 95)
    any_mm = False
    for r in results:
        fit = "✓" if r.fits else "✗"
        mark = " *" if r.multimodal else ""
        any_mm = any_mm or r.multimodal
        repo = (r.repo + mark)
        repo = repo if len(repo) <= 55 else "…" + repo[-54:]
        print(f"{repo:<55} {r.quant:>10} {r.size_gb:>7.1f}G  {fit:>4}  {r.downloads:>8}")
    if any_mm:
        print("\n* multimodal — c3po ne charge que la partie texte (pas de vision/audio).")
    print(f"\nInstaller : c3po load <repo>  (quant auto selon la VRAM)")


def cmd_stats(args):
    """Affiche les stats d'un modèle : métadonnées GGUF + benchmark sur ce hardware."""
    from .stats import collect_stats, format_stats

    model_path = _resolve_model(args.model)
    if model_path is None:
        return

    print(f"Chargement et benchmark de {model_path.name}… (quelques secondes)")
    stats = collect_stats(model_path, n_ctx=args.ctx, **_engine_overrides(args))

    if args.json:
        import json
        from dataclasses import asdict
        print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
    else:
        print()
        print(format_stats(stats))


def cmd_serve(args):
    """Lance le serveur HTTP compatible OpenAI."""
    import os
    import subprocess

    model_path = _resolve_model(args.model) if args.model else None

    env = os.environ.copy()
    if model_path:
        env["LLM_RUNTIME_MODEL"] = str(model_path)
    env["LLM_RUNTIME_CTX"] = str(args.ctx)
    if args.n_gpu_layers is not None:
        env["LLM_RUNTIME_N_GPU_LAYERS"] = str(args.n_gpu_layers)
    if args.threads is not None:
        env["LLM_RUNTIME_THREADS"] = str(args.threads)
    if args.flash_attn is not None:
        env["LLM_RUNTIME_FLASH_ATTN"] = "1" if args.flash_attn else "0"
    if args.kv_type is not None:
        env["LLM_RUNTIME_KV_TYPE"] = _KV_ALIASES[args.kv_type]
    if args.speculative:
        env["LLM_RUNTIME_SPECULATIVE"] = "1"

    print(f"Démarrage du serveur sur http://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print("⚠️  Écoute sur toutes les interfaces réseau, sans authentification.")
    if model_path:
        print(f"Modèle : {model_path.name}")
    else:
        print("Modèle : sélection automatique (best_model)")
    print("─" * 50)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "llm_runtime.server:app",
        "--host", args.host,
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
        best = best_model(profile.gpu_memory_gb)
        if best is None:
            print("Aucun modèle disponible.")
            return None
        print(f"Modèle sélectionné automatiquement : {best.name}")
        return best.path

    try:
        return find_model(query).path
    except ValueError as e:
        print(str(e))
        return None


# ---------------------------------------------------------------------------
# Parser CLI
# ---------------------------------------------------------------------------

def _add_engine_args(parser: argparse.ArgumentParser, ctx_default: int = 4096) -> None:
    """Leviers d'inférence communs (exposés, pas cachés). None = valeur auto-calculée."""
    parser.add_argument("--ctx", type=int, default=ctx_default,
                        help=f"Taille du contexte en tokens (défaut: {ctx_default}, plafonné au modèle)")
    parser.add_argument("--n-gpu-layers", type=int, default=None, dest="n_gpu_layers",
                        help="Couches envoyées sur GPU (-1 = toutes, 0 = CPU only ; auto si omis)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Threads CPU (auto si omis)")
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=None,
                        dest="flash_attn", help="Forcer/désactiver la flash attention (auto si omis)")
    parser.add_argument("--kv-type", choices=["f16", "q8", "q4"], default=None, dest="kv_type",
                        help="Précision du KV cache (auto si omis : F16, ou Q8 si besoin pour "
                             "tenir le contexte ; Q4 uniquement explicite)")
    parser.add_argument("--speculative", action="store_true",
                        help="Prompt-lookup decoding : accélère les sorties qui recopient "
                             "l'entrée (code, RAG, édition) ; à éviter sur du texte créatif")


# Alias CLI courts → noms ggml canoniques
_KV_ALIASES = {"f16": "f16", "q8": "q8_0", "q4": "q4_0"}


def _engine_overrides(args) -> dict:
    """Extrait les leviers explicites depuis les args parsés."""
    kv = getattr(args, "kv_type", None)
    return {
        "n_gpu_layers": getattr(args, "n_gpu_layers", None),
        "n_threads": getattr(args, "threads", None),
        "flash_attn": getattr(args, "flash_attn", None),
        "kv_type": _KV_ALIASES.get(kv) if kv else None,
        "speculative": getattr(args, "speculative", False),
    }


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
    _add_engine_args(p_run)
    p_run.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
                       help="Limite de tokens générés (auto si omis : jusqu'à la fin de la "
                            "réponse ou la limite de contexte)")
    p_run.add_argument("--temperature", type=float, default=0.7)

    # search
    p_search = sub.add_parser("search", help="Cherche sur HF les modèles GGUF qui tiennent dans ton hardware")
    p_search.add_argument("query", help="Terme de recherche (ex: qwen2.5, llama 3, mistral)")
    p_search.add_argument("--limit", type=int, default=20, help="Nombre de repos à inspecter (défaut: 20)")
    p_search.add_argument("--all", action="store_true", help="Afficher aussi les modèles qui ne tiennent pas")

    # load
    p_load = sub.add_parser("load", help="Télécharge un modèle GGUF depuis Hugging Face")
    p_load.add_argument("ref", help="Repo HF, ex: bartowski/Qwen2.5-7B-Instruct-GGUF[:Q4_K_M]")
    p_load.add_argument("--quant", default=None,
                        help="Quantization à télécharger (ex: Q5_K_M). Auto selon la VRAM si omis.")
    p_load.add_argument("--dir", default=None,
                        help="Dossier de destination (défaut: ~/.c3po/models ou $C3PO_MODELS_DIR)")

    # stats
    p_stats = sub.add_parser("stats", help="Métadonnées d'un modèle + benchmark sur ce hardware")
    p_stats.add_argument("model", nargs="?", default=None,
                         help="Nom ou chemin du modèle (auto si omis)")
    _add_engine_args(p_stats)
    p_stats.add_argument("--json", action="store_true", help="Sortie au format JSON")

    # serve
    p_serve = sub.add_parser("serve", help="Lance le serveur HTTP compatible OpenAI")
    p_serve.add_argument("model", nargs="?", default=None,
                         help="Nom ou chemin du modèle (auto si omis)")
    p_serve.add_argument("--host", default="127.0.0.1",
                         help="Interface d'écoute (défaut: 127.0.0.1 ; 0.0.0.0 pour exposer sur le réseau)")
    p_serve.add_argument("--port", type=int, default=8000)
    _add_engine_args(p_serve)

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
    _add_engine_args(p_batch, ctx_default=2048)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "list":   cmd_list,
        "info":   cmd_info,
        "search": cmd_search,
        "load":   cmd_load,
        "run":    cmd_run,
        "stats": cmd_stats,
        "serve": cmd_serve,
        "batch": cmd_batch,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()

"""
CLI c3po — interface en ligne de commande pour llm-runtime.

Usage:
  c3po list
  c3po info
  c3po doctor [<model>]
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
# Politique d'instance unique
# ---------------------------------------------------------------------------

def _terminate_other_instances_before_start() -> None:
    """Ferme les autres modèles c3po déjà chargés avant de démarrer celui-ci."""
    from .instances import terminate_other_instances

    stopped = terminate_other_instances()
    if stopped:
        print("Instances c3po existantes arrêtées :")
        for i in stopped:
            print(f"  - PID {i['pid']} : {i['model']} (~{i['size_gb']:.1f} Go)")
        print()


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


def cmd_doctor(args):
    """Diagnostique l'environnement c3po et éventuellement un modèle."""
    from .doctor import collect_doctor, format_doctor

    print(format_doctor(collect_doctor(args.model)))


def cmd_run(args):
    """Lance une session de chat interactive avec un modèle."""
    from .models import list_models
    from .engine import Engine

    model_path = _resolve_model(args.model)
    if model_path is None:
        return

    _terminate_other_instances_before_start()

    print(f"Chargement de {model_path.name}…")
    try:
        engine = Engine(model_path, n_ctx=args.ctx, **_engine_overrides(args))
    except RuntimeError as e:
        print(str(e))
        return
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
            history = _maybe_compact_history(engine, history)
            max_tokens = _safe_generation_max_tokens(engine, history, args.max_tokens)
            if max_tokens <= 0:
                print(
                    "[Message trop long pour la fenêtre de contexte après compaction ; "
                    "réduisez le prompt ou relancez avec --ctx plus grand.]"
                )
                history.pop()
                continue

            print("Assistant : ", end="", flush=True)
            full_response = ""

            try:
                for chunk in engine.chat(
                    history,
                    max_tokens=max_tokens,
                    temperature=args.temperature,
                    stream=True,
                ):
                    delta = chunk["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        full_response += delta
            except ValueError as e:
                print(f"\n[Erreur de génération : {e}]")
                print("[Le contexte a probablement atteint sa limite ; l'historique sera compacté au prochain tour.]")
                continue

            print()  # newline après la réponse
            history.append({"role": "assistant", "content": full_response})

    except KeyboardInterrupt:
        print("\nAu revoir.")


def cmd_batch(args):
    """Traitement batch mono-instance sur fichiers ou prompts."""
    from .batch import (
        tasks_from_files, tasks_from_prompts,
        run_batch, save_results, optimal_jobs,
    )

    model_path = _resolve_model(args.model)
    if model_path is None:
        return

    _terminate_other_instances_before_start()

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
                        engine_overrides=_engine_overrides(args),
                        max_tokens=args.max_tokens)

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

    _terminate_other_instances_before_start()

    if not args.json:
        print(f"Chargement et benchmark de {model_path.name}… (quelques secondes)")
    try:
        stats = collect_stats(model_path, n_ctx=args.ctx, **_engine_overrides(args))
    except RuntimeError as e:
        print(str(e))
        return

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

    _terminate_other_instances_before_start()

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
    if args.repeat_penalty is not None:
        env["LLM_RUNTIME_REPEAT_PENALTY"] = str(args.repeat_penalty)

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
# Compaction de conversation interactive
# ---------------------------------------------------------------------------

_COMPACTION_RATIO = 0.95
_COMPACTION_KEEP_MESSAGES = 4
_GENERATION_SAFETY_TOKENS = 64


def _maybe_compact_history(engine, history: list[dict]) -> list[dict]:
    """
    Compacte l'historique interactif quand il approche de la fenêtre de contexte.

    Le seuil est calculé sur le `n_ctx` réellement appliqué par l'Engine, donc après
    plafonnement éventuel par le modèle ou réduction mémoire.
    """
    limit = max(1, int(engine.params.n_ctx * _COMPACTION_RATIO))
    try:
        tokens = engine.count_messages_tokens(history)
    except Exception:
        return history

    if tokens < limit:
        return history

    compacted = _compact_history(engine, history, limit)
    if compacted is history:
        print(
            f"\n[Compaction impossible : le dernier message occupe déjà ~{tokens}/{engine.params.n_ctx} tokens.]"
        )
        return history

    after = engine.count_messages_tokens(compacted)
    print(
        f"\n[Historique compacté : ~{tokens} → ~{after} tokens "
        f"(seuil {limit}/{engine.params.n_ctx}).]"
    )
    return compacted


def _safe_generation_max_tokens(
    engine,
    history: list[dict],
    requested_max_tokens: int | None,
) -> int:
    """
    Borne la génération pour ne jamais atteindre la fin du contexte llama.cpp.

    `max_tokens=None` est pratique pour laisser le modèle finir naturellement, mais
    quand l'historique approche de `n_ctx`, llama-cpp-python peut échouer au bord du
    buffer au lieu de s'arrêter proprement. On transforme donc le None en budget sûr
    par tour, et on plafonne aussi un `--max-tokens` explicite trop ambitieux.
    """
    try:
        prompt_tokens = engine.count_messages_tokens(history)
    except Exception:
        return requested_max_tokens if requested_max_tokens is not None else max(
            1, engine.params.n_ctx // 2
        )

    available = engine.params.n_ctx - prompt_tokens - _GENERATION_SAFETY_TOKENS
    if available <= 0:
        return 0
    if requested_max_tokens is None or requested_max_tokens <= 0:
        return available
    return min(requested_max_tokens, available)


def _compact_history(engine, history: list[dict], token_limit: int) -> list[dict]:
    if len(history) <= 1:
        return history

    keep_count = min(_COMPACTION_KEEP_MESSAGES, max(1, len(history) - 1))
    old = history[:-keep_count]
    tail = history[-keep_count:]
    if not old:
        return history

    transcript = _render_transcript(old)
    max_summary_tokens = max(128, min(512, engine.params.n_ctx // 8))
    summary_prompt = [
        {
            "role": "system",
            "content": (
                "Tu compactes une conversation pour permettre de continuer longtemps. "
                "Réponds uniquement en Markdown, avec exactement ces rubriques :\n"
                "## Objectif courant\n"
                "## Décisions prises\n"
                "## Fichiers, modèles et commandes\n"
                "## Contraintes utilisateur\n"
                "## État des tâches\n"
                "## Prochains pas\n"
                "Conserve les faits utiles, les préférences, les erreurs rencontrées, "
                "les commandes importantes et les décisions techniques. Sois concis, "
                "mais ne perds aucune contrainte nécessaire pour continuer."
            ),
        },
        {"role": "user", "content": transcript},
    ]
    result = engine.chat(
        summary_prompt,
        max_tokens=max_summary_tokens,
        temperature=0.0,
        stream=False,
    )
    summary = result["choices"][0]["message"]["content"].strip()
    compacted = [
        {
            "role": "system",
            "content": "Mémoire structurée compactée de la conversation précédente :\n" + summary,
        },
        *tail,
    ]

    # Si les derniers messages sont eux-mêmes trop gros, on réduit progressivement
    # le contexte brut conservé. On ne supprime jamais le dernier message utilisateur.
    while len(compacted) > 2 and engine.count_messages_tokens(compacted) >= token_limit:
        compacted.pop(1)

    return compacted


def _render_transcript(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(lines)


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
    parser.add_argument("--repeat-penalty", type=float, default=None, dest="repeat_penalty",
                        help="Pénalité de répétition (défaut 1.1 ; 1.0 = aucune ; "
                             "monter à ~1.2-1.3 si le modèle boucle)")


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
        "repeat_penalty": getattr(args, "repeat_penalty", None),
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

    # doctor
    p_doctor = sub.add_parser("doctor", help="Diagnostique l'installation et un modèle")
    p_doctor.add_argument("model", nargs="?", default=None,
                          help="Nom ou chemin du modèle à inspecter")

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
    p_batch = sub.add_parser("batch", help="Traitement batch mono-instance")
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
                         help="Compatibilité seulement : ignoré, le batch est mono-instance")
    p_batch.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
                         help="Limite de tokens par tâche (auto si omis : jusqu'à la fin "
                              "de la réponse ou la limite de contexte)")
    _add_engine_args(p_batch, ctx_default=2048)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "list":   cmd_list,
        "info":   cmd_info,
        "doctor": cmd_doctor,
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

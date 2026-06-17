"""
Traitement batch mono-instance : les tâches passent en file derrière un seul modèle chargé.

Cas d'usage :
  - Fichiers  : lire N fichiers, appliquer un prompt, collecter les réponses
  - Prompts   : envoyer N prompts différents, collecter les réponses
"""

from __future__ import annotations
import json
import multiprocessing as mp
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class BatchTask:
    """Une tâche unitaire dans le batch."""
    id: int
    prompt: str                    # prompt final envoyé au modèle
    source: str = ""               # chemin fichier ou label libre
    mode: Literal["file", "prompt"] = "prompt"


@dataclass
class BatchResult:
    """Résultat d'une tâche."""
    id: int
    source: str
    prompt: str
    response: str
    duration_s: float
    error: str | None = None
    worker_pid: int = 0
    tokens: int = 0


@dataclass
class BatchSummary:
    total: int = 0
    success: int = 0
    errors: int = 0
    duration_s: float = 0.0
    tokens_per_second: float = 0.0
    results: list[BatchResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Worker (tourne dans un sous-process)
# ---------------------------------------------------------------------------

_gen_max_tokens: int | None = None


def _worker_init(model_path: str, n_ctx: int, engine_overrides: dict | None = None,
                 max_tokens: int | None = None):
    """Initialise le modèle dans le worker batch mono-instance."""
    global _engine, _gen_max_tokens
    from llm_runtime.engine import Engine
    # force=True : le batch a déjà arrêté les autres instances au niveau CLI.
    # On évite qu'un lock résiduel bloque l'initialisation du worker.
    _engine = Engine(model_path, n_ctx=n_ctx, force=True, **(engine_overrides or {}))
    _gen_max_tokens = max_tokens


def _worker_run(task: BatchTask) -> BatchResult:
    """Traite une tâche dans le worker. Le modèle est déjà chargé."""
    import os
    global _engine

    t0 = time.time()
    tokens = 0
    try:
        messages = [{"role": "user", "content": task.prompt}]
        result = _engine.chat(messages, max_tokens=_gen_max_tokens, temperature=0.7, stream=False)
        response = result["choices"][0]["message"]["content"]
        tokens = result.get("usage", {}).get("completion_tokens", 0)
        error = None
    except Exception as e:
        response = ""
        error = str(e)

    return BatchResult(
        id=task.id,
        source=task.source,
        prompt=task.prompt,
        response=response,
        duration_s=round(time.time() - t0, 2),
        error=error,
        worker_pid=os.getpid(),
        tokens=tokens,
    )


# ---------------------------------------------------------------------------
# Construction des tâches
# ---------------------------------------------------------------------------

def tasks_from_files(
    paths: list[Path],
    prompt_template: str,
    placeholder: str = "{content}",
) -> list[BatchTask]:
    """
    Crée une tâche par fichier.
    Le contenu du fichier est injecté dans le template via {placeholder}.

    Exemple :
        prompt_template = "Résume ce texte :\\n{content}"
        placeholder = "{content}"
    """
    tasks = []
    for i, path in enumerate(paths):
        try:
            content = path.read_text(errors="replace")
        except Exception as e:
            content = f"[Erreur lecture: {e}]"

        prompt = prompt_template.replace(placeholder, content)
        tasks.append(BatchTask(
            id=i,
            prompt=prompt,
            source=str(path),
            mode="file",
        ))
    return tasks


def tasks_from_prompts(prompts: list[str]) -> list[BatchTask]:
    """Crée une tâche par prompt (liste de chaînes)."""
    return [
        BatchTask(id=i, prompt=p, source=f"prompt_{i}", mode="prompt")
        for i, p in enumerate(prompts)
    ]


# ---------------------------------------------------------------------------
# Calcul du nombre optimal de jobs
# ---------------------------------------------------------------------------

def optimal_jobs(model_path: str | Path, n_ctx: int = 2048) -> int:
    """
    Retourne le nombre de workers batch par défaut.

    c3po suit désormais une politique mono-instance : un seul modèle chargé à la
    fois, les tâches batch passent en file. Cela évite les copies multiples du
    modèle en mémoire et rend le comportement identique sur Metal, CUDA et CPU.
    """
    return 1


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

def _validate_model(model_path: str) -> None:
    """
    Vérifie que le modèle se charge correctement AVANT de créer le pool.
    Lève une exception claire si ça échoue — évite les workers en boucle infinie.
    """
    from llama_cpp import Llama
    try:
        llm = Llama(model_path=model_path, n_gpu_layers=0, n_ctx=512, verbose=False)
        del llm
    except Exception as e:
        raise ValueError(
            f"Impossible de charger le modèle : {Path(model_path).name}\n"
            f"Erreur : {e}\n"
            f"Conseil : vérifiez que ce modèle est supporté (c3po list)."
        )


def run_batch(
    tasks: list[BatchTask],
    model_path: str | Path,
    jobs: int | None = None,
    n_ctx: int = 2048,
    verbose: bool = True,
    engine_overrides: dict | None = None,
    max_tokens: int | None = None,
) -> BatchSummary:
    """
    Lance le batch en multi-process.

    jobs=None → calculé automatiquement selon la mémoire disponible.
    """
    model_path = str(Path(model_path).resolve())

    # Validation préalable — évite le respawn infini de workers si le modèle ne charge pas
    if verbose:
        print(f"Validation du modèle {Path(model_path).name}…")
    _validate_model(model_path)

    if jobs not in (None, 1) and verbose:
        print("Architecture mono-instance : --jobs est ignoré, utilisation de 1 worker.")
    jobs = 1

    if verbose:
        print(f"Batch : {len(tasks)} tâches — {jobs} worker(s) — modèle : {Path(model_path).name}")
        print("-" * 60)

    t0 = time.time()
    results: list[BatchResult] = []

    # multiprocessing.pool avec initializer : chaque worker charge le modèle une fois.
    # On force le contexte "spawn" : avec "fork" (défaut Linux), les workers héritent
    # du contexte CUDA déjà initialisé dans le parent (par _validate_model / le backend
    # ggml), ce qui est incompatible avec CUDA et provoque un segfault des workers.
    # "spawn" démarre des process neufs qui initialisent CUDA proprement — c'est aussi
    # le défaut macOS, donc le comportement devient identique sur les deux plateformes.
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=jobs,
        initializer=_worker_init,
        initargs=(model_path, n_ctx, engine_overrides, max_tokens),
    ) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_run, tasks), start=1):
            results.append(result)
            if verbose:
                status = "OK" if result.error is None else "NO"
                source = Path(result.source).name if result.source else f"#{result.id}"
                print(f"  [{i:>3}/{len(tasks)}] {status} {source:<40} {result.duration_s:.1f}s  pid:{result.worker_pid}")

    total_duration = round(time.time() - t0, 2)
    successes = [r for r in results if r.error is None]

    total_tokens = sum(r.tokens for r in successes)
    tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0.0

    summary = BatchSummary(
        total=len(tasks),
        success=len(successes),
        errors=len(results) - len(successes),
        duration_s=total_duration,
        tokens_per_second=round(tokens_per_second, 2),
        results=sorted(results, key=lambda r: r.id),
    )

    if verbose:
        print("-" * 60)
        print(f"Terminé en {total_duration:.1f}s — {summary.success}/{summary.total} succès")

    return summary


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def save_results(summary: BatchSummary, output_path: str | Path):
    """Sauvegarde les résultats en JSON."""
    output_path = Path(output_path)
    data = {
        "summary": {
            "total": summary.total,
            "success": summary.success,
            "errors": summary.errors,
            "duration_s": summary.duration_s,
            "tokens_per_second": summary.tokens_per_second,
        },
        "results": [asdict(r) for r in summary.results],
    }
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"Résultats sauvegardés : {output_path}")

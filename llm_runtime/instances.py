"""
Suivi léger des instances c3po en cours d'exécution (run/serve/batch).

Chaque process qui charge un modèle écrit un fichier de lock dans un
répertoire temporaire partagé. Les commandes qui chargent un nouveau modèle
peuvent ainsi terminer les anciennes instances avant de démarrer, pour garder
un seul gros modèle vivant et des calculs mémoire prévisibles.
"""

from __future__ import annotations
import atexit
import json
import os
import signal
import tempfile
import time
from pathlib import Path

LOCK_DIR = Path(tempfile.gettempdir()) / "c3po-instances"
TERMINATE_TIMEOUT_S = 3.0


def register_instance(model_name: str, size_gb: float) -> None:
    """Enregistre l'instance courante (PID) et programme son nettoyage à la sortie."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)

    lock_path = LOCK_DIR / f"{os.getpid()}.json"
    lock_path.write_text(json.dumps({
        "pid": os.getpid(),
        "model": model_name,
        "size_gb": size_gb,
        "started_at": time.time(),
    }))

    atexit.register(lambda: lock_path.unlink(missing_ok=True))


def active_instances(exclude_pid: int | None = None) -> list[dict]:
    """
    Retourne les instances actives (PID vivant), en nettoyant au passage
    les fichiers de lock orphelins (process mort sans nettoyage propre).
    """
    if not LOCK_DIR.exists():
        return []

    exclude_pid = exclude_pid if exclude_pid is not None else os.getpid()
    instances = []

    for lock_path in LOCK_DIR.glob("*.json"):
        try:
            info = json.loads(lock_path.read_text())
        except (json.JSONDecodeError, OSError):
            lock_path.unlink(missing_ok=True)
            continue

        pid = info.get("pid")
        if pid == exclude_pid:
            continue

        try:
            os.kill(pid, 0)
        except (OSError, TypeError):
            lock_path.unlink(missing_ok=True)
            continue

        instances.append(info)

    return instances


def terminate_other_instances(timeout_s: float = TERMINATE_TIMEOUT_S) -> list[dict]:
    """
    Termine les autres instances c3po enregistrées.

    Retourne les instances visées. On envoie SIGTERM d'abord, puis SIGKILL aux
    process qui restent vivants après `timeout_s`.
    """
    targets = active_instances()
    if not targets:
        return []

    for info in targets:
        _kill_pid(info.get("pid"), signal.SIGTERM)

    deadline = time.time() + timeout_s
    remaining = targets
    while remaining and time.time() < deadline:
        time.sleep(0.05)
        remaining = [i for i in remaining if _pid_alive(i.get("pid"))]

    for info in remaining:
        _kill_pid(info.get("pid"), signal.SIGKILL)

    # Nettoie les locks des process qui viennent de sortir.
    active_instances()
    return targets


def _pid_alive(pid) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def _kill_pid(pid, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (OSError, TypeError):
        pass


def check_memory_pressure(new_size_gb: float, available_gb: float) -> str | None:
    """
    Compare la mémoire estimée requise (instances actives + nouveau modèle)
    à la mémoire disponible. Retourne un message d'avertissement si ça dépasse,
    sinon None.
    """
    from .models import FIT_MARGIN

    others = active_instances()
    total_gb = new_size_gb * FIT_MARGIN + sum(i["size_gb"] * FIT_MARGIN for i in others)

    if total_gb <= available_gb:
        return None

    lines = [
        f"⚠️  Mémoire estimée nécessaire ({total_gb:.1f} Go) > disponible ({available_gb:.1f} Go).",
        "Instances actives :",
    ]
    for i in others:
        lines.append(f"  - PID {i['pid']} : {i['model']} (~{i['size_gb']:.1f} Go)")
    lines.append(f"  - (nouveau) : ~{new_size_gb:.1f} Go")

    return "\n".join(lines)

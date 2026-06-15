"""
Détection du hardware disponible.
Retourne un HardwareProfile qui sera utilisé pour choisir
les paramètres d'inférence optimaux.
"""

from __future__ import annotations
import platform
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum


class Backend(str, Enum):
    METAL = "metal"      # Apple Silicon
    CUDA = "cuda"        # Nvidia GPU
    CPU = "cpu"          # Fallback


@dataclass
class HardwareProfile:
    backend: Backend

    # Mémoire disponible pour les poids du modèle (en Go)
    gpu_memory_gb: float        # VRAM sur Nvidia, RAM unifiée sur Apple
    cpu_memory_gb: float        # RAM système

    # Nombre de CPU cores physiques (pour le threading)
    cpu_cores: int

    # Infos lisibles
    device_name: str

    def __str__(self) -> str:
        return (
            f"Backend     : {self.backend.value}\n"
            f"Device      : {self.device_name}\n"
            f"GPU memory  : {self.gpu_memory_gb:.1f} Go\n"
            f"CPU memory  : {self.cpu_memory_gb:.1f} Go\n"
            f"CPU cores   : {self.cpu_cores}"
        )


# ---------------------------------------------------------------------------
# Détection Apple Silicon
# ---------------------------------------------------------------------------

def _detect_apple() -> HardwareProfile | None:
    """Retourne un profil si on tourne sur Apple Silicon, sinon None."""
    if platform.system() != "Darwin":
        return None

    machine = platform.machine()
    if machine != "arm64":
        return None  # Mac Intel, pas de Metal intéressant

    # RAM unifiée via sysctl
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        ).strip()
        total_ram_gb = int(raw) / (1024 ** 3)
    except Exception:
        total_ram_gb = 0.0

    # Nom du chip (ex: "Apple M4")
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        chip = "Apple Silicon"

    # Cores physiques
    try:
        cores = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"], text=True
            ).strip()
        )
    except Exception:
        cores = 4

    # Sur Apple Silicon la RAM est unifiée :
    # on peut utiliser ~75% pour le modèle, le reste pour l'OS
    usable_for_model = total_ram_gb * 0.75

    return HardwareProfile(
        backend=Backend.METAL,
        gpu_memory_gb=usable_for_model,
        cpu_memory_gb=total_ram_gb,
        cpu_cores=cores,
        device_name=chip,
    )


# ---------------------------------------------------------------------------
# Détection Nvidia / CUDA
# ---------------------------------------------------------------------------

def _detect_nvidia() -> HardwareProfile | None:
    """Retourne un profil si un GPU Nvidia est disponible, sinon None."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    if not out:
        return None

    # On prend le premier GPU trouvé
    line = out.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None

    name = parts[0]
    # nvidia-smi retourne en MiB
    vram_free_gb = int(parts[2]) / 1024

    # RAM système
    cpu_memory_gb = _get_system_ram_gb()
    cpu_cores = _get_cpu_cores()

    return HardwareProfile(
        backend=Backend.CUDA,
        gpu_memory_gb=vram_free_gb,
        cpu_memory_gb=cpu_memory_gb,
        cpu_cores=cpu_cores,
        device_name=name,
    )


# ---------------------------------------------------------------------------
# Fallback CPU
# ---------------------------------------------------------------------------

def _detect_cpu_fallback() -> HardwareProfile:
    cpu_memory_gb = _get_system_ram_gb()
    cpu_cores = _get_cpu_cores()

    return HardwareProfile(
        backend=Backend.CPU,
        gpu_memory_gb=0.0,
        cpu_memory_gb=cpu_memory_gb,
        cpu_cores=cpu_cores,
        device_name=platform.processor() or "CPU",
    )


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _get_system_ram_gb() -> float:
    try:
        if platform.system() == "Darwin":
            raw = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip()
            return int(raw) / (1024 ** 3)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _get_cpu_cores() -> int:
    try:
        if platform.system() == "Darwin":
            return int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.physicalcpu"], text=True
                ).strip()
            )
        elif platform.system() == "Linux":
            out = subprocess.check_output(
                ["nproc", "--all"], text=True
            ).strip()
            return int(out)
    except Exception:
        pass
    import os
    return os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def detect_hardware() -> HardwareProfile:
    """
    Détecte le meilleur backend disponible dans l'ordre :
    Apple Silicon (Metal) → Nvidia (CUDA) → CPU
    """
    profile = _detect_apple() or _detect_nvidia() or _detect_cpu_fallback()
    return profile


if __name__ == "__main__":
    print(detect_hardware())

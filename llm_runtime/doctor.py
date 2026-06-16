"""
Diagnostic d'environnement et de compatibilité modèle.

`c3po doctor` ne charge pas le modèle : il inspecte l'installation Python,
llama-cpp-python, le hardware détecté et l'en-tête GGUF. Le but est d'expliquer
rapidement pourquoi un modèle récent ou multimodal peut ne pas fonctionner.
"""

from __future__ import annotations
import importlib
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .gguf import model_shape
from .hardware import Backend, detect_hardware
from .models import find_model


@dataclass
class DoctorReport:
    python: str
    python_executable: str
    c3po_executable: str | None
    c3po_shebang: str | None
    platform: str
    backend: str
    device: str
    gpu_memory_gb: float
    llama_cpp_installed: bool
    llama_cpp_version: str | None = None
    gpu_offload_supported: bool | None = None
    model_name: str | None = None
    model_path: str | None = None
    model_size_gb: float | None = None
    arch: str | None = None
    n_ctx_train: int | None = None
    n_layers: int | None = None
    n_embd: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    mmproj_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


def collect_doctor(model_query: str | None = None) -> DoctorReport:
    profile = detect_hardware()
    llama_info = _llama_cpp_info()
    report = DoctorReport(
        python=sys.version.split()[0],
        python_executable=sys.executable,
        c3po_executable=shutil.which("c3po"),
        c3po_shebang=_script_shebang(shutil.which("c3po")),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        backend=profile.backend.value,
        device=profile.device_name,
        gpu_memory_gb=profile.gpu_memory_gb,
        llama_cpp_installed=llama_info["installed"],
        llama_cpp_version=llama_info["version"],
        gpu_offload_supported=llama_info["gpu_offload_supported"],
    )

    if not report.llama_cpp_installed:
        report.errors.append(
            "llama-cpp-python n'est pas installé : c3po peut lister/inspecter, "
            "mais ne peut pas charger de modèle."
        )
        report.actions.extend(_install_actions(profile.backend))
    elif profile.backend in ("cuda", "metal") and report.gpu_offload_supported is False:
        report.warnings.append(
            "llama-cpp-python semble compilé sans offload GPU. Réinstalle-le avec le "
            "backend adapté (Metal/CUDA) pour éviter l'inférence CPU."
        )
        report.actions.extend(_rebuild_actions(profile.backend))

    _check_python_mismatch(report)

    if model_query is not None:
        _inspect_model(report, model_query)

    return report


def _script_shebang(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            first = fh.readline().strip()
    except OSError:
        return None
    return first[2:] if first.startswith("#!") else None


def _check_python_mismatch(report: DoctorReport) -> None:
    if not report.c3po_shebang:
        return
    shebang = report.c3po_shebang
    current = report.python_executable
    if shebang != current and "env " not in shebang:
        report.warnings.append(
            "Le binaire c3po trouvé dans PATH ne pointe pas vers le même Python que "
            "celui qui exécute ce diagnostic. Vérifie que tu utilises le bon c3po."
        )


def _install_actions(backend: Backend) -> list[str]:
    if backend == Backend.CUDA:
        return [
            "Installer les outils de build CUDA : sudo apt install -y nvidia-cuda-toolkit",
            "Installer CMake : python3 -m pip install cmake",
            "Compiler llama-cpp-python avec CUDA : CMAKE_ARGS=\"-DGGML_CUDA=on\" FORCE_CMAKE=1 python3 -m pip install --no-cache-dir llama-cpp-python",
            "Réinstaller c3po : python3 -m pip install --user -e \".[dev]\"",
        ]
    if backend == Backend.METAL:
        return [
            "Installer/mettre à jour llama-cpp-python avec Metal : CMAKE_ARGS=\"-DGGML_METAL=on\" FORCE_CMAKE=1 python3 -m pip install --no-cache-dir llama-cpp-python",
            "Réinstaller c3po : python3 -m pip install --user -e \".[dev]\"",
        ]
    return [
        "Installer llama-cpp-python : python3 -m pip install llama-cpp-python",
        "Réinstaller c3po : python3 -m pip install --user -e \".[dev]\"",
    ]


def _rebuild_actions(backend: Backend) -> list[str]:
    if backend == Backend.CUDA:
        return [
            "Recompiler llama-cpp-python avec CUDA : CMAKE_ARGS=\"-DGGML_CUDA=on\" FORCE_CMAKE=1 python3 -m pip install --force-reinstall --no-cache-dir llama-cpp-python",
            "Relancer c3po doctor pour vérifier que l'offload GPU passe à oui.",
        ]
    if backend == Backend.METAL:
        return [
            "Recompiler llama-cpp-python avec Metal : CMAKE_ARGS=\"-DGGML_METAL=on\" FORCE_CMAKE=1 python3 -m pip install --force-reinstall --no-cache-dir llama-cpp-python",
            "Relancer c3po doctor pour vérifier que l'offload GPU passe à oui.",
        ]
    return [
        "Aucun backend GPU détecté ; l'inférence CPU est attendue.",
    ]


def _llama_cpp_info() -> dict:
    try:
        llama_cpp = importlib.import_module("llama_cpp")
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "gpu_offload_supported": None,
        }

    gpu_offload_supported = None
    fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if callable(fn):
        try:
            gpu_offload_supported = bool(fn())
        except Exception:
            gpu_offload_supported = None

    return {
        "installed": True,
        "version": getattr(llama_cpp, "__version__", None),
        "gpu_offload_supported": gpu_offload_supported,
    }


def _inspect_model(report: DoctorReport, model_query: str) -> None:
    try:
        info = find_model(model_query)
    except ValueError as e:
        report.errors.append(str(e))
        return

    report.model_name = info.name
    report.model_path = str(info.path)
    report.model_size_gb = round(info.size_gb, 2)

    shape = model_shape(info.path)
    report.arch = shape["arch"]
    report.n_ctx_train = shape["n_ctx_train"]
    report.n_layers = shape["n_layers"]
    report.n_embd = shape["n_embd"]
    report.n_heads = shape["n_heads"]
    report.n_kv_heads = shape["n_kv_heads"]

    if report.arch is None:
        report.warnings.append(
            "En-tête GGUF illisible ou incomplet. Si le chargement échoue, vérifie que "
            "le fichier est bien un GGUF supporté par ta version de llama.cpp."
        )

    path = Path(info.path)
    report.mmproj_files = sorted(p.name for p in path.parent.glob("*mmproj*.gguf"))
    if report.mmproj_files:
        report.warnings.append(
            "Projecteur multimodal mmproj détecté à côté du modèle. c3po ne sert pas "
            "encore image/audio ; support multimodal à implémenter via libmtmd/llama-server."
        )

    if info.size_gb * 1.15 > report.gpu_memory_gb:
        report.warnings.append(
            f"Le modèle ({info.size_gb:.1f} Go × marge) dépasse la mémoire estimée "
            f"disponible ({report.gpu_memory_gb:.1f} Go)."
        )

    if report.arch is not None:
        report.actions.append(
            "Si ce modèle échoue au chargement malgré un GGUF valide, mets à jour "
            "llama-cpp-python ou rebuild depuis une version récente de llama.cpp."
        )


def format_doctor(report: DoctorReport) -> str:
    def line(label: str, value) -> str:
        return f"  {label:<18}: {value}"

    gpu = (
        "?"
        if report.gpu_offload_supported is None
        else ("oui" if report.gpu_offload_supported else "non")
    )
    out = [
        "── Environnement ──",
        line("Python", report.python),
        line("Python exe", report.python_executable),
        line("c3po exe", report.c3po_executable or "?"),
        line("c3po shebang", report.c3po_shebang or "?"),
        line("Plateforme", report.platform),
        line("Backend détecté", report.backend),
        line("Device", report.device),
        line("Mémoire modèle", f"{report.gpu_memory_gb:.1f} Go"),
        "",
        "── llama-cpp-python ──",
        line("Installé", "oui" if report.llama_cpp_installed else "non"),
        line("Version", report.llama_cpp_version or "?"),
        line("Offload GPU", gpu),
    ]

    if report.model_path:
        out += [
            "",
            f"── Modèle : {report.model_name} ──",
            line("Chemin", report.model_path),
            line("Taille", f"{report.model_size_gb} Go"),
            line("Architecture", report.arch or "?"),
            line("Contexte max", report.n_ctx_train or "?"),
            line("Couches", report.n_layers or "?"),
            line("Embedding", report.n_embd or "?"),
            line("Têtes", report.n_heads or "?"),
            line("Têtes KV", report.n_kv_heads or "?"),
        ]
        if report.mmproj_files:
            out.append(line("mmproj", ", ".join(report.mmproj_files)))

    if report.warnings:
        out += ["", "── Avertissements ──"]
        out += [f"  - {w}" for w in report.warnings]

    if report.errors:
        out += ["", "── Erreurs ──"]
        out += [f"  - {e}" for e in report.errors]

    if report.actions:
        out += ["", "── Actions recommandées ──"]
        out += [f"  - {a}" for a in report.actions]

    if not report.warnings and not report.errors:
        out += ["", "OK : aucun problème évident détecté."]

    return "\n".join(out)

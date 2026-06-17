"""Windows DLL path setup for optional GPU runtime packages."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path


_DLL_HANDLES = []


def configure_windows_gpu_dll_paths() -> None:
    """Expose NVIDIA wheel DLL directories to ctypes on Windows."""
    if os.name != "nt":
        return

    for base in _site_package_dirs():
        nvidia_dir = base / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for bin_dir in nvidia_dir.glob("*/bin"):
            if bin_dir.is_dir():
                _add_dll_dir(bin_dir)


def _site_package_dirs() -> list[Path]:
    paths: list[str] = []
    try:
        paths.extend(site.getsitepackages())
    except Exception:
        pass
    user_site = site.getusersitepackages()
    if user_site:
        paths.append(user_site)

    for entry in sys.path:
        if entry and entry.endswith("site-packages"):
            paths.append(entry)

    seen: set[Path] = set()
    result: list[Path] = []
    for item in paths:
        path = Path(item)
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _add_dll_dir(path: Path) -> None:
    path_str = str(path)
    current = os.environ.get("PATH", "")
    if path_str not in current.split(os.pathsep):
        os.environ["PATH"] = path_str + os.pathsep + current

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        try:
            _DLL_HANDLES.append(add_dll_directory(path_str))
        except OSError:
            pass

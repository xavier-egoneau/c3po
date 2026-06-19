"""Checker fix_median_unnamed : la fonction median (dans stats.py) doit être correcte.
Le prompt ne nomme PAS le fichier -> teste la SÉLECTION de contexte (trouver le bon fichier
parmi des distracteurs) autant que la correction."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import uuid
from pathlib import Path


def _load_median(workdir: Path):
    path = Path(workdir) / "stats.py"
    spec = importlib.util.spec_from_file_location(f"stats_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return getattr(module, "median")


def check(workdir):
    workdir = Path(workdir)
    out = [("stats.py présent", (workdir / "stats.py").is_file(), "")]
    if not (workdir / "stats.py").is_file():
        return out
    try:
        median = _load_median(workdir)
    except Exception as exc:  # noqa: BLE001
        out.append(("stats.median importable", False, repr(exc)))
        return out
    out.append(("stats.median importable", True, ""))

    cases = [([3, 1, 2], 2), ([1, 2, 3, 4], 2.5), ([1, 1], 1.0), ([5, 2, 8, 1], 3.5), ([7], 7), ([10, 0, 5, 3], 4.0)]
    bad = []
    for xs, expected in cases:
        try:
            got = median(xs)
        except Exception as exc:  # noqa: BLE001
            got = f"err:{exc}"
        if got != expected:
            bad.append((xs, expected, got))
    out.append((f"médiane correcte ({len(cases) - len(bad)}/{len(cases)})", not bad, f"erreurs: {bad[:3]}"))
    return out

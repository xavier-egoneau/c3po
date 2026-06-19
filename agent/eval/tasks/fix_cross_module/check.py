"""Checker fix_cross_module : le symptôme est dans report.monthly_report, mais le bug est
dans stats.median (importé par report). Teste que la sélection ramène les DEUX fichiers liés."""

from __future__ import annotations

import contextlib
import importlib
import io
import sys
from pathlib import Path


def check(workdir):
    workdir = Path(workdir)
    has_both = (workdir / "report.py").is_file() and (workdir / "stats.py").is_file()
    out = [("report.py + stats.py présents", has_both, "")]
    if not has_both:
        return out

    sys.path.insert(0, str(workdir))
    for name in ("report", "stats"):
        sys.modules.pop(name, None)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            report = importlib.import_module("report")
        out.append(("report importable (chaîne d'imports OK)", True, ""))
        cases = [([1, 2, 3, 4], 2.5), ([3, 1, 2], 2), ([1, 1], 1.0), ([5, 2, 8, 1], 3.5), ([10, 0, 5, 3], 4.0)]
        bad = []
        for xs, expected in cases:
            try:
                got = report.monthly_report(xs).get("median")
            except Exception as exc:  # noqa: BLE001
                got = f"err:{exc}"
            if got != expected:
                bad.append((xs, expected, got))
        out.append((f"médiane via monthly_report ({len(cases) - len(bad)}/{len(cases)})", not bad, f"{bad[:2]}"))
    except Exception as exc:  # noqa: BLE001
        out.append(("report importable / correct", False, repr(exc)))
    finally:
        if str(workdir) in sys.path:
            sys.path.remove(str(workdir))
        for name in ("report", "stats"):
            sys.modules.pop(name, None)
    return out

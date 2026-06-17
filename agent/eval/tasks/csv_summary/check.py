"""Checker csv_summary : exécute solution.py puis vérifie summary.json."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

EXPECTED = {"books": 3, "food": 17, "toys": 7}


def check(workdir):
    workdir = Path(workdir)
    out = [("solution.py existe", (workdir / "solution.py").is_file(), "")]
    if not (workdir / "solution.py").is_file():
        return out

    proc = subprocess.run(
        [sys.executable, "solution.py"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    out.append(("solution.py s'exécute sans erreur", proc.returncode == 0, proc.stderr.strip()[-300:]))

    summary = workdir / "summary.json"
    out.append(("summary.json produit", summary.is_file(), str(summary)))
    if not summary.is_file():
        return out

    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
        normalized = {str(k): float(v) for k, v in data.items()}
        expected = {k: float(v) for k, v in EXPECTED.items()}
        out.append(("totaux par catégorie corrects", normalized == expected, f"obtenu={normalized}"))
    except Exception as exc:  # noqa: BLE001
        out.append(("totaux par catégorie corrects", False, repr(exc)))
    return out

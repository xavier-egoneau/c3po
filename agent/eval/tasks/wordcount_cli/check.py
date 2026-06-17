"""Checker wordcount_cli : invoque cli.py en subprocess sur des fichiers connus."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(workdir: Path, target: Path):
    proc = subprocess.run(
        [sys.executable, "cli.py", str(target)],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    return proc


def check(workdir):
    workdir = Path(workdir)
    out = [("cli.py existe", (workdir / "cli.py").is_file(), "")]
    if not (workdir / "cli.py").is_file():
        return out

    sample = workdir / "sample.txt"
    sample.write_text("le petit chat boit du lait\nau soleil\n", encoding="utf-8")  # 8 mots
    proc = _run(workdir, sample)
    digits = proc.stdout.strip()
    out.append(("compte correct (8)", digits == "8", f"stdout={proc.stdout.strip()!r} stderr={proc.stderr.strip()[-200:]}"))

    empty = workdir / "empty.txt"
    empty.write_text("", encoding="utf-8")
    proc_empty = _run(workdir, empty)
    out.append(("fichier vide -> 0", proc_empty.stdout.strip() == "0", f"stdout={proc_empty.stdout.strip()!r}"))
    return out

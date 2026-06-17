"""Checker email_validator : importe solution.is_valid_email et le confronte
à un jeu de cas valides/invalides, dont des pièges qu'une regex naïve rate."""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

VALID = [
    "a@b.com",
    "john.doe@example.co.uk",
    "x+tag@gmail.com",
    "u_n-a.me@sub.domain.org",
]
INVALID = [
    "plainaddress",
    "@no-local.com",
    "no-at.com",
    "a@@b.com",
    "a@b",
    "trailingdot@x.com.",
    "double..dot@x.com",
    "bad domain@x.com",
    "a@-bad.com",
]


def _load(workdir: Path):
    path = Path(workdir) / "solution.py"
    spec = importlib.util.spec_from_file_location(f"sol_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe(fn, arg):
    try:
        return fn(arg)
    except Exception:  # noqa: BLE001
        return None


def check(workdir):
    workdir = Path(workdir)
    sol = workdir / "solution.py"
    out = [("solution.py existe", sol.is_file(), str(sol))]
    if not sol.is_file():
        return out

    try:
        fn = getattr(_load(workdir), "is_valid_email")
    except Exception as exc:  # noqa: BLE001
        out.append(("is_valid_email importable", False, repr(exc)))
        return out
    out.append(("is_valid_email importable", True, ""))

    ok_valid = [a for a in VALID if _safe(fn, a) is True]
    out.append(
        (
            "adresses valides acceptées",
            len(ok_valid) == len(VALID),
            f"{len(ok_valid)}/{len(VALID)} — ratées: {[a for a in VALID if a not in ok_valid]}",
        )
    )
    ok_invalid = [a for a in INVALID if _safe(fn, a) is False]
    out.append(
        (
            "adresses invalides rejetées",
            len(ok_invalid) == len(INVALID),
            f"{len(ok_invalid)}/{len(INVALID)} — ratées: {[a for a in INVALID if a not in ok_invalid]}",
        )
    )
    return out

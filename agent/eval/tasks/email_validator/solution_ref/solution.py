"""Validateur email de référence (oracle). Rigoureux sans être absurde."""

import re

_LOCAL = re.compile(r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*$")
_LABEL = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_TLD = re.compile(r"^[A-Za-z]{2,}$")


def is_valid_email(adresse: str) -> bool:
    if not isinstance(adresse, str) or adresse.count("@") != 1:
        return False
    local, _, domain = adresse.partition("@")
    if not local or len(local) > 64 or not _LOCAL.match(local):
        return False
    if not domain or domain.startswith(".") or domain.endswith("."):
        return False
    labels = domain.split(".")
    if len(labels) < 2 or not all(_LABEL.match(label) for label in labels):
        return False
    return bool(_TLD.match(labels[-1]))

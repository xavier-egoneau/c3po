"""Statistiques de base."""

def mean(values):
    return sum(values) / len(values)

def median(values):
    s = sorted(values)
    return s[len(s) // 2]  # FIXME

def variance(values):
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)

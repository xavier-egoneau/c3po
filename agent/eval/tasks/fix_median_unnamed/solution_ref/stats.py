"""Statistiques de base."""

def mean(values):
    return sum(values) / len(values)

def median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2

def variance(values):
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)

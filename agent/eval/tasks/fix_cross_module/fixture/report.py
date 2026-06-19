"""Génération de rapports."""
from stats import median


def monthly_report(values):
    """Résumé mensuel d'une série de valeurs."""
    return {"count": len(values), "median": median(values)}

"""Checker fonctionnel fe_dashboard : KPI calculés + tri interactif (Playwright DOM)."""

from pathlib import Path

from agent.eval.webcheck import open_page, playwright_available, safe_click, safe_count, safe_text


def check(workdir):
    workdir = Path(workdir)
    out = [("index.html existe", (workdir / "index.html").is_file(), "")]
    if not (workdir / "index.html").is_file():
        return out
    if not playwright_available():
        out.append(("playwright requis", False, "pip install -e .[agent]"))
        return out

    with open_page(workdir) as (page, errors):
        nb = safe_text(page, "#nb")
        out.append(("#nb = 3 commandes", nb == "3", f"lu={nb!r}"))

        total = safe_text(page, "#total")
        out.append(("#total = 199 (59+128+12)", total is not None and "199" in total, f"lu={total!r}"))

        n = safe_count(page, "#list li")
        out.append(("liste = 3 éléments", n == 3, f"compté={n}"))

        safe_click(page, "#sort")
        first = safe_text(page, "#list li:first-child")
        out.append(("après tri décroissant, Karim (128) en tête", first is not None and "Karim" in first, f"1er={first!r}"))

        out.append(("aucune erreur JS", not errors, " | ".join(errors)[:200]))
    return out

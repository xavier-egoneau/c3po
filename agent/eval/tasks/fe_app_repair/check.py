"""Checker fonctionnel fe_app_repair (endurance) : après réparation, l'app todo doit
marcher de bout en bout — ajout, compteur, bouton Vider — sans erreur JS."""

from pathlib import Path

from agent.eval.webcheck import (
    open_page,
    playwright_available,
    safe_click,
    safe_count,
    safe_fill,
    safe_text,
)


def check(workdir):
    workdir = Path(workdir)
    out = [("index.html existe", (workdir / "index.html").is_file(), "")]
    if not (workdir / "index.html").is_file():
        return out
    if not playwright_available():
        out.append(("playwright requis", False, "pip install -e .[agent]"))
        return out

    with open_page(workdir) as (page, errors):
        out.append(("page se charge sans erreur JS", not errors, " | ".join(errors)[:200]))

        safe_fill(page, "#taskInput", "Acheter du pain")
        safe_click(page, "#addBtn")
        n1 = safe_count(page, "#todoList li")
        out.append(("ajout d'une tâche -> 1 élément", n1 == 1, f"li={n1}"))
        c1 = safe_text(page, "#count")
        out.append(("compteur = 1", c1 == "1", f"lu={c1!r}"))

        safe_fill(page, "#taskInput", "Lire un livre")
        safe_click(page, "#addBtn")
        c2 = safe_text(page, "#count")
        out.append(("compteur = 2 après 2e ajout", c2 == "2", f"lu={c2!r}"))

        safe_click(page, "#clearBtn")
        n0 = safe_count(page, "#todoList li")
        c0 = safe_text(page, "#count")
        out.append(("Vider -> liste et compteur à 0", n0 == 0 and c0 == "0", f"li={n0} count={c0!r}"))

        # erreurs JS accumulées pendant toute la session (chaque bug planté en lèverait)
        out.append(("aucune erreur JS au total", not errors, " | ".join(errors)[:200]))
    return out

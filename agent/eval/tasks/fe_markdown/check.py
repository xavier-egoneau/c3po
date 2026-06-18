"""Checker fonctionnel fe_markdown : on tape du Markdown, l'aperçu rend le HTML."""

from pathlib import Path

from agent.eval.webcheck import open_page, playwright_available, safe_fill


def check(workdir):
    workdir = Path(workdir)
    out = [("index.html existe", (workdir / "index.html").is_file(), "")]
    if not (workdir / "index.html").is_file():
        return out
    if not playwright_available():
        out.append(("playwright requis", False, "pip install -e .[agent]"))
        return out

    source = "## Sous-titre\n**gras** et *ital*\n- un\n- deux"
    with open_page(workdir) as (page, errors):
        filled = safe_fill(page, "#md", source)
        out.append(("textarea #md remplie", filled, ""))
        try:
            html = (page.inner_html("#preview", timeout=1000) or "").lower()
        except Exception as exc:  # noqa: BLE001
            out.append(("#preview lisible", False, repr(exc)))
            return out

        out.append(("titre rendu (<h2>)", "<h2" in html and "sous-titre" in html, html[:120]))
        out.append(("gras rendu (<strong>/<b>)", "<strong>" in html or "<b>" in html, ""))
        out.append(("italique rendu (<em>/<i>)", "<em>" in html or "<i>" in html, ""))
        out.append(("liste rendue (2 <li>)", html.count("<li") == 2, f"<li>×{html.count('<li')}"))
        out.append(("aucune erreur JS", not errors, " | ".join(errors)[:200]))
    return out

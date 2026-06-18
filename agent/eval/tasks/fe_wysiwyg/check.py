"""Checker fonctionnel fe_wysiwyg : taper du texte, le sélectionner, cliquer gras/italique
applique réellement le formatage (le bouton ne doit pas voler la sélection)."""

from pathlib import Path

from agent.eval.webcheck import open_page, playwright_available, safe_click


def _has_bold(html: str) -> bool:
    h = html.lower()
    return "<b>" in h or "<strong" in h or "font-weight" in h or "bold" in h


def _has_italic(html: str) -> bool:
    h = html.lower()
    return "<i>" in h or "<em" in h or "font-style" in h or "italic" in h


def check(workdir):
    workdir = Path(workdir)
    out = [("index.html existe", (workdir / "index.html").is_file(), "")]
    if not (workdir / "index.html").is_file():
        return out
    if not playwright_available():
        out.append(("playwright requis", False, "pip install -e .[agent]"))
        return out

    with open_page(workdir) as (page, errors):
        if not safe_click(page, "#editor"):
            out.append(("#editor éditable", False, "introuvable"))
            return out
        page.keyboard.type("Bonjour le monde")

        page.keyboard.press("Control+a")
        safe_click(page, "#bold")
        html_bold = page.inner_html("#editor") if page.query_selector("#editor") else ""
        out.append(("le gras s'applique à la sélection", _has_bold(html_bold), html_bold[:160]))

        page.keyboard.press("Control+a")
        safe_click(page, "#italic")
        html_ital = page.inner_html("#editor") if page.query_selector("#editor") else ""
        out.append(("l'italique s'applique", _has_italic(html_ital), html_ital[:160]))

        out.append(("le texte saisi est conservé", "Bonjour" in (page.inner_text("#editor") or ""), ""))
        out.append(("aucune erreur JS", not errors, " | ".join(errors)[:200]))
    return out

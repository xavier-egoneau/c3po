"""Checker fonctionnel fe_upload : sélectionner des fichiers -> leurs noms s'affichent."""

from pathlib import Path

from agent.eval.webcheck import open_page, playwright_available, safe_count, safe_text


def check(workdir):
    workdir = Path(workdir)
    out = [("index.html existe", (workdir / "index.html").is_file(), "")]
    if not (workdir / "index.html").is_file():
        return out
    if not playwright_available():
        out.append(("playwright requis", False, "pip install -e .[agent]"))
        return out

    # Deux fichiers à "uploader" (contenu sans importance).
    f1 = workdir / "facture.pdf"
    f2 = workdir / "photo.png"
    f1.write_text("x", encoding="utf-8")
    f2.write_text("y", encoding="utf-8")

    with open_page(workdir) as (page, errors):
        try:
            page.set_input_files("#file", [str(f1), str(f2)])
        except Exception as exc:  # noqa: BLE001
            out.append(("input #file accepte des fichiers", False, repr(exc)))
            return out
        out.append(("input #file accepte des fichiers", True, ""))

        n = safe_count(page, "#files li")
        out.append(("2 fichiers listés", n == 2, f"compté={n}"))

        listed = safe_text(page, "#files") or ""
        out.append(("noms affichés (facture.pdf, photo.png)", "facture.pdf" in listed and "photo.png" in listed, f"lu={listed!r}"))

        out.append(("aucune erreur JS", not errors, " | ".join(errors)[:200]))
    return out

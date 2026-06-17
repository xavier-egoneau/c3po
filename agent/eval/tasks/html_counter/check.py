"""Checker html_counter : structurel + câblage JS. Pas de moteur JS embarqué,
donc on vérifie la présence des ids requis, un script local (inline ou lié et
existant), l'absence de dépendance externe, et un câblage plausible (le JS
référence les deux ids et incrémente). Volontairement local à CETTE tâche —
surtout pas un linter web générique réutilisé partout."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.inline_script = ""
        self.script_srcs: list[str] = []
        self.link_hrefs: list[str] = []
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if a.get("id"):
            self.ids.add(a["id"])
        if tag == "script":
            if a.get("src"):
                self.script_srcs.append(a["src"])
            else:
                self._in_script = True
        if tag == "link" and a.get("href"):
            self.link_hrefs.append(a["href"])

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, data):
        if self._in_script:
            self.inline_script += data


def _is_external(ref: str) -> bool:
    return ref.lower().startswith(("http://", "https://", "//"))


def check(workdir):
    workdir = Path(workdir)
    index = workdir / "index.html"
    out = [("index.html existe", index.is_file(), "")]
    if not index.is_file():
        return out

    parser = _Parser()
    parser.feed(index.read_text(encoding="utf-8", errors="replace"))

    out.append(("bouton id='inc' présent", "inc" in parser.ids, f"ids={sorted(parser.ids)}"))
    out.append(("compteur id='count' présent", "count" in parser.ids, ""))

    # JS = inline, ou fichiers .js liés et réellement présents dans le dossier.
    js = parser.inline_script
    local_srcs = [s for s in parser.script_srcs if not _is_external(s)]
    for src in local_srcs:
        candidate = (index.parent / src.split("?", 1)[0]).resolve()
        if candidate.is_file():
            js += "\n" + candidate.read_text(encoding="utf-8", errors="replace")
    out.append(("script local présent (inline ou lié existant)", bool(js.strip()), ""))

    external = [s for s in parser.script_srcs if _is_external(s)] + [h for h in parser.link_hrefs if _is_external(h)]
    out.append(("aucune dépendance externe", not external, f"externes={external}"))

    wires_ids = ("count" in js) and ("inc" in js or "addeventlistener" in js.lower() or "onclick" in js.lower())
    increments = bool(re.search(r"\+\+|\+\s*1|parseint|count\s*=", js, re.I))
    out.append(("JS câble le compteur et incrémente", wires_ids and increments, ""))
    return out

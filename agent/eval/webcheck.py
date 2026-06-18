"""Helpers Playwright pour les checkers front-end (importés par les check.py).

La vérif de l'éval est DÉTERMINISTE : on lit le DOM réel après interaction, pas la
vision (qui reste l'outil flou de l'agent, pas un juge d'éval). Si Playwright manque,
le checker doit le signaler (échec explicite), pas faussement réussir.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


def playwright_available() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except ImportError:
        return False


@contextmanager
def open_page(workdir, html: str = "index.html"):
    """Ouvre workdir/html dans Chromium headless. Yield (page, errors) ; `errors`
    accumule les pageerror / console.error JS pendant toute la session."""
    from playwright.sync_api import sync_playwright

    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.on("pageerror", lambda exc, acc=errors: acc.append(f"pageerror: {exc}"))
            page.on(
                "console",
                lambda msg, acc=errors: acc.append(f"console.error: {msg.text}") if msg.type == "error" else None,
            )
            page.goto((Path(workdir) / html).resolve().as_uri(), wait_until="load", timeout=5000)
            yield page, errors
        finally:
            browser.close()


def safe_text(page, selector: str) -> str | None:
    try:
        return (page.text_content(selector, timeout=1000) or "").strip()
    except Exception:  # noqa: BLE001
        return None


def safe_click(page, selector: str) -> bool:
    try:
        page.click(selector, timeout=1000)
        return True
    except Exception:  # noqa: BLE001
        return False


def safe_fill(page, selector: str, value: str) -> bool:
    try:
        page.fill(selector, value, timeout=1000)
        return True
    except Exception:  # noqa: BLE001
        return False


def safe_visible(page, selector: str) -> bool:
    try:
        return page.is_visible(selector, timeout=1000)
    except Exception:  # noqa: BLE001
        return False


def safe_count(page, selector: str) -> int:
    try:
        return page.locator(selector).count()
    except Exception:  # noqa: BLE001
        return -1


def safe_input_value(page, selector: str) -> str | None:
    try:
        return page.input_value(selector, timeout=1000)
    except Exception:  # noqa: BLE001
        return None

"""Shared marfago-labs site chrome for static HTML reports."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THEME_PATH = _REPO_ROOT / "docs" / "lab-theme.css"
_ORG = "https://marfago-labs.github.io"


def load_lab_theme_css() -> str:
    return _THEME_PATH.read_text(encoding="utf-8")


def site_header(*, active: str = "") -> str:
    links = [
        ("home", "Home", f"{_ORG}/"),
        ("blog", "Blog", f"{_ORG}/blog/"),
        ("projects", "Projects", f"{_ORG}/projects/"),
        ("about", "About", f"{_ORG}/about/"),
    ]
    nav = "".join(
        f'<a class="nav-link{" active" if key == active else ""}" href="{href}">{label}</a>'
        for key, label, href in links
    )
    return f"""<header class="site-header">
  <div class="wrap">
    <a class="brand" href="{_ORG}/">marfago labs</a>
    <nav aria-label="Main">{nav}</nav>
  </div>
</header>"""


def site_footer(*, note: str) -> str:
    return f"""<footer class="site-footer">
  <div class="wrap">
    <p class="muted">{note}</p>
    <p class="muted small">The code is evidence. This site is the argument.</p>
  </div>
</footer>"""

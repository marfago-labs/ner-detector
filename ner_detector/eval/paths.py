"""Repo-relative path display for benchmark reports (no machine-local leaks)."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """ner-detector repository root (parent of ``ner_detector/`` package)."""
    return Path(__file__).resolve().parents[2]


def display_repo_path(path: Path | str) -> str:
    """Return a portable path for reports: repo-relative posix when possible."""
    p = Path(path).resolve()
    try:
        return p.relative_to(repo_root()).as_posix()
    except ValueError:
        return p.as_posix()

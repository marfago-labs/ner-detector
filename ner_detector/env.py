"""Load project ``.env`` into ``os.environ`` before ML backends initialize."""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_LOADED = False


def load_project_env(*, env_path: Path | None = None) -> bool:
    """Load ``.env`` from the repo root. Returns True if a file was loaded."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return False
    path = env_path or (_PROJECT_ROOT / ".env")
    if not path.is_file():
        _ENV_LOADED = True
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        _ENV_LOADED = True
        return False
    load_dotenv(path, override=False)
    _ENV_LOADED = True
    return True


def transformers_verbosity() -> str | None:
    return os.environ.get("TRANSFORMERS_VERBOSITY")

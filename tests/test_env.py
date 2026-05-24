"""Tests for .env loading."""

from __future__ import annotations

import os
from pathlib import Path

from ner_detector import env as env_module


def test_load_project_env_reads_file(tmp_path: Path, monkeypatch) -> None:
    env_module._ENV_LOADED = False
    dotenv = tmp_path / ".env"
    dotenv.write_text("TRANSFORMERS_VERBOSITY=error\n", encoding="utf-8")
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)
    assert env_module.load_project_env(env_path=dotenv) is True
    assert os.environ.get("TRANSFORMERS_VERBOSITY") == "error"
    env_module._ENV_LOADED = False

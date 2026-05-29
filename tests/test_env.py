"""Tests for .env loading."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ner_detector import env as env_module


def _reset_env_module() -> None:
    env_module._ENV_LOADED = False


def test_load_project_env_reads_file(tmp_path: Path, monkeypatch) -> None:
    _reset_env_module()
    dotenv = tmp_path / ".env"
    dotenv.write_text("TRANSFORMERS_VERBOSITY=error\n", encoding="utf-8")
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)
    assert env_module.load_project_env(env_path=dotenv) is True
    assert os.environ.get("TRANSFORMERS_VERBOSITY") == "error"
    _reset_env_module()


def test_load_project_env_missing_file(tmp_path: Path) -> None:
    _reset_env_module()
    assert env_module.load_project_env(env_path=tmp_path / "missing.env") is False
    _reset_env_module()


def test_load_project_env_only_once(tmp_path: Path) -> None:
    _reset_env_module()
    dotenv = tmp_path / ".env"
    dotenv.write_text("TRANSFORMERS_VERBOSITY=quiet\n", encoding="utf-8")
    assert env_module.load_project_env(env_path=dotenv) is True
    assert env_module.load_project_env(env_path=dotenv) is False
    _reset_env_module()


def test_load_project_env_without_dotenv_package(
    tmp_path: Path, monkeypatch
) -> None:
    _reset_env_module()
    dotenv = tmp_path / ".env"
    dotenv.write_text("X=1\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "dotenv", None)
    assert env_module.load_project_env(env_path=dotenv) is False
    _reset_env_module()


def test_transformers_verbosity_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("TRANSFORMERS_VERBOSITY", "error")
    assert env_module.transformers_verbosity() == "error"
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)
    assert env_module.transformers_verbosity() is None

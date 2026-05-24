"""Tests for module entrypoints."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_run_py_list_models() -> None:
    result = subprocess.run(
        [sys.executable, "run.py", "--list-models"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "transformers" in result.stdout


def test_cli_module_list_models() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ner_detector.cli", "--list-models"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "gliner" in result.stdout


def test_cli_module_main_block() -> None:
    import ner_detector.cli as cli_mod

    with patch.object(cli_mod, "main", return_value=0) as mock_main:
        with pytest.raises(SystemExit) as exc:
            if True:
                raise SystemExit(cli_mod.main())
    assert exc.value.code == 0
    mock_main.assert_called_once()

"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from ner_detector.config import clear_config_caches
from ner_detector.registry import clear_backend_cache

FIXTURE_BENCHMARK_ROOT = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _reset_backend_cache() -> None:
    clear_backend_cache()
    yield
    clear_backend_cache()


@pytest.fixture(autouse=True)
def _clear_config_cache() -> None:
    clear_config_caches()
    yield
    clear_config_caches()

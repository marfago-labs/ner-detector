"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from ner_detector import config as config_module
from ner_detector.config import default_model_id, load_model_config


def test_load_model_config_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_MODELS_PATH", Path("/nonexistent/default_models.yaml"))
    assert load_model_config() == {}


def test_load_model_config_non_dict_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not a mapping\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "_MODELS_PATH", path)
    assert load_model_config() == {}


def test_default_model_id_unknown_backend() -> None:
    assert default_model_id("unknown") == ""


def test_default_model_id_hardcoded_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "load_model_config", lambda: {})
    assert default_model_id("transformers") == "dslim/bert-base-NER"
    assert default_model_id("gliner") == "urchade/gliner_medium-v2.1"
    assert default_model_id("nuner") == "numind/NuNER_Zero"
    assert default_model_id("generative_ner") == "Universal-NER/UniNER-7B-type"


def test_default_model_id_from_yaml_section(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        config_module,
        "load_model_config",
        lambda: {"transformers": {"default": "  custom/ner  "}},
    )
    assert default_model_id("transformers") == "custom/ner"

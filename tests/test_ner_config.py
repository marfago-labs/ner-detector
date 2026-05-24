"""Tests for config/ner.yaml runtime settings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ner_detector import config as config_module
from ner_detector.cli import main
from ner_detector.config import (
    NerRuntimeConfig,
    load_ner_config,
    resolve_ner_settings,
    resolve_label_preset,
)


def test_load_ner_config_from_file(tmp_path: Path) -> None:
    path = tmp_path / "ner.yaml"
    path.write_text(
        "backend: gliner\nmodel_id: test/model\nthreshold: 0.7\nlabels: [a, b]\n",
        encoding="utf-8",
    )
    cfg = load_ner_config(str(path))
    assert cfg.backend == "gliner"
    assert cfg.model_id == "test/model"
    assert cfg.threshold == 0.7
    assert cfg.labels == ["a", "b"]


def test_resolve_cli_overrides_config(tmp_path: Path) -> None:
    path = tmp_path / "ner.yaml"
    path.write_text("backend: pattern\nthreshold: 0.2\n", encoding="utf-8")
    resolved = resolve_ner_settings(
        config_path=str(path),
        backend="transformers",
        model_id="override/model",
        threshold=0.9,
    )
    assert resolved.backend == "transformers"
    assert resolved.model_id == "override/model"
    assert resolved.threshold == 0.9


def test_resolve_model_default_from_catalog(tmp_path: Path) -> None:
    path = tmp_path / "ner.yaml"
    path.write_text("backend: transformers\n", encoding="utf-8")
    resolved = resolve_ner_settings(config_path=str(path))
    assert resolved.model_id == "dslim/bert-base-NER"


def test_resolve_gliner_labels_from_preset(tmp_path: Path) -> None:
    path = tmp_path / "ner.yaml"
    path.write_text("backend: gliner\nlabel_preset: scientific\n", encoding="utf-8")
    resolved = resolve_ner_settings(config_path=str(path))
    assert resolved.labels is not None
    assert "method" in resolved.labels


def test_resolve_label_preset_missing_returns_none() -> None:
    assert resolve_label_preset("nonexistent_preset_xyz") is None


def test_invalid_ner_config_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("backend: not_a_backend\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_ner_config(str(path))


def test_cli_show_config(capsys, tmp_path: Path) -> None:
    path = tmp_path / "ner.yaml"
    path.write_text("backend: pattern\nthreshold: 0.6\n", encoding="utf-8")
    code = main(["--show-config", "--config", str(path)])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["backend"] == "pattern"
    assert data["threshold"] == 0.6


def test_ner_runtime_config_labels_from_string() -> None:
    cfg = NerRuntimeConfig.model_validate(
        {"labels": "person, company , city"}
    )
    assert cfg.labels == ["person", "company", "city"]

"""Tests for label mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from ner_detector.eval import label_map as lm


def test_normalize_label_transformers() -> None:
    assert lm.normalize_label("PER") == "person"
    assert lm.normalize_label("ORG") == "organization"


def test_normalize_label_unknown_passthrough() -> None:
    assert lm.normalize_label("custom_type") == "custom_type"


def test_load_label_maps_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lm, "_DEFAULT_MAPS_PATH", tmp_path / "nope.yaml")
    lm.clear_label_map_cache()
    maps = lm.load_label_maps()
    assert "unified" in maps


def test_load_label_maps_non_dict_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "maps.yaml"
    path.write_text("- x\n", encoding="utf-8")
    monkeypatch.setattr(lm, "_DEFAULT_MAPS_PATH", path)
    lm.clear_label_map_cache()
    assert "unified" in lm.load_label_maps()


def test_load_label_maps_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "maps.yaml"
    path.write_text("unified:\n  FOO: bar\n", encoding="utf-8")
    monkeypatch.setattr(lm, "_DEFAULT_MAPS_PATH", path)
    lm.clear_label_map_cache()
    assert lm.normalize_label("FOO", "unified") == "bar"

"""Tests for gold dataset loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ner_detector.eval.loaders import (
    benchmark_root,
    dataset_path,
    load_dataset,
    load_gold_jsonl,
)


def test_benchmark_root_default() -> None:
    root = benchmark_root()
    assert (root / "datasets").is_dir()


def test_load_gold_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "1",
                "text": "Alice",
                "entities": [{"text": "Alice", "label": "person", "start": 0, "end": 5}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = load_gold_jsonl(path)
    assert len(examples) == 1


def test_load_gold_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(
        "\n"
        + json.dumps(
            {
                "id": "1",
                "text": "Bob",
                "entities": [{"text": "Bob", "label": "person", "start": 0, "end": 3}],
            }
        )
        + "\n\n",
        encoding="utf-8",
    )
    assert len(load_gold_jsonl(path)) == 1


def test_load_gold_jsonl_invalid(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid gold"):
        load_gold_jsonl(path)


def test_load_gold_jsonl_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gold_jsonl(tmp_path / "nope.jsonl")


def test_load_dataset_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset("missing_set", root=tmp_path)


def test_load_dataset_max_examples() -> None:
    examples = load_dataset("marfago_gold", max_examples=2)
    assert len(examples) == 2


def test_dataset_path() -> None:
    p = dataset_path("marfago_gold", root=benchmark_root())
    assert p.name == "marfago_gold.jsonl"

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
    resolve_benchmark_root,
)


def test_benchmark_root_default() -> None:
    root = benchmark_root()
    assert root.name == "ner-dataset"
    sibling_datasets = root / "datasets"
    if sibling_datasets.is_dir():
        assert (sibling_datasets / "arxiv_gold.jsonl").is_file()


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
    from tests.conftest import FIXTURE_BENCHMARK_ROOT

    examples = load_dataset("marfago_gold", root=FIXTURE_BENCHMARK_ROOT, max_examples=2)
    assert len(examples) == 2


def test_dataset_path() -> None:
    from tests.conftest import FIXTURE_BENCHMARK_ROOT

    p = dataset_path("marfago_gold", root=FIXTURE_BENCHMARK_ROOT)
    assert p.name == "marfago_gold.jsonl"


def test_dataset_path_prefers_sibling_ner_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labs = tmp_path / "labs"
    detector = labs / "ner-detector"
    dataset_repo = labs / "ner-dataset" / "datasets"
    dataset_repo.mkdir(parents=True)
    custom = dataset_repo / "custom.jsonl"
    custom.write_text(
        json.dumps(
            {
                "id": "1",
                "text": "x",
                "entities": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ner_detector.eval.loaders._PACKAGE_ROOT",
        detector,
    )
    monkeypatch.setattr(
        "ner_detector.eval.loaders._SIBLING_NER_DATASET",
        labs / "ner-dataset",
    )
    resolved = dataset_path("custom")
    assert resolved == custom
    examples = load_dataset("custom")
    assert examples[0].id == "1"


def test_load_arxiv_gold() -> None:
    from ner_detector.eval.gold_validate import validate_arxiv_gold

    examples = load_dataset("arxiv_gold")
    assert len(examples) == 10
    validate_arxiv_gold(examples).raise_if_invalid()
    labels = {e.label for ex in examples for e in ex.entities}
    assert labels <= {
        "model",
        "dataset",
        "benchmark",
        "metric",
        "method",
        "number",
        "organization",
    }
    for ex in examples:
        for ent in ex.entities:
            assert ex.text[ent.start : ent.end] == ent.text


def test_resolve_benchmark_root_sibling() -> None:
    cfg = Path(__file__).resolve().parents[1] / "benchmark" / "config" / "compare_generated.yaml"
    root = resolve_benchmark_root("../ner-dataset", config_path=cfg)
    assert (root / "datasets" / "synthetic_news_100.jsonl").is_file()


def test_resolve_benchmark_root_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "sample.jsonl").write_text(
        json.dumps({"id": "1", "text": "x", "entities": []}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NER_DATASET_DIR", str(datasets))
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "benchmark" / "config" / "bench.yaml"
    cfg.parent.mkdir(parents=True)
    root = resolve_benchmark_root("../ner-dataset", config_path=cfg)
    assert root == tmp_path
    assert (root / "datasets" / "sample.jsonl").is_file()


def test_load_synthetic_from_resolved_root() -> None:
    cfg = Path(__file__).resolve().parents[1] / "benchmark" / "config" / "compare_generated.yaml"
    root = resolve_benchmark_root("../ner-dataset", config_path=cfg)
    examples = load_dataset("synthetic_news_100", root=root, max_examples=3)
    assert len(examples) == 3

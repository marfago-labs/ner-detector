"""Tests for gold dataset validation and arxiv_gold compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from ner_detector.eval.gold_validate import (
    ARXIV_GOLD_LABELS,
    validate_arxiv_gold,
    validate_gold_examples,
)
from ner_detector.eval.loaders import load_dataset
from ner_detector.eval.metrics import score_example
from ner_detector.eval.runner import load_benchmark_config, run_benchmark
from ner_detector.eval.types import GoldEntity, GoldExample
from ner_detector.types import DetectedEntity


def test_validate_gold_examples_detects_bad_span() -> None:
    example = GoldExample(
        id="x",
        text="Alice works here.",
        entities=[GoldEntity(text="Bob", label="person", start=0, end=3)],
    )
    report = validate_gold_examples([example], dataset_name="t")
    assert not report.ok
    assert "mismatch" in report.issues[0].message


def test_validate_gold_examples_detects_empty_text_and_duplicates() -> None:
    example = GoldExample(
        id="x",
        text="aa",
        entities=[
            GoldEntity(text="a", label="person", start=0, end=1),
            GoldEntity(text="a", label="person", start=0, end=1),
        ],
    )
    report = validate_gold_examples([example], dataset_name="t")
    assert not report.ok
    assert any("duplicate gold span" in issue.message for issue in report.issues)


def test_validate_gold_examples_detects_empty_example_text() -> None:
    example = GoldExample(id="x", text="   ", entities=[])
    report = validate_gold_examples([example], dataset_name="t")
    assert not report.ok
    assert "empty example text" in report.issues[0].message


def test_validate_gold_examples_detects_invalid_bounds() -> None:
    example = GoldExample(
        id="x",
        text="abc",
        entities=[GoldEntity(text="abc", label="person", start=2, end=1)],
    )
    report = validate_gold_examples([example], dataset_name="t")
    assert not report.ok
    assert "invalid span offsets" in report.issues[0].message


def test_raise_if_invalid_ok() -> None:
    example = GoldExample(
        id="x",
        text="GPT-3",
        entities=[GoldEntity(text="GPT-3", label="model", start=0, end=5)],
    )
    validate_arxiv_gold([example]).raise_if_invalid()


def test_validate_gold_examples_detects_unknown_label() -> None:
    example = GoldExample(
        id="x",
        text="GPT-3 rocks.",
        entities=[GoldEntity(text="GPT-3", label="model", start=0, end=5)],
    )
    report = validate_gold_examples(
        [example],
        dataset_name="t",
        allowed_labels=frozenset({"person"}),
    )
    assert not report.ok
    assert "unexpected label" in report.issues[0].message


def test_arxiv_gold_in_sibling_repo_is_valid() -> None:
    examples = load_dataset("arxiv_gold")
    report = validate_arxiv_gold(examples)
    report.raise_if_invalid()
    assert report.n_examples == 10
    assert report.n_entities > 0
    labels = {ent.label for ex in examples for ent in ex.entities}
    assert labels <= set(ARXIV_GOLD_LABELS)


def test_bert_predictions_do_not_match_arxiv_gold_labels() -> None:
    examples = load_dataset("arxiv_gold", max_examples=1)
    preds = [
        DetectedEntity(text="Transformer", label="MISC", start=263, end=274, score=0.9),
        DetectedEntity(text="German", label="ORG", start=0, end=6, score=0.9),
    ]
    summary = score_example(examples[0], preds, label_map="unified")
    assert summary.strict.tp == 0
    assert summary.n_pred_spans == 2


def test_gliner_style_predictions_can_score_on_arxiv_gold() -> None:
    examples = load_dataset("arxiv_gold", max_examples=1)
    ex = examples[0]
    transformer = next(ent for ent in ex.entities if ent.label == "model")
    preds = [
        DetectedEntity(
            text=transformer.text,
            label="model",
            start=transformer.start,
            end=transformer.end,
            score=0.9,
        )
    ]
    summary = score_example(ex, preds, label_map="unified")
    assert summary.strict.tp == 1


def test_benchmark_config_scopes_bert_away_from_arxiv(tmp_path: Path) -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "benchmark" / "config" / "compare_backends.yaml"
    cfg = load_benchmark_config(cfg_path)
    bert = next(r for r in cfg.runs if r.name == "bert-conll")
    assert bert.datasets is not None
    assert "arxiv_gold" not in bert.datasets
    assert "synthetic_news_100" in bert.datasets


def test_run_benchmark_respects_per_run_datasets(tmp_path: Path) -> None:
    from tests.conftest import FIXTURE_BENCHMARK_ROOT

    config = tmp_path / "cfg.yaml"
    config.write_text(
        "runs:\n"
        "  - name: bert\n    backend: transformers\n    model_id: x\n"
        "    datasets:\n      - marfago_gold\n"
        "  - name: gliner\n    backend: gliner\n    model_id: y\n"
        "    datasets:\n      - conll_dev_sample\n"
        "datasets:\n  - marfago_gold\n  - conll_dev_sample\n"
        f"benchmark_root: {FIXTURE_BENCHMARK_ROOT.as_posix()}\n",
        encoding="utf-8",
    )

    def _fake_detect(text: str, **kwargs: object) -> list[DetectedEntity]:
        return [DetectedEntity(text="Alice Smith", label="person", start=0, end=11)]

    with patch("ner_detector.eval.runner.detect_entities", _fake_detect):
        benchmark = run_benchmark(config, tmp_path / "out", max_examples=1)

    pairs = {(r.run_name, r.dataset) for r in benchmark.results}
    assert ("bert", "marfago_gold") in pairs
    assert ("bert", "conll_dev_sample") not in pairs
    assert ("gliner", "conll_dev_sample") in pairs
    assert ("gliner", "marfago_gold") not in pairs


def test_fixture_marfago_gold_jsonl_is_valid_json() -> None:
    path = Path(__file__).resolve().parent / "fixtures" / "datasets" / "marfago_gold.jsonl"
    examples: list[GoldExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            examples.append(GoldExample.model_validate(json.loads(line)))
    report = validate_gold_examples(examples, dataset_name="marfago_gold")
    report.raise_if_invalid()

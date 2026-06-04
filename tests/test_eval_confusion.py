"""Tests for label confusion matrix computation."""

from __future__ import annotations

import pytest

from ner_detector.eval.confusion import (
    MISSED_COL,
    SPURIOUS_ROW,
    LabelConfusionMatrix,
    label_confusion_matrix,
)
from ner_detector.eval.metrics import score_example
from ner_detector.eval.types import EvalSpan, GoldEntity, GoldExample
from ner_detector.types import DetectedEntity


def test_label_confusion_correct_label_relaxed() -> None:
    gold = [EvalSpan(0, 5, "person", "Alice")]
    pred = [EvalSpan(0, 5, "person", "Alice")]
    matrix = label_confusion_matrix(gold, pred, mode="relaxed")
    assert matrix.get("person", "person") == 1
    assert matrix.total() == 1


def test_label_confusion_wrong_label_same_span_relaxed() -> None:
    gold = [EvalSpan(0, 6, "organization", "OpenAI")]
    pred = [EvalSpan(0, 6, "person", "OpenAI")]
    matrix = label_confusion_matrix(gold, pred, mode="relaxed")
    assert matrix.get("organization", "person") == 1
    assert matrix.get("organization", "organization") == 0


def test_label_confusion_missed_and_spurious() -> None:
    gold = [
        EvalSpan(0, 5, "person", "Alice"),
        EvalSpan(10, 16, "organization", "OpenAI"),
    ]
    pred = [EvalSpan(0, 5, "person", "Alice")]
    matrix = label_confusion_matrix(gold, pred, mode="relaxed")
    assert matrix.get("person", "person") == 1
    assert matrix.get("organization", MISSED_COL) == 1
    assert matrix.total() == 2


def test_label_confusion_spurious_prediction() -> None:
    gold = [EvalSpan(0, 5, "person", "Alice")]
    pred = [
        EvalSpan(0, 5, "person", "Alice"),
        EvalSpan(20, 26, "organization", "OpenAI"),
    ]
    matrix = label_confusion_matrix(gold, pred, mode="relaxed")
    assert matrix.get("person", "person") == 1
    assert matrix.get(SPURIOUS_ROW, "organization") == 1


def test_label_confusion_strict_requires_exact_offsets() -> None:
    gold = [EvalSpan(0, 6, "organization", "OpenAI")]
    pred = [EvalSpan(0, 5, "organization", "OpenAI")]
    relaxed = label_confusion_matrix(gold, pred, mode="relaxed")
    strict = label_confusion_matrix(gold, pred, mode="strict")
    assert relaxed.get("organization", "organization") == 1
    assert strict.get("organization", MISSED_COL) == 1
    assert strict.get(SPURIOUS_ROW, "organization") == 1


def test_label_confusion_merge() -> None:
    a = LabelConfusionMatrix(counts={("person", "person"): 2})
    b = LabelConfusionMatrix(counts={("person", "person"): 1, ("model", MISSED_COL): 1})
    merged = a.merge(b)
    assert merged.get("person", "person") == 3
    assert merged.get("model", MISSED_COL) == 1


def test_score_example_populates_confusion() -> None:
    text = "Alice works at OpenAI."
    gold = GoldExample(
        id="1",
        text=text,
        entities=[
            GoldEntity(text="Alice", label="person", start=0, end=5),
            GoldEntity(text="OpenAI", label="organization", start=14, end=20),
        ],
    )
    preds = [
        DetectedEntity(text="Alice", label="person", start=0, end=5),
        DetectedEntity(text="OpenAI", label="organization", start=14, end=20),
    ]
    summary = score_example(gold, preds, label_map="unified")
    assert summary.confusion_relaxed.get("person", "person") == 1
    assert summary.confusion_relaxed.get("organization", "organization") == 1
    assert summary.confusion_strict.get("person", "person") == 1


def test_score_summary_merge_confusion() -> None:
    from ner_detector.eval.metrics import ScoreSummary

    left = score_example(
        GoldExample(
            id="a",
            text="Bob",
            entities=[GoldEntity(text="Bob", label="person", start=0, end=3)],
        ),
        [DetectedEntity(text="Bob", label="person", start=0, end=3)],
    )
    right = score_example(
        GoldExample(
            id="b",
            text="GPT-3",
            entities=[GoldEntity(text="GPT-3", label="model", start=0, end=5)],
        ),
        [DetectedEntity(text="GPT-3", label="dataset", start=0, end=5)],
    )
    merged = ScoreSummary()
    merged.merge(left)
    merged.merge(right)
    assert merged.confusion_relaxed.get("person", "person") == 1
    assert merged.confusion_relaxed.get("model", "dataset") == 1


def test_label_confusion_to_dict() -> None:
    matrix = LabelConfusionMatrix(counts={("model", "dataset"): 2})
    data = matrix.to_dict()
    assert data["total"] == 2
    assert any(row["gold"] == "model" and row["pred"] == "dataset" for row in data["counts"])

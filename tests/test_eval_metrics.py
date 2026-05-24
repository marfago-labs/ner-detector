"""Tests for evaluation metrics."""

from __future__ import annotations

from ner_detector.eval.metrics import score_example
from ner_detector.eval.types import GoldEntity, GoldExample
from ner_detector.types import DetectedEntity


def test_strict_span_match() -> None:
    text = "Alice Smith works at OpenAI."
    gold = GoldExample(
        id="1",
        text=text,
        entities=[
            GoldEntity(text="Alice Smith", label="person", start=0, end=11),
            GoldEntity(text="OpenAI", label="organization", start=21, end=27),
        ],
    )
    preds = [
        DetectedEntity(text="Alice Smith", label="PER", start=0, end=11, score=0.9),
        DetectedEntity(text="OpenAI", label="ORG", start=21, end=27, score=0.8),
    ]
    summary = score_example(gold, preds, label_map="unified")
    sp, sr, sf1 = summary.strict_prf()
    assert sf1 == 1.0
    assert sp == 1.0
    assert sr == 1.0


def test_relaxed_recovers_partial_overlap() -> None:
    text = "OpenAI"
    gold = GoldExample(
        id="2",
        text=text,
        entities=[GoldEntity(text="OpenAI", label="organization", start=0, end=6)],
    )
    preds = [DetectedEntity(text="OpenAI", label="ORG", start=0, end=5, score=0.9)]
    summary = score_example(gold, preds, label_map="unified")
    assert summary.strict_prf()[2] == 0.0
    rp, rr, rf1 = summary.relaxed_prf()
    assert rf1 == 1.0
    assert rp == 1.0
    assert rr == 1.0


def test_score_example_skipped_unaligned_prediction() -> None:
    gold = GoldExample(id="4", text="hello", entities=[])
    preds = [DetectedEntity(text="MISSING", label="person")]
    summary = score_example(gold, preds)
    assert summary.skipped_predictions == 1
    assert summary.n_pred_spans == 0


def test_false_positive_and_negative() -> None:
    text = "Alice at OpenAI"
    gold = GoldExample(
        id="3",
        text=text,
        entities=[GoldEntity(text="Alice", label="person", start=0, end=5)],
    )
    preds = [
        DetectedEntity(text="Alice", label="person", start=0, end=5),
        DetectedEntity(text="OpenAI", label="organization", start=9, end=15),
    ]
    summary = score_example(gold, preds)
    assert summary.strict.tp == 1
    assert summary.strict.fp == 1
    assert summary.strict.fn == 0

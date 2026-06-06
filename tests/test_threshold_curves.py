"""Tests for PR/ROC threshold curve evaluation."""

from __future__ import annotations

import pytest

from ner_detector.eval.metrics import match_prediction_to_gold
from ner_detector.eval.metrics import gold_to_span
from ner_detector.eval.threshold_curves import (
    CurvePoint,
    auc_pr,
    auc_roc,
    build_ranked_predictions,
    curves_for_run,
    point_at_threshold,
    pr_curve_points,
    roc_curve_points,
    total_gold_units,
    trapezoid_auc,
    uses_threshold_backend,
)
from ner_detector.eval.types import GoldEntity, GoldExample, EvalSpan
from ner_detector.types import DetectedEntity


def test_uses_threshold_backend() -> None:
    assert uses_threshold_backend("gliner")
    assert uses_threshold_backend("transformers")
    assert not uses_threshold_backend("pattern")
    assert not uses_threshold_backend("llm")


def test_trapezoid_auc_unit_square() -> None:
    assert trapezoid_auc([0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.5)


def test_build_ranked_predictions_skips_unaligned() -> None:
    ex = GoldExample(id="1", text="hello", entities=[])
    preds = [DetectedEntity(text="MISSING", label="person", score=0.9)]
    ranked = build_ranked_predictions([ex], [preds])
    assert ranked == []


def test_pr_curve_perfect_ranking() -> None:
    text = "Alice at OpenAI"
    ex = GoldExample(
        id="1",
        text=text,
        entities=[
            GoldEntity(text="Alice", label="person", start=0, end=5),
            GoldEntity(text="OpenAI", label="organization", start=9, end=15),
        ],
    )
    preds = [
        DetectedEntity(text="Alice", label="person", start=0, end=5, score=0.99),
        DetectedEntity(text="OpenAI", label="organization", start=9, end=15, score=0.88),
    ]
    ranked = build_ranked_predictions([ex], [preds])
    points = pr_curve_points(ranked, [ex], mode="strict")
    assert points[0].recall == 0.0
    assert points[-1].recall == pytest.approx(1.0)
    assert points[-1].precision == pytest.approx(1.0)
    assert auc_pr(points) == pytest.approx(1.0, abs=0.01)


def test_pr_curve_false_positive_lowers_precision() -> None:
    text = "Alice"
    ex = GoldExample(
        id="1",
        text=text,
        entities=[GoldEntity(text="Alice", label="person", start=0, end=5)],
    )
    preds = [
        DetectedEntity(text="Alice", label="person", start=0, end=5, score=0.9),
        DetectedEntity(text="Bob", label="person", start=0, end=3, score=0.8),
    ]
    ranked = build_ranked_predictions([ex], [preds])
    points = pr_curve_points(ranked, [ex], mode="strict")
    assert points[-1].recall == pytest.approx(1.0)
    assert points[-1].precision == pytest.approx(0.5)


def test_document_mode_counts_unique_strings() -> None:
    text = "GPT-3 GPT-3"
    ex = GoldExample(
        id="1",
        text=text,
        entities=[
            GoldEntity(text="GPT-3", label="model", start=0, end=5),
            GoldEntity(text="GPT-3", label="model", start=6, end=11),
        ],
    )
    preds = [DetectedEntity(text="GPT-3", label="model", start=0, end=5, score=0.7)]
    assert total_gold_units([ex], label_map="unified", mode="document") == 1
    ranked = build_ranked_predictions([ex], [preds])
    points = pr_curve_points(ranked, [ex], mode="document")
    assert points[-1].recall == pytest.approx(1.0)


def test_roc_curve_proposal_level() -> None:
    text = "Alice"
    ex = GoldExample(
        id="1",
        text=text,
        entities=[GoldEntity(text="Alice", label="person", start=0, end=5)],
    )
    preds = [
        DetectedEntity(text="Alice", label="person", start=0, end=5, score=0.9),
        DetectedEntity(text="Wrong", label="person", start=0, end=5, score=0.4),
    ]
    ranked = build_ranked_predictions([ex], [preds])
    roc = roc_curve_points(ranked, [ex], mode="strict")
    assert roc[-1].tpr == pytest.approx(1.0)
    assert roc[-1].fpr == pytest.approx(1.0)
    assert auc_roc(roc) > 0.5


def test_roc_empty_candidates() -> None:
    ex = GoldExample(id="1", text="x", entities=[])
    roc = roc_curve_points([], [ex], mode="strict")
    assert len(roc) == 1


def test_point_at_threshold() -> None:
    points = [
        CurvePoint(0.0, 1.0, threshold=None),
        CurvePoint(0.5, 0.8, threshold=0.6),
        CurvePoint(1.0, 0.7, threshold=0.3),
    ]
    op = point_at_threshold(points, 0.5)
    assert op is not None
    assert op.threshold == pytest.approx(0.6)


def test_curves_for_run_structure() -> None:
    ex = GoldExample(
        id="1",
        text="Alice",
        entities=[GoldEntity(text="Alice", label="person", start=0, end=5)],
    )
    preds = [DetectedEntity(text="Alice", label="person", start=0, end=5, score=0.9)]
    out = curves_for_run([ex], [preds], operating_threshold=0.5)
    assert set(out.keys()) == {"strict", "relaxed", "document"}
    assert out["strict"]["auc_pr"] >= 0.0
    assert out["strict"]["operating_point"] is not None


def test_match_prediction_to_gold_modes() -> None:
    gold = [gold_to_span(GoldEntity(text="A", label="person", start=0, end=1), label_map="unified")]
    pred = EvalSpan(start=0, end=1, label="person", text="A")
    assert match_prediction_to_gold(pred, gold, set(), mode="strict") == 0
    assert match_prediction_to_gold(pred, gold, {0}, mode="strict") is None


def test_build_ranked_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        build_ranked_predictions([], [[]])

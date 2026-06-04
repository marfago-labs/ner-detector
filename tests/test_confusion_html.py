"""Tests for confusion matrix HTML rendering."""

from __future__ import annotations

from ner_detector.eval.confusion import MISSED_COL, SPURIOUS_ROW, LabelConfusionMatrix
from ner_detector.eval.confusion_html import render_confusion_matrix_html, render_run_confusion_html


def test_render_confusion_matrix_includes_headers() -> None:
    matrix = LabelConfusionMatrix(
        counts={
            ("person", "person"): 2,
            ("organization", MISSED_COL): 1,
            (SPURIOUS_ROW, "model"): 1,
        }
    )
    html = render_confusion_matrix_html(matrix, caption="Test matrix")
    assert "Gold ↓ / Pred →" in html
    assert "person" in html
    assert MISSED_COL in html
    assert SPURIOUS_ROW in html
    assert "cell-hit" in html


def test_render_run_confusion_html_skips_empty() -> None:
    assert render_run_confusion_html("run-a", LabelConfusionMatrix.empty(), LabelConfusionMatrix.empty()) == ""


def test_render_run_confusion_html_includes_both_modes() -> None:
    relaxed = LabelConfusionMatrix(counts={("model", "model"): 1})
    strict = LabelConfusionMatrix(counts={("model", "dataset"): 1})
    html = render_run_confusion_html("gliner-medium", relaxed, strict)
    assert "gliner-medium" in html
    assert "Relaxed span pairing" in html
    assert "Strict span pairing" in html

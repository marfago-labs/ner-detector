"""Tests for curve SVG rendering."""

from __future__ import annotations

from ner_detector.eval.curve_svg import (
    render_curve_panel,
    render_curves_section_html,
    render_dataset_curves_html,
)


def test_render_curve_panel_polyline() -> None:
    svg = render_curve_panel(
        title="PR test",
        x_label="Recall",
        y_label="Precision",
        series=[("run-a", [(0.0, 1.0), (1.0, 0.5)], "#112233")],
    )
    assert "<polyline" in svg
    assert "Recall" in svg


def test_render_dataset_curves_html() -> None:
    runs = [
        {
            "run_name": "gliner-medium",
            "pr": [{"recall": 0.0, "precision": 1.0}, {"recall": 1.0, "precision": 0.8}],
            "roc": [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 0.5, "tpr": 1.0}],
            "auc_pr": 0.9,
        },
    ]
    html = render_dataset_curves_html("synthetic_news_100", runs, mode="strict")
    assert "synthetic_news_100" in html
    assert "gliner-medium" in html
    assert "polyline" in html


def test_render_curves_section_empty() -> None:
    assert "No threshold curves" in render_curves_section_html({})


def test_render_curves_section_with_data() -> None:
    payload = {
        "datasets": {
            "ds1": [
                {
                    "run_name": "bert",
                    "modes": {
                        "strict": {
                            "pr": [{"recall": 0, "precision": 1}, {"recall": 1, "precision": 0.5}],
                            "roc": [{"fpr": 0, "tpr": 0}, {"fpr": 1, "tpr": 1}],
                            "auc_pr": 0.75,
                        },
                    },
                },
            ],
        },
    }
    html = render_curves_section_html(payload)
    assert "ds1" in html
    assert "strict" in html

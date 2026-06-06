"""Tests for benchmark report methodology tab."""

from __future__ import annotations

from pathlib import Path

from ner_detector.eval.report_methodology import (
    render_ner_methodology_content,
    render_report_tabs,
)
from ner_detector.eval.metrics import EntityScores, ScoreSummary
from ner_detector.eval.runner import BenchmarkResult, RunResult


def test_render_report_tabs_structure() -> None:
    html = render_report_tabs(
        tab_prefix="test",
        results_html="<p>results</p>",
        methodology_html="<p>method</p>",
    )
    assert "test-results" in html
    assert "test-methodology" in html
    assert "tab-panel-results" in html
    assert "<p>results</p>" in html
    assert "<p>method</p>" in html


def test_render_ner_methodology_repeats_unstable(tmp_path: Path) -> None:
    from ner_detector.eval.metrics import EntityScores, ScoreSummary
    from ner_detector.eval.runner import RunResult

    cfg = tmp_path / "compare.yaml"
    cfg.write_text(
        "runs:\n  - name: pattern\n    backend: pattern\n"
        "datasets:\n  - marfago_gold\n",
        encoding="utf-8",
    )
    br = BenchmarkResult(config_path=cfg, output_dir=tmp_path / "out", repeats=2)
    br.results.append(
        RunResult(
            "pattern",
            "pattern",
            None,
            "marfago_gold",
            summary=ScoreSummary(strict=EntityScores(tp=1, fp=0, fn=0)),
            scores_reproducible=False,
            n_repeats=2,
        )
    )
    html = render_ner_methodology_content(br)
    assert "Repeated trials" in html
    assert "Non-reproducible" in html


def test_render_ner_methodology_includes_config(tmp_path: Path) -> None:
    cfg = tmp_path / "compare.yaml"
    cfg.write_text(
        "runs:\n  - name: pattern\n    backend: pattern\n"
        "datasets:\n  - marfago_gold\n"
        "label_map: unified\n",
        encoding="utf-8",
    )
    br = BenchmarkResult(config_path=cfg, output_dir=tmp_path / "out")
    html = render_ner_methodology_content(br)
    assert "Benchmark process" in html


def test_render_report_tabs_with_curves() -> None:
    from ner_detector.eval.report_methodology import render_report_tabs, report_tab_styles

    html = render_report_tabs(
        tab_prefix="t",
        results_html="<p>r</p>",
        methodology_html="<p>m</p>",
        curves_html="<p>c</p>",
    )
    assert "tab-panel-curves" in html
    assert "Threshold curves" in html
    assert "t-curves" in report_tab_styles("t", has_curves=True)

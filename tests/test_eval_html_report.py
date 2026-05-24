"""Tests for HTML radar report generation."""

from __future__ import annotations

from pathlib import Path

from ner_detector.eval.html_report import render_html_report, write_html_report
from ner_detector.eval.metrics import EntityScores, ScoreSummary
from ner_detector.eval.radar_svg import (
    build_run_color_map,
    build_radar_series,
    render_radar_section_html,
)
from ner_detector.eval.runner import BenchmarkResult, RunResult


def _sample_benchmark() -> BenchmarkResult:
    summary_a = ScoreSummary(
        strict=EntityScores(tp=6, fp=2, fn=4),
        relaxed=EntityScores(tp=7, fp=1, fn=3),
        n_examples=2,
    )
    summary_b = ScoreSummary(
        strict=EntityScores(tp=10, fp=1, fn=2),
        relaxed=EntityScores(tp=10, fp=1, fn=2),
        n_examples=2,
    )
    br = BenchmarkResult(config_path=Path("cfg.yaml"), output_dir=Path("out"))
    br.results = [
        RunResult(
            "pattern",
            "pattern",
            None,
            "marfago_gold",
            summary=summary_a,
            latency_ms_per_example=1.0,
        ),
        RunResult(
            "bert-conll",
            "transformers",
            "dslim/bert-base-NER",
            "marfago_gold",
            summary=summary_b,
            latency_ms_per_example=50.0,
        ),
        RunResult(
            "gliner",
            "gliner",
            "urchade/gliner_medium-v2.1",
            "marfago_gold",
            error="missing gliner",
        ),
    ]
    return br


def test_build_radar_series_from_leaderboard() -> None:
    rows = [
        {
            "run_name": "a",
            "strict_f1": 0.3,
            "relaxed_f1": 0.4,
            "precision": 0.35,
            "recall": 0.3,
            "speed": 1.0,
        },
        {
            "run_name": "b",
            "strict_f1": 0.8,
            "relaxed_f1": 0.85,
            "precision": 0.9,
            "recall": 0.75,
            "speed": 0.2,
        },
    ]
    series = build_radar_series(rows)
    assert len(series) == 2
    assert series[0]["area"] >= 0


def test_run_color_map_stable_across_charts() -> None:
    cmap = build_run_color_map(["pattern", "bert-conll", "gliner-medium"])
    high_pattern = [
        {
            "run_name": "pattern",
            "strict_f1": 0.9,
            "relaxed_f1": 0.9,
            "precision": 0.9,
            "recall": 0.9,
            "speed": 1.0,
        },
        {
            "run_name": "bert-conll",
            "strict_f1": 0.2,
            "relaxed_f1": 0.2,
            "precision": 0.2,
            "recall": 0.2,
            "speed": 0.1,
        },
    ]
    low_pattern = [
        {
            "run_name": "bert-conll",
            "strict_f1": 0.9,
            "relaxed_f1": 0.9,
            "precision": 0.9,
            "recall": 0.9,
            "speed": 1.0,
        },
        {
            "run_name": "pattern",
            "strict_f1": 0.2,
            "relaxed_f1": 0.2,
            "precision": 0.2,
            "recall": 0.2,
            "speed": 0.1,
        },
    ]
    html_a = render_radar_section_html(high_pattern, dataset_name="a", color_map=cmap)
    html_b = render_radar_section_html(low_pattern, dataset_name="b", color_map=cmap)
    pattern_color = cmap["pattern"]
    assert f'fill="{pattern_color}"' in html_a
    assert f'fill="{pattern_color}"' in html_b
    assert cmap["bert-conll"] != pattern_color


def test_render_radar_section_svg() -> None:
    rows = [
        {
            "run_name": "pattern",
            "strict_f1": 0.3,
            "relaxed_f1": 0.35,
            "precision": 0.33,
            "recall": 0.3,
            "speed": 1.0,
        }
    ]
    html_out = render_radar_section_html(rows, dataset_name="marfago_gold")
    assert 'class="radar-chart"' in html_out
    assert "<svg" in html_out
    assert "marfago_gold" in html_out


def test_render_html_light_theme_no_chartjs() -> None:
    html = render_html_report(_sample_benchmark())
    assert "chart.js" not in html.lower()
    assert "--bg: #f4f5f7" in html
    assert 'class="radar-chart"' in html
    assert "Leaderboard" in html
    assert "rank-1" in html
    assert "missing gliner" in html
    assert 'class="report-tabs"' in html
    assert "ner-bench-methodology" in html
    assert "Metrics &amp; methodology" in html
    assert "Strict span F1" in html
    assert "Radar chart — Speed axis" in html


def test_write_html_report(tmp_path: Path) -> None:
    path = write_html_report(_sample_benchmark(), tmp_path / "report.html")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "<polygon" in text

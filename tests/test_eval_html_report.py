"""Tests for HTML radar report generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ner_detector.eval.confusion import LabelConfusionMatrix
from ner_detector.eval.html_report import (
    LATENCY_REFERENCE_MS,
    _aggregate_global_results,
    _run_to_leaderboard_row,
    _speed_score,
    render_html_report,
    write_html_report,
)
from ner_detector.eval.metrics import EntityScores, ScoreSummary
from ner_detector.eval.radar_svg import (
    build_radar_series,
    build_run_color_map,
    render_radar_section_html,
)
from ner_detector.eval.runner import BenchmarkResult, RunResult


def _confusion_sample(counts: dict[tuple[str, str], int]) -> LabelConfusionMatrix:
    matrix = LabelConfusionMatrix.empty()
    for (gold, pred), count in counts.items():
        for _ in range(count):
            matrix.increment(gold, pred)
    return matrix


def _sample_benchmark() -> BenchmarkResult:
    confusion_a = _confusion_sample({("person", "person"): 2, ("organization", "organization"): 1})
    summary_a = ScoreSummary(
        strict=EntityScores(tp=6, fp=2, fn=4),
        relaxed=EntityScores(tp=7, fp=1, fn=3),
        n_examples=2,
        n_gold_spans=10,
        confusion_relaxed=confusion_a,
        confusion_strict=confusion_a,
    )
    summary_b = ScoreSummary(
        strict=EntityScores(tp=10, fp=1, fn=2),
        relaxed=EntityScores(tp=10, fp=1, fn=2),
        n_examples=2,
        n_gold_spans=12,
    )
    summary_c = ScoreSummary(
        strict=EntityScores(tp=4, fp=3, fn=6),
        relaxed=EntityScores(tp=5, fp=2, fn=5),
        n_examples=3,
        n_gold_spans=15,
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
            "pattern",
            "pattern",
            None,
            "conll_dev_sample",
            summary=summary_c,
            latency_ms_per_example=2.0,
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
    assert "--bg: #f8f6f2" in html
    assert "site-header" in html
    assert "marfago labs" in html
    assert 'class="radar-chart"' in html
    assert 'id="global"' in html
    assert 'id="dataset-marfago-gold"' in html
    assert 'id="dataset-conll-dev-sample"' in html
    assert "Global — all datasets" in html
    assert 'class="report-index"' in html
    assert 'href="#global"' in html
    assert 'href="#dataset-marfago-gold"' in html
    assert "rank-1" in html
    assert "missing gliner" in html
    assert 'class="report-tabs"' in html
    assert "ner-bench-methodology" in html
    assert "Metrics &amp; methodology" in html
    assert "Strict span F1" in html
    assert "Radar chart" in html
    assert "Label confusion matrices" in html
    assert "confusion-matrix" in html
    assert "Gold ↓ / Pred →" in html
    assert ">Speed</text>" in html


def test_speed_score_uses_one_second_reference() -> None:
    assert LATENCY_REFERENCE_MS == 1000.0
    assert _speed_score(0.0) == 1.0
    assert _speed_score(133.0) == 0.867
    assert _speed_score(539.0) == pytest.approx(0.461)
    assert _speed_score(1000.0) == 0.0
    assert _speed_score(1500.0) == 0.0


def test_run_to_leaderboard_row_includes_speed() -> None:
    row = _run_to_leaderboard_row(_sample_benchmark().results[1])
    assert row["speed"] == _speed_score(50.0)


def test_aggregate_global_results_pools_counts() -> None:
    br = _sample_benchmark()
    global_rows = _aggregate_global_results(br.results)
    pattern = next(r for r in global_rows if r.run_name == "pattern")
    assert pattern.summary.strict.tp == 10
    assert pattern.summary.n_examples == 5
    assert pattern.dataset == "(all datasets)"


def test_write_html_report(tmp_path: Path) -> None:
    path = write_html_report(_sample_benchmark(), tmp_path / "report.html")
    assert path.is_file()
    index_path = tmp_path / "index.html"
    assert index_path.is_file()
    text = path.read_text(encoding="utf-8")
    assert index_path.read_text(encoding="utf-8") == text
    assert "<polygon" in text


def test_html_report_omits_datasets_without_results() -> None:
    br = _sample_benchmark()
    html = render_html_report(br)
    assert 'id="dataset-marfago-gold"' in html
    assert 'id="dataset-conll-dev-sample"' in html
    assert "No successful runs for synthetic" not in html
    assert "0 examples" not in html


def test_html_report_partial_run_notice() -> None:
    br = _sample_benchmark()
    br.config_path = (
        Path(__file__).resolve().parents[1] / "benchmark" / "config" / "compare_generated.yaml"
    )
    html = render_html_report(br)
    assert "Partial run" in html
    assert "synthetic_news_100" in html

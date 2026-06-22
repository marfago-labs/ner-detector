"""Tests for repo-relative path display in benchmark reports."""

from __future__ import annotations

from pathlib import Path

from ner_detector.eval.html_report import render_html_report
from ner_detector.eval.metrics import EntityScores, ScoreSummary
from ner_detector.eval.paths import display_repo_path, repo_root
from ner_detector.eval.report import render_markdown_report
from ner_detector.eval.runner import BenchmarkResult, RunResult


def test_display_repo_path_relative_under_repo() -> None:
    root = repo_root()
    abs_config = root / "benchmark" / "config" / "compare_backends.yaml"
    assert display_repo_path(abs_config) == "benchmark/config/compare_backends.yaml"


def test_display_repo_path_outside_repo_uses_posix() -> None:
    outside = Path("/tmp/outside-benchmark")
    shown = display_repo_path(outside)
    assert "\\" not in shown or shown.startswith("/")
    assert "Users" not in shown


def test_render_markdown_report_no_drive_letters() -> None:
    root = repo_root()
    benchmark = BenchmarkResult(
        config_path=root / "benchmark" / "config" / "compare_backends.yaml",
        output_dir=root / "benchmark" / "results" / "latest",
    )
    benchmark.results = [
        RunResult(
            "pattern",
            "pattern",
            None,
            "synthetic_news_100",
            summary=ScoreSummary(
                strict=EntityScores(tp=1, fp=0, fn=0),
                n_examples=1,
            ),
            latency_ms_per_example=1.0,
        ),
    ]
    md = render_markdown_report(benchmark)
    assert "Config: `benchmark/config/compare_backends.yaml`" in md
    assert "Output: `benchmark/results/latest`" in md
    assert ":\\" not in md
    assert "F:" not in md


def test_render_html_report_no_drive_letters() -> None:
    root = repo_root()
    benchmark = BenchmarkResult(
        config_path=root / "benchmark" / "config" / "compare_backends.yaml",
        output_dir=root / "benchmark" / "results" / "latest",
    )
    benchmark.results = [
        RunResult(
            "pattern",
            "pattern",
            None,
            "synthetic_news_100",
            summary=ScoreSummary(
                strict=EntityScores(tp=1, fp=0, fn=0),
                n_examples=1,
            ),
            latency_ms_per_example=1.0,
        ),
    ]
    html = render_html_report(benchmark)
    assert "benchmark/config/compare_backends.yaml" in html
    assert "benchmark/results/latest" in html
    assert "F:\\\\workspace" not in html
    assert "C:\\\\Users" not in html


def test_render_ner_methodology_content_no_drive_letters() -> None:
    from ner_detector.eval.report_methodology import render_ner_methodology_content

    root = repo_root()
    benchmark = BenchmarkResult(
        config_path=root / "benchmark" / "config" / "compare_backends.yaml",
        output_dir=root / "benchmark" / "results" / "latest",
    )
    html_out = render_ner_methodology_content(benchmark)
    assert "benchmark/config/compare_backends.yaml" in html_out
    assert "benchmark/results/latest" in html_out
    assert "F:\\\\workspace" not in html_out


def test_benchmark_result_to_dict_uses_display_paths() -> None:
    root = repo_root()
    benchmark = BenchmarkResult(
        config_path=root / "benchmark" / "config" / "compare_backends.yaml",
        output_dir=root / "benchmark" / "results" / "latest",
    )
    data = benchmark.to_dict()
    assert data["config_path"] == "benchmark/config/compare_backends.yaml"
    assert data["output_dir"] == "benchmark/results/latest"

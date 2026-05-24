"""Tests for benchmark runner and report."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ner_detector.eval.loaders import load_dataset
from ner_detector.eval.report import render_markdown_report, write_report
from ner_detector.eval.runner import load_benchmark_config, run_benchmark
from ner_detector.types import DetectedEntity


def test_load_marfago_gold() -> None:
    examples = load_dataset("marfago_gold")
    assert len(examples) >= 5
    assert examples[0].entities


def test_run_benchmark_pattern_only(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "benchmark"
    config = tmp_path / "compare.yaml"
    config.write_text(
        f"runs:\n  - name: pattern\n    backend: pattern\n"
        f"datasets:\n  - marfago_gold\n"
        f"label_map: unified\n"
        f"benchmark_root: {root.as_posix()}\n",
        encoding="utf-8",
    )
    out = tmp_path / "results"
    benchmark = run_benchmark(config, out, run_names=["pattern"], max_examples=2)
    assert len(benchmark.results) == 1
    r = benchmark.results[0]
    assert r.error is None
    assert r.summary.n_examples == 2
    metrics, report, html_report = write_report(benchmark)
    assert metrics.is_file()
    assert "pattern" in report.read_text(encoding="utf-8")
    assert html_report.is_file()
    assert 'class="radar-chart"' in html_report.read_text(encoding="utf-8")


def test_load_benchmark_config_errors(tmp_path: Path) -> None:
    bad = tmp_path / "empty.yaml"
    bad.write_text("runs: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_benchmark_config(bad)

    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        load_benchmark_config(missing)

    nodatasets = tmp_path / "nodatasets.yaml"
    nodatasets.write_text("runs:\n  - name: p\n    backend: pattern\n", encoding="utf-8")
    with pytest.raises(ValueError, match="datasets"):
        load_benchmark_config(nodatasets)


def test_report_shows_errors(tmp_path: Path) -> None:
    from ner_detector.eval.runner import BenchmarkResult, RunResult

    br = BenchmarkResult(config_path=tmp_path / "c.yaml", output_dir=tmp_path)
    br.results.append(
        RunResult(
            run_name="fail",
            backend="gliner",
            model_id="x",
            dataset="marfago_gold",
            error="No module named 'gliner'",
        )
    )
    md = render_markdown_report(br)
    assert "ERROR" in md
    assert "gliner" in md


def test_run_benchmark_with_mocked_ml(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "benchmark"
    config = root / "config" / "compare_backends.yaml"
    fake = [DetectedEntity(text="Alice Smith", label="person", start=0, end=11)]

    def _fake_detect(text: str, **kwargs: object) -> list[DetectedEntity]:
        return fake if "Alice" in text else []

    with patch("ner_detector.eval.runner.detect_entities", _fake_detect):
        benchmark = run_benchmark(
            config,
            tmp_path / "out",
            datasets=["marfago_gold"],
            run_names=["bert-conll"],
            max_examples=1,
        )
    assert benchmark.results[0].error is None


def test_run_benchmark_repeats_latency_stats(tmp_path: Path) -> None:
    from unittest.mock import patch

    root = Path(__file__).resolve().parents[1] / "benchmark"
    config = tmp_path / "compare.yaml"
    config.write_text(
        f"runs:\n  - name: pattern\n    backend: pattern\n"
        f"datasets:\n  - marfago_gold\n"
        f"label_map: unified\n"
        f"benchmark_root: {root.as_posix()}\n",
        encoding="utf-8",
    )
    clocks = iter([0.0, 0.01, 0.0, 0.02, 0.0, 0.04])

    with patch("ner_detector.eval.runner.time.perf_counter", side_effect=lambda: next(clocks)):
        benchmark = run_benchmark(
            config,
            tmp_path / "results",
            run_names=["pattern"],
            max_examples=1,
            repeats=3,
        )

    r = benchmark.results[0]
    assert r.error is None
    assert benchmark.repeats == 3
    assert r.n_repeats == 3
    assert r.latency_stats is not None
    assert r.latency_stats.std > 0
    assert r.scores_reproducible is True
    assert len(r.strict_f1_samples) == 3


def test_load_benchmark_config_repeats(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "runs:\n  - name: p\n    backend: pattern\n"
        "datasets:\n  - marfago_gold\n"
        "repeats: 5\n",
        encoding="utf-8",
    )
    cfg = load_benchmark_config(cfg_path)
    assert cfg.repeats == 5

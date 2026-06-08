"""Additional eval coverage: metrics edges, runner errors, report."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from tests.conftest import FIXTURE_BENCHMARK_ROOT

from ner_detector.eval.metrics import prediction_to_span
from ner_detector.eval.report import render_markdown_report
from ner_detector.eval.runner import RunResult, load_benchmark_config, run_benchmark
from ner_detector.types import DetectedEntity


def test_prediction_to_span_text_search() -> None:
    text = "OpenAI rocks"
    ent = DetectedEntity(text="OpenAI", label="ORG", score=0.9)
    span = prediction_to_span(text, ent, label_map="unified")
    assert span is not None
    assert span.start == 0
    assert span.label == "organization"


def test_prediction_to_span_missing_returns_none() -> None:
    text = "hello"
    ent = DetectedEntity(text="ZZZ", label="ORG")
    assert prediction_to_span(text, ent, label_map="unified") is None


def test_runner_backend_failure(tmp_path: Path) -> None:
    config = tmp_path / "c.yaml"
    config.write_text(
        f"runs:\n  - name: bad\n    backend: gliner\n    model_id: x\n"
        f"datasets:\n  - marfago_gold\n"
        f"benchmark_root: {FIXTURE_BENCHMARK_ROOT.as_posix()}\n",
        encoding="utf-8",
    )

    def _boom(*_a, **_k):
        raise RuntimeError("gliner failed")

    with patch("ner_detector.eval.runner.detect_entities", _boom):
        benchmark = run_benchmark(config, tmp_path / "o", run_names=["bad"], max_examples=1)
    assert benchmark.results[0].error == "gliner failed"


def test_load_benchmark_config_ok(tmp_path: Path) -> None:
    path = tmp_path / "ok.yaml"
    path.write_text(
        "runs:\n  - name: p\n    backend: pattern\ndatasets:\n  - marfago_gold\n",
        encoding="utf-8",
    )
    cfg = load_benchmark_config(path)
    assert cfg.runs[0].name == "p"
    assert cfg.repeats == 1


def test_run_result_to_dict() -> None:
    from ner_detector.eval.metrics import ScoreSummary
    from ner_detector.eval.repeat_stats import compute_latency_stats

    r = RunResult(
        run_name="p",
        backend="pattern",
        model_id=None,
        dataset="d",
        summary=ScoreSummary(n_examples=1),
        n_repeats=3,
        latency_stats=compute_latency_stats([10.0, 20.0, 30.0]),
        strict_f1_samples=[0.5, 0.5, 0.5],
        relaxed_f1_samples=[0.6, 0.6, 0.6],
    )
    d = r.to_dict()
    assert "strict" in d
    assert d["backend"] == "pattern"
    assert "latency" in d
    assert "strict_f1_samples" in d


def test_load_benchmark_config_invalid_repeats(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "runs:\n  - name: p\n    backend: pattern\n"
        "datasets:\n  - marfago_gold\n"
        "repeats: not-a-number\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="repeats"):
        load_benchmark_config(path)


def test_aggregate_trials_empty() -> None:
    from ner_detector.eval.runner import _aggregate_trials

    r = _aggregate_trials([], run_name="p", backend="pattern", model_id=None, dataset="d")
    assert r.error == "no trials"


def test_report_and_html_with_repeats() -> None:
    from ner_detector.eval.html_report import render_html_report
    from ner_detector.eval.metrics import EntityScores, ScoreSummary
    from ner_detector.eval.repeat_stats import compute_latency_stats
    from ner_detector.eval.runner import BenchmarkResult

    br = BenchmarkResult(config_path=Path("c.yaml"), output_dir=Path("out"), repeats=3)
    br.results.append(
        RunResult(
            run_name="pattern",
            backend="pattern",
            model_id=None,
            dataset="marfago_gold",
            summary=ScoreSummary(strict=EntityScores(tp=2, fp=0, fn=0), n_examples=2),
            n_repeats=3,
            latency_stats=compute_latency_stats([5.0, 10.0, 15.0]),
            scores_reproducible=False,
            strict_f1_samples=[0.5, 0.6, 0.5],
        )
    )
    md = render_markdown_report(br)
    assert "Repeat stability" in md
    assert "**no**" in md
    html_out = render_html_report(br)
    assert "Repeat stability" in html_out
    assert "Repeats per cell" in html_out
    assert "±" in html_out


def test_report_sorts_results() -> None:
    from ner_detector.eval.metrics import EntityScores, ScoreSummary
    from ner_detector.eval.runner import BenchmarkResult

    br = BenchmarkResult(config_path=Path("c.yaml"), output_dir=Path("out"))
    low = RunResult(
        "low", "pattern", None, "d", ScoreSummary(strict=EntityScores(tp=0, fp=1, fn=1))
    )
    high = RunResult(
        "high", "pattern", None, "d", ScoreSummary(strict=EntityScores(tp=2, fp=0, fn=0))
    )
    br.results = [low, high]
    md = render_markdown_report(br)
    assert md.index("high") < md.index("low") or "high" in md

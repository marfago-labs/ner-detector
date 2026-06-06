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
from tests.conftest import FIXTURE_BENCHMARK_ROOT


def test_load_marfago_gold() -> None:
    examples = load_dataset("marfago_gold", root=FIXTURE_BENCHMARK_ROOT)
    assert len(examples) >= 5
    assert examples[0].entities


def test_run_benchmark_pattern_only(tmp_path: Path) -> None:
    config = tmp_path / "compare.yaml"
    config.write_text(
        f"runs:\n  - name: pattern\n    backend: pattern\n"
        f"datasets:\n  - marfago_gold\n"
        f"label_map: unified\n"
        f"benchmark_root: {FIXTURE_BENCHMARK_ROOT.as_posix()}\n",
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


def test_load_benchmark_config_llm_fields(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "runs:\n"
        "  - name: llm-m\n"
        "    backend: llm\n"
        "    provider: mock\n"
        "    model_id: mock/ner\n"
        "    temperature: 0\n"
        "    max_chars: 4000\n"
        "datasets:\n  - marfago_gold\n",
        encoding="utf-8",
    )
    loaded = load_benchmark_config(cfg)
    run = loaded.runs[0]
    assert run.provider == "mock"
    assert run.model_id == "mock/ner"
    assert run.temperature == 0.0
    assert run.max_chars == 4000


def test_load_benchmark_config_label_definition_preset(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "runs:\n"
        "  - name: llm-sci\n"
        "    backend: llm\n"
        "    provider: openrouter\n"
        "    label_definition_preset: scientific\n"
        "datasets:\n  - arxiv_gold\n",
        encoding="utf-8",
    )
    run = load_benchmark_config(cfg).runs[0]
    assert run.label_definition_preset == "scientific"


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
            datasets=["synthetic_news_100"],
            run_names=["bert-conll"],
            max_examples=1,
        )
    assert benchmark.results[0].error is None


def test_run_benchmark_repeats_latency_stats(tmp_path: Path) -> None:
    from unittest.mock import patch

    config = tmp_path / "compare.yaml"
    config.write_text(
        f"runs:\n  - name: pattern\n    backend: pattern\n"
        f"datasets:\n  - marfago_gold\n"
        f"label_map: unified\n"
        f"benchmark_root: {FIXTURE_BENCHMARK_ROOT.as_posix()}\n",
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


def test_run_benchmark_clears_cache_between_repeats_not_datasets(tmp_path: Path) -> None:
    config = tmp_path / "compare.yaml"
    config.write_text(
        f"runs:\n  - name: pattern\n    backend: pattern\n"
        f"datasets:\n  - marfago_gold\n  - conll_dev_sample\n"
        f"label_map: unified\n"
        f"benchmark_root: {FIXTURE_BENCHMARK_ROOT.as_posix()}\n",
        encoding="utf-8",
    )
    clear_calls: list[int] = []

    def _track_clear() -> None:
        clear_calls.append(1)

    with patch("ner_detector.eval.runner.clear_backend_cache", side_effect=_track_clear):
        run_benchmark(
            config,
            tmp_path / "results",
            run_names=["pattern"],
            max_examples=1,
            repeats=3,
        )

    # 3 repeats → clear only before repeats 2 and 3, not before each dataset.
    assert len(clear_calls) == 2


def test_run_compare_generated_pattern_all_datasets(tmp_path: Path) -> None:
    cfg = Path(__file__).resolve().parents[1] / "benchmark" / "config" / "compare_generated.yaml"
    out = tmp_path / "results"
    benchmark = run_benchmark(
        cfg,
        out,
        run_names=["pattern"],
        max_examples=2,
    )
    datasets = {r.dataset for r in benchmark.results}
    assert datasets == {
        "arxiv_gold",
        "synthetic_news_100",
        "synthetic_blog_100",
        "synthetic_transcript_100",
        "synthetic_scientific_100",
        "synthetic_mixed_100",
    }
    metrics, report, html_report = write_report(benchmark)
    html = html_report.read_text(encoding="utf-8")
    assert "Partial run" not in html
    for name in datasets:
        assert f">{name}</h2>" in html or f"dataset-{_section_id(name)}" in html
    assert (out / "index.html").is_file()


def test_runner_continues_after_example_failure(tmp_path: Path) -> None:
    from ner_detector.eval.runner import _run_backend_on_dataset
    from ner_detector.eval.types import BackendRunSpec

    examples = load_dataset("marfago_gold", root=FIXTURE_BENCHMARK_ROOT, max_examples=2)
    calls = {"n": 0}

    def flaky_detect(text: str, spec, **kwargs: object) -> list[DetectedEntity]:
        del text, spec, kwargs
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        ent = examples[1].entities[0]
        return [
            DetectedEntity(
                text=ent.text,
                label=ent.label,
                start=ent.start,
                end=ent.end,
            ),
        ]

    spec = BackendRunSpec(name="p", backend="pattern")
    with patch("ner_detector.eval.runner._detect_for_spec", flaky_detect):
        result = _run_backend_on_dataset(spec, examples, label_map="unified")
    assert result.summary.n_examples == 1
    assert result.error is not None
    assert "1/2 examples failed" in result.error


def _section_id(name: str) -> str:
    slug = name.strip().lower().replace("_", "-").replace(" ", "-")
    return slug

"""Tests for threshold curve benchmark integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from tests.conftest import FIXTURE_BENCHMARK_ROOT

from ner_detector.eval.curve_runner import (
    run_threshold_curves,
    write_threshold_curves,
)
from ner_detector.eval.report import write_report
from ner_detector.eval.runner import BenchmarkResult, RunResult, run_benchmark
from ner_detector.eval.threshold_curves import curves_for_run
from ner_detector.eval.types import GoldEntity, GoldExample
from ner_detector.types import DetectedEntity


def test_write_threshold_curves_artifacts(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text("runs: []\ndatasets: []\n", encoding="utf-8")
    benchmark = BenchmarkResult(config_path=config, output_dir=tmp_path / "out")
    from ner_detector.eval.curve_runner import CurveRunRecord, ThresholdCurvesResult

    curves = ThresholdCurvesResult(
        config_path=config,
        output_dir=benchmark.output_dir,
        records=[
            CurveRunRecord(
                run_name="mock",
                dataset="ds",
                backend="gliner",
                model_id="m",
                operating_threshold=0.5,
                modes=curves_for_run(
                    [
                        GoldExample(
                            id="1",
                            text="Alice",
                            entities=[
                                GoldEntity(text="Alice", label="person", start=0, end=5),
                            ],
                        ),
                    ],
                    [[DetectedEntity(text="Alice", label="person", start=0, end=5, score=0.9)]],
                ),
            ),
        ],
    )
    json_path, _ = write_threshold_curves(benchmark, curves)
    assert json_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "ds" in data["datasets"]


def test_run_threshold_curves_mocked(tmp_path: Path) -> None:
    config = tmp_path / "compare.yaml"
    config.write_text(
        f"runs:\n  - name: gliner-m\n    backend: gliner\n    model_id: test/model\n"
        f"    threshold: 0.5\n    labels: [person]\n"
        f"datasets:\n  - marfago_gold\n"
        f"label_map: unified\n"
        f"benchmark_root: {FIXTURE_BENCHMARK_ROOT.as_posix()}\n",
        encoding="utf-8",
    )
    benchmark = BenchmarkResult(config_path=config, output_dir=tmp_path / "results")
    benchmark.results.append(
        RunResult(
            run_name="gliner-m",
            backend="gliner",
            model_id="test/model",
            dataset="marfago_gold",
        ),
    )
    fake_preds = [
        [DetectedEntity(text="Alice", label="person", start=0, end=5, score=0.9)],
    ]

    with patch(
        "ner_detector.eval.curve_runner._collect_predictions",
        return_value=(fake_preds, None),
    ):
        curves = run_threshold_curves(benchmark, max_examples=1)
    assert len(curves.records) == 1
    assert curves.records[0].error is None
    assert "strict" in curves.records[0].modes


def test_write_report_skips_curves_when_disabled(tmp_path: Path) -> None:
    config = tmp_path / "compare.yaml"
    config.write_text(
        f"runs:\n  - name: pattern\n    backend: pattern\n"
        f"datasets:\n  - marfago_gold\n"
        f"benchmark_root: {FIXTURE_BENCHMARK_ROOT.as_posix()}\n",
        encoding="utf-8",
    )
    benchmark = run_benchmark(config, tmp_path / "r", max_examples=1)
    write_report(benchmark, curves=False)
    assert not (tmp_path / "r" / "curves.json").exists()


def test_run_threshold_curves_collect_error(tmp_path: Path) -> None:
    config = tmp_path / "compare.yaml"
    config.write_text(
        f"runs:\n  - name: gliner-m\n    backend: gliner\n    model_id: test/model\n"
        f"    threshold: 0.5\n    labels: [person]\n"
        f"datasets:\n  - marfago_gold\n"
        f"benchmark_root: {FIXTURE_BENCHMARK_ROOT.as_posix()}\n",
        encoding="utf-8",
    )
    benchmark = BenchmarkResult(config_path=config, output_dir=tmp_path / "results")
    benchmark.results.append(
        RunResult("gliner-m", "gliner", "test/model", "marfago_gold"),
    )
    with patch(
        "ner_detector.eval.curve_runner._collect_predictions",
        return_value=([], "model load failed"),
    ):
        curves = run_threshold_curves(benchmark, max_examples=1)
    assert curves.records[0].error == "model load failed"


def test_html_report_curves_tab(tmp_path: Path) -> None:
    from ner_detector.eval.curve_runner import CurveRunRecord, ThresholdCurvesResult
    from ner_detector.eval.html_report import render_html_report

    config = tmp_path / "c.yaml"
    config.write_text("runs: []\ndatasets: []\n", encoding="utf-8")
    br = BenchmarkResult(config_path=config, output_dir=tmp_path)
    br.results.append(
        RunResult("gliner-m", "gliner", "m", "ds", error=None),
    )
    modes = curves_for_run(
        [GoldExample(id="1", text="A", entities=[])],
        [[]],
    )
    curves = ThresholdCurvesResult(
        config_path=config,
        output_dir=tmp_path,
        records=[
            CurveRunRecord("gliner-m", "ds", "gliner", "m", 0.5, modes=modes),
        ],
    )
    html = render_html_report(br, curves=curves)
    assert "Threshold curves" in html
    assert "tab-panel-curves" in html

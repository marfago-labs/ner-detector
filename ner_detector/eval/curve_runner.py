"""Collect low-threshold predictions and write PR/ROC curve artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ner_detector.detect import detect_entities
from ner_detector.eval.curve_svg import render_curves_section_html
from ner_detector.eval.loaders import load_dataset, resolve_benchmark_root
from ner_detector.eval.runner import BenchmarkResult, load_benchmark_config
from ner_detector.eval.threshold_curves import (
    DEFAULT_INFERENCE_THRESHOLD,
    curves_for_run,
    uses_threshold_backend,
)
from ner_detector.eval.types import BackendRunSpec, GoldExample
from ner_detector.registry import clear_backend_cache
from ner_detector.types import DetectedEntity, NerBackend


@dataclass
class CurveRunRecord:
    run_name: str
    dataset: str
    backend: str
    model_id: str | None
    operating_threshold: float
    modes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "dataset": self.dataset,
            "backend": self.backend,
            "model_id": self.model_id,
            "operating_threshold": self.operating_threshold,
            "inference_threshold": DEFAULT_INFERENCE_THRESHOLD,
            "modes": self.modes,
            "error": self.error,
        }


@dataclass
class ThresholdCurvesResult:
    config_path: Path
    output_dir: Path
    inference_threshold: float = DEFAULT_INFERENCE_THRESHOLD
    records: list[CurveRunRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        by_dataset: dict[str, list[dict[str, Any]]] = {}
        for rec in self.records:
            if rec.error:
                continue
            by_dataset.setdefault(rec.dataset, []).append(rec.to_dict())
        return {
            "config_path": str(self.config_path),
            "output_dir": str(self.output_dir),
            "inference_threshold": self.inference_threshold,
            "datasets": by_dataset,
            "records": [r.to_dict() for r in self.records],
        }


def _collect_predictions(
    spec: BackendRunSpec,
    examples: list[GoldExample],
) -> tuple[list[list[DetectedEntity]], str | None]:
    preds: list[list[DetectedEntity]] = []
    backend: NerBackend = spec.backend  # type: ignore[assignment]
    try:
        for ex in examples:
            batch = detect_entities(
                ex.text,
                backend=backend,
                model_id=spec.model_id,
                labels=spec.labels,
                threshold=DEFAULT_INFERENCE_THRESHOLD,
            )
            preds.append(batch)
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)
    return preds, None


def run_threshold_curves(
    benchmark: BenchmarkResult,
    *,
    max_examples: int | None = None,
) -> ThresholdCurvesResult:
    """Run curve collection for threshold backends in the benchmark config."""
    cfg = load_benchmark_config(benchmark.config_path)
    root = resolve_benchmark_root(cfg.benchmark_root, config_path=benchmark.config_path)
    result = ThresholdCurvesResult(
        config_path=benchmark.config_path,
        output_dir=benchmark.output_dir,
    )

    seen: set[tuple[str, str]] = set()
    for run_result in benchmark.results:
        if not uses_threshold_backend(run_result.backend):
            continue
        key = (run_result.run_name, run_result.dataset)
        if key in seen or run_result.error:
            continue
        seen.add(key)
        spec = next((r for r in cfg.runs if r.name == run_result.run_name), None)
        if spec is None:
            continue
        examples = load_dataset(run_result.dataset, root=root, max_examples=max_examples)
        clear_backend_cache()
        preds, err = _collect_predictions(spec, examples)
        if err:
            result.records.append(
                CurveRunRecord(
                    run_name=spec.name,
                    dataset=run_result.dataset,
                    backend=spec.backend,
                    model_id=spec.model_id,
                    operating_threshold=spec.threshold,
                    error=err,
                ),
            )
            continue
        modes = curves_for_run(
            examples,
            preds,
            label_map=cfg.label_map,
            operating_threshold=spec.threshold,
        )
        result.records.append(
            CurveRunRecord(
                run_name=spec.name,
                dataset=run_result.dataset,
                backend=spec.backend,
                model_id=spec.model_id,
                operating_threshold=spec.threshold,
                modes=modes,
            ),
        )
    return result


def write_threshold_curves(
    benchmark: BenchmarkResult,
    curves: ThresholdCurvesResult,
) -> tuple[Path, Path | None]:
    """Write ``curves.json`` and optional ``curves/index.html`` fragment path."""
    out = benchmark.output_dir
    out.mkdir(parents=True, exist_ok=True)
    curves_path = out / "curves.json"
    curves_path.write_text(
        json.dumps(curves.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    curves_dir = out / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    section = render_curves_section_html(curves.to_dict())
    snippet_path = curves_dir / "section.html"
    snippet_path.write_text(section, encoding="utf-8")
    return curves_path, snippet_path

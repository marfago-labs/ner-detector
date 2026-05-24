"""Run multi-backend NER benchmarks against gold data."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ner_detector.detect import detect_entities
from ner_detector.eval.loaders import benchmark_root, load_dataset
from ner_detector.eval.metrics import ScoreSummary, score_example
from ner_detector.eval.repeat_stats import LatencyStats, compute_latency_stats, format_latency_mean_std
from ner_detector.eval.types import BackendRunSpec, BenchmarkConfig, GoldExample
from ner_detector.registry import clear_backend_cache
from ner_detector.types import NerBackend


@dataclass
class RunResult:
    run_name: str
    backend: str
    model_id: str | None
    dataset: str
    summary: ScoreSummary = field(default_factory=ScoreSummary)
    latency_ms_total: float = 0.0
    latency_ms_per_example: float = 0.0
    error: str | None = None
    n_repeats: int = 1
    latency_stats: LatencyStats | None = None
    scores_reproducible: bool = True
    strict_f1_samples: list[float] = field(default_factory=list)
    relaxed_f1_samples: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        sp, sr, sf1 = self.summary.strict_prf()
        rp, rr, rf1 = self.summary.relaxed_prf()
        out: dict[str, Any] = {
            "run_name": self.run_name,
            "backend": self.backend,
            "model_id": self.model_id,
            "dataset": self.dataset,
            "error": self.error,
            "n_repeats": self.n_repeats,
            "scores_reproducible": self.scores_reproducible,
            "n_examples": self.summary.n_examples,
            "n_gold_spans": self.summary.n_gold_spans,
            "n_pred_spans": self.summary.n_pred_spans,
            "skipped_predictions": self.summary.skipped_predictions,
            "strict": {"precision": sp, "recall": sr, "f1": sf1, **self._counts("strict")},
            "relaxed": {"precision": rp, "recall": rr, "f1": rf1, **self._counts("relaxed")},
            "latency_ms_total": round(self.latency_ms_total, 2),
            "latency_ms_per_example": round(self.latency_ms_per_example, 2),
        }
        if self.latency_stats is not None:
            out["latency"] = self.latency_stats.to_dict()
        if self.n_repeats > 1 and self.strict_f1_samples:
            out["strict_f1_samples"] = [round(v, 6) for v in self.strict_f1_samples]
            out["relaxed_f1_samples"] = [round(v, 6) for v in self.relaxed_f1_samples]
        return out

    def _counts(self, mode: str) -> dict[str, int]:
        scores = self.summary.strict if mode == "strict" else self.summary.relaxed
        return {"tp": scores.tp, "fp": scores.fp, "fn": scores.fn}


@dataclass
class BenchmarkResult:
    config_path: Path
    output_dir: Path
    results: list[RunResult] = field(default_factory=list)
    repeats: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "output_dir": str(self.output_dir),
            "repeats": self.repeats,
            "runs": [r.to_dict() for r in self.results],
        }


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid benchmark config: {path}")

    runs_raw = raw.get("runs", [])
    datasets_raw = raw.get("datasets", [])
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ValueError("Benchmark config must include a non-empty 'runs' list")
    if not isinstance(datasets_raw, list) or not datasets_raw:
        raise ValueError("Benchmark config must include a non-empty 'datasets' list")

    runs: list[BackendRunSpec] = []
    for item in runs_raw:
        if not isinstance(item, dict):
            continue
        runs.append(
            BackendRunSpec(
                name=str(item.get("name", item.get("backend", "run"))),
                backend=str(item["backend"]),
                model_id=item.get("model_id"),
                labels=item.get("labels"),
                threshold=float(item.get("threshold", 0.5)),
            )
        )

    repeats_raw = raw.get("repeats", 5)
    try:
        repeats = max(1, int(repeats_raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid repeats in benchmark config: {repeats_raw!r}") from exc

    return BenchmarkConfig(
        runs=runs,
        datasets=[str(d) for d in datasets_raw],
        label_map=str(raw.get("label_map", "unified")),
        benchmark_root=raw.get("benchmark_root"),
        repeats=repeats,
    )


_SCORE_EPS = 1e-9


def _strict_f1(result: RunResult) -> float:
    return result.summary.strict_prf()[2]


def _relaxed_f1(result: RunResult) -> float:
    return result.summary.relaxed_prf()[2]


def _scores_reproducible(trials: list[RunResult]) -> bool:
    ok = [t for t in trials if not t.error]
    if len(ok) <= 1:
        return True
    ref = ok[0]
    rs, rr = _strict_f1(ref), _relaxed_f1(ref)
    for t in ok[1:]:
        if abs(_strict_f1(t) - rs) > _SCORE_EPS or abs(_relaxed_f1(t) - rr) > _SCORE_EPS:
            return False
        if (
            t.summary.strict.tp != ref.summary.strict.tp
            or t.summary.strict.fp != ref.summary.strict.fp
            or t.summary.strict.fn != ref.summary.strict.fn
        ):
            return False
    return True


def _aggregate_trials(
    trials: list[RunResult],
    *,
    run_name: str,
    backend: str,
    model_id: str | None,
    dataset: str,
) -> RunResult:
    if not trials:
        return RunResult(run_name, backend, model_id, dataset, error="no trials")

    errors = [t.error for t in trials if t.error]
    if errors:
        last_ok = next((t for t in reversed(trials) if not t.error), trials[-1])
        return RunResult(
            run_name=run_name,
            backend=backend,
            model_id=model_id,
            dataset=dataset,
            summary=last_ok.summary,
            latency_ms_total=last_ok.latency_ms_total,
            latency_ms_per_example=last_ok.latency_ms_per_example,
            error=errors[-1],
            n_repeats=len(trials),
        )

    latencies = [t.latency_ms_per_example for t in trials]
    stats = compute_latency_stats(latencies)
    base = trials[-1]
    return RunResult(
        run_name=run_name,
        backend=backend,
        model_id=model_id,
        dataset=dataset,
        summary=base.summary,
        latency_ms_total=statistics.mean([t.latency_ms_total for t in trials]),
        latency_ms_per_example=stats.mean,
        error=None,
        n_repeats=len(trials),
        latency_stats=stats,
        scores_reproducible=_scores_reproducible(trials),
        strict_f1_samples=[_strict_f1(t) for t in trials],
        relaxed_f1_samples=[_relaxed_f1(t) for t in trials],
    )


def _merge_summary(target: ScoreSummary, added: ScoreSummary) -> None:
    target.strict.add(added.strict)
    target.relaxed.add(added.relaxed)
    target.n_examples += added.n_examples
    target.n_gold_spans += added.n_gold_spans
    target.n_pred_spans += added.n_pred_spans
    target.skipped_predictions += added.skipped_predictions


def _run_backend_on_dataset(
    spec: BackendRunSpec,
    examples: list[GoldExample],
    *,
    label_map: str,
) -> RunResult:
    summary = ScoreSummary()
    t0 = time.perf_counter()
    backend: NerBackend = spec.backend  # type: ignore[assignment]
    try:
        for ex in examples:
            preds = detect_entities(
                ex.text,
                backend=backend,
                model_id=spec.model_id,
                labels=spec.labels,
                threshold=spec.threshold,
            )
            ex_scores = score_example(ex, preds, label_map=label_map)
            _merge_summary(summary, ex_scores)
    except Exception as exc:  # noqa: BLE001 — surface backend failures in report
        elapsed = (time.perf_counter() - t0) * 1000
        return RunResult(
            run_name=spec.name,
            backend=spec.backend,
            model_id=spec.model_id,
            dataset="",
            summary=summary,
            latency_ms_total=elapsed,
            latency_ms_per_example=elapsed / max(summary.n_examples, 1),
            error=str(exc),
        )

    elapsed = (time.perf_counter() - t0) * 1000
    n = max(summary.n_examples, 1)
    return RunResult(
        run_name=spec.name,
        backend=spec.backend,
        model_id=spec.model_id,
        dataset="",
        summary=summary,
        latency_ms_total=elapsed,
        latency_ms_per_example=elapsed / n,
        error=None,
    )


def run_benchmark(
    config_path: Path,
    output_dir: Path,
    *,
    datasets: list[str] | None = None,
    run_names: list[str] | None = None,
    max_examples: int | None = None,
    repeats: int | None = None,
) -> BenchmarkResult:
    """Execute all configured backend × dataset combinations."""
    cfg = load_benchmark_config(config_path)
    n_repeats = max(1, repeats if repeats is not None else cfg.repeats)
    root = benchmark_root(cfg.benchmark_root)
    selected_datasets = datasets if datasets else cfg.datasets
    selected_runs = (
        [r for r in cfg.runs if r.name in run_names]
        if run_names
        else cfg.runs
    )

    benchmark = BenchmarkResult(
        config_path=config_path,
        output_dir=output_dir,
        repeats=n_repeats,
    )

    for dataset_name in selected_datasets:
        examples = load_dataset(dataset_name, root=root, max_examples=max_examples)
        for spec in selected_runs:
            trials: list[RunResult] = []
            for _ in range(n_repeats):
                clear_backend_cache()
                trial = _run_backend_on_dataset(spec, examples, label_map=cfg.label_map)
                trial.dataset = dataset_name
                trials.append(trial)
            result = _aggregate_trials(
                trials,
                run_name=spec.name,
                backend=spec.backend,
                model_id=spec.model_id,
                dataset=dataset_name,
            )
            benchmark.results.append(result)

    return benchmark

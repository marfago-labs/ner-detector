#!/usr/bin/env python3
"""Run NER backend benchmark and write comparison report."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from ner_detector.env import load_project_env

load_project_env()

from ner_detector.eval.report import write_report
from ner_detector.eval.runner import run_benchmark

_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _ROOT / "config" / "compare_backends.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare NER backends on gold datasets and write a report.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=str(_DEFAULT_CONFIG),
        help="Benchmark YAML (default: benchmark/config/compare_backends.yaml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory (default: benchmark/results/run-<timestamp>)",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names (default: all in config)",
    )
    parser.add_argument(
        "--runs",
        default=None,
        help="Comma-separated run names to include (default: all in config)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit examples per dataset",
    )
    parser.add_argument(
        "--pattern-only",
        action="store_true",
        help="Only run the pattern backend (no ML download)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        metavar="N",
        help="Run each backend×dataset N times (default: config repeats or 5). "
        "Clears model cache between repeats; reports mean±std latency and score stability.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    if args.output:
        output_dir = Path(args.output)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = _ROOT / "results" / f"run-{stamp}"

    datasets = (
        [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.datasets
        else None
    )
    run_names = (
        [r.strip() for r in args.runs.split(",") if r.strip()] if args.runs else None
    )
    if args.pattern_only:
        run_names = ["pattern"]

    try:
        benchmark = run_benchmark(
            config_path,
            output_dir,
            datasets=datasets,
            run_names=run_names,
            max_examples=args.max_examples,
            repeats=args.repeats,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if benchmark.repeats > 1:
        print(f"Repeats per cell: {benchmark.repeats}")
    metrics_path, report_path, html_path = write_report(benchmark)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

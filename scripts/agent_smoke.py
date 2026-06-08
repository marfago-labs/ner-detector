#!/usr/bin/env python3
"""Offline agent smoke test: pattern backend, 2 arxiv_gold examples, JSON summary."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ner_detector.env import load_project_env

load_project_env()

from ner_detector.eval.loaders import load_dataset  # noqa: E402
from ner_detector.eval.report import write_report  # noqa: E402
from ner_detector.eval.runner import run_benchmark  # noqa: E402

_CONFIG = _ROOT / "benchmark" / "config" / "compare_backends.yaml"
_DATASET = "arxiv_gold"
_MAX_EXAMPLES = 2


def main() -> int:
    try:
        load_dataset(_DATASET, max_examples=1)
    except FileNotFoundError as exc:
        _emit(False, error=str(exc))
        return 1

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = _ROOT / "benchmark" / "results" / f"agent-smoke-{stamp}"

    try:
        benchmark = run_benchmark(
            _CONFIG,
            output_dir,
            datasets=[_DATASET],
            run_names=["pattern"],
            max_examples=_MAX_EXAMPLES,
        )
    except (FileNotFoundError, ValueError) as exc:
        _emit(False, error=str(exc))
        return 1

    metrics_path, report_path, html_path = write_report(
        benchmark,
        curves=False,
        max_examples=_MAX_EXAMPLES,
    )
    _emit(
        True,
        output_dir=str(output_dir),
        metrics_json=str(metrics_path),
        report_md=str(report_path),
        report_html=str(html_path),
        dataset=_DATASET,
        max_examples=_MAX_EXAMPLES,
        backend="pattern",
    )
    return 0


def _emit(ok: bool, **fields: object) -> None:
    payload = {"ok": ok, **fields}
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())

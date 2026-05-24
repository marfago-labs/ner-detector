"""Generate benchmark comparison reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ner_detector.eval.repeat_stats import format_latency_mean_std
from ner_detector.eval.runner import BenchmarkResult, RunResult


def _latency_cell(r: RunResult) -> str:
    if r.error:
        return "—"
    if r.latency_stats is not None:
        return format_latency_mean_std(
            r.latency_stats.mean,
            r.latency_stats.std,
            n_repeats=r.n_repeats,
        )
    return f"{r.latency_ms_per_example:.1f}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _run_key(result: RunResult) -> str:
    return f"{result.run_name} ({result.dataset})"


def render_markdown_report(benchmark: BenchmarkResult) -> str:
    repeats_note = ""
    if benchmark.repeats > 1:
        repeats_note = (
            f"\nRepeats per backend×dataset: **{benchmark.repeats}** "
            "(model cache cleared each repeat; latency = mean ± std ms/example).\n"
        )

    lat_header = (
        "Latency mean±std (ms/ex)"
        if benchmark.repeats > 1
        else "Latency (ms/ex)"
    )
    lines = [
        "# NER backend benchmark report",
        "",
        f"Config: `{benchmark.config_path}`",
        f"Output: `{benchmark.output_dir}`",
        repeats_note,
        "## Summary (strict span F1)",
        "",
        f"| Run | Dataset | Backend | Model | F1 | P | R | {lat_header} |",
        "|-----|---------|---------|-------|-----|---|---|-----------------|",
    ]

    for r in sorted(benchmark.results, key=lambda x: (-x.summary.strict_prf()[2], x.run_name)):
        if r.error:
            lines.append(
                f"| {r.run_name} | {r.dataset} | {r.backend} | {r.model_id or '—'} | "
                f"ERROR | — | — | — |"
            )
            continue
        p, rec, f1 = r.summary.strict_prf()
        stable = ""
        if benchmark.repeats > 1 and not r.error:
            stable = " ✓" if r.scores_reproducible else " ⚠"
        lines.append(
            f"| {r.run_name} | {r.dataset} | {r.backend} | {r.model_id or '—'} | "
            f"{_fmt_pct(f1)}{stable} | {_fmt_pct(p)} | {_fmt_pct(rec)} | "
            f"{_latency_cell(r)} |"
        )

    lines.extend(
        [
            "",
            "## Summary (relaxed span F1, ≥50% overlap)",
            "",
            "| Run | Dataset | F1 | P | R | TP | FP | FN |",
            "|-----|---------|-----|---|---|----|----|-----|",
        ]
    )
    for r in benchmark.results:
        if r.error:
            lines.append(f"| {r.run_name} | {r.dataset} | ERROR | — | — | — | — | — |")
            continue
        p, rec, f1 = r.summary.relaxed_prf()
        s = r.summary.relaxed
        lines.append(
            f"| {r.run_name} | {r.dataset} | {_fmt_pct(f1)} | {_fmt_pct(p)} | {_fmt_pct(rec)} | "
            f"{s.tp} | {s.fp} | {s.fn} |"
        )

    errors = [r for r in benchmark.results if r.error]
    if errors:
        lines.extend(["", "## Errors", ""])
        for r in errors:
            lines.append(f"- **{r.run_name}** on `{r.dataset}`: {r.error}")

    if benchmark.repeats > 1:
        lines.extend(
            [
                "",
                "## Repeat stability",
                "",
                "| Run | Dataset | Repeats | Scores stable | Strict F1 (all trials) | Latency min–max (ms/ex) |",
                "|-----|---------|---------|---------------|------------------------|-------------------------|",
            ]
        )
        for r in benchmark.results:
            if r.error:
                lines.append(
                    f"| {r.run_name} | {r.dataset} | {r.n_repeats} | ERROR | — | — |"
                )
                continue
            f1_vals = ", ".join(_fmt_pct(v) for v in r.strict_f1_samples) or _fmt_pct(
                r.summary.strict_prf()[2]
            )
            stable = "yes" if r.scores_reproducible else "**no**"
            lat_range = "—"
            if r.latency_stats is not None:
                st = r.latency_stats
                lat_range = f"{st.min:.1f}–{st.max:.1f}"
            lines.append(
                f"| {r.run_name} | {r.dataset} | {r.n_repeats} | {stable} | {f1_vals} | {lat_range} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- **Strict**: exact `(start, end, label)` after label normalization.",
            "- **Relaxed**: same label and span overlap ratio ≥ 0.5.",
            "- Compare backends only on datasets whose gold labels match the run (see `benchmark/config/label_maps.yaml`).",
            "- `pattern` is intended for domain gold (e.g. `marfago_gold`), not CoNLL entity types.",
        ]
    )
    if benchmark.repeats > 1:
        lines.append(
            "- **Repeats**: each trial clears the backend cache (cold model load + inference). "
            "Scores should be identical across trials; latency mean±std smooths CPU scheduling noise."
        )
    lines.append("")
    return "\n".join(lines)


def write_report(benchmark: BenchmarkResult) -> tuple[Path, Path, Path]:
    """Write ``metrics.json``, ``report.md``, and ``report.html`` under output dir."""
    out = benchmark.output_dir
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    report_path = out / "report.md"
    html_path = out / "report.html"
    metrics_path.write_text(
        json.dumps(benchmark.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_markdown_report(benchmark), encoding="utf-8")
    from ner_detector.eval.html_report import render_html_report

    html_path.write_text(render_html_report(benchmark), encoding="utf-8")
    return metrics_path, report_path, html_path

"""HTML benchmark report — marfago-labs theme (cream/teal, static SVG radar)."""

from __future__ import annotations

import html
from pathlib import Path

from ner_detector.eval.confusion_html import CONFUSION_MATRIX_CSS, render_run_confusion_html
from ner_detector.eval.curve_runner import ThresholdCurvesResult
from ner_detector.eval.curve_svg import curve_chart_css, render_curves_section_html
from ner_detector.eval.lab_chrome import load_lab_theme_css, site_footer, site_header
from ner_detector.eval.metrics import ScoreSummary
from ner_detector.eval.paths import display_repo_path
from ner_detector.eval.radar_svg import (
    RADAR_CHART_CSS,
    build_run_color_map,
    render_radar_section_html,
)
from ner_detector.eval.repeat_stats import format_latency_mean_std
from ner_detector.eval.report_methodology import (
    render_ner_methodology_content,
    render_report_tabs,
    report_tab_styles,
)
from ner_detector.eval.runner import BenchmarkResult, RunResult, load_benchmark_config

REPORT_PAGE_EXTRA_CSS = (
    """
    .report-tabs section.block { margin-bottom: 1.25rem; }
    .report-tabs section.block:last-child { margin-bottom: 0; }
    .report-tabs .notice { margin-bottom: 0.75rem; }
    th[title] { cursor: help; border-bottom: 1px dotted var(--border); }
    """
    + CONFUSION_MATRIX_CSS
)

LATENCY_REFERENCE_MS = 1000.0


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _speed_score(latency_ms: float, *, reference_ms: float = LATENCY_REFERENCE_MS) -> float:
    """Radar Speed axis: 1 at 0 ms/example, 0 at reference (default 1 s) and above."""
    if reference_ms <= 0:
        return 1.0
    return max(0.0, 1.0 - (latency_ms / reference_ms))


def _run_to_leaderboard_row(result: RunResult) -> dict[str, object]:
    sp, sr, sf1 = result.summary.strict_prf()
    _rp, _rr, rf1 = result.summary.relaxed_prf()
    dp, dr, df1 = result.summary.document_prf()
    return {
        "run_name": result.run_name,
        "backend": result.backend,
        "model_id": result.model_id,
        "dataset": result.dataset,
        "strict_f1": sf1,
        "relaxed_f1": rf1,
        "document_f1": df1,
        "precision": sp,
        "recall": sr,
        "speed": _speed_score(result.latency_ms_per_example),
    }


def _run_names_for_color_map(benchmark: BenchmarkResult) -> list[str]:
    """Config run order first, then any extra names from results."""
    names: list[str] = []
    seen: set[str] = set()
    try:
        cfg = load_benchmark_config(benchmark.config_path)
        for spec in cfg.runs:
            if spec.name not in seen:
                names.append(spec.name)
                seen.add(spec.name)
    except (OSError, ValueError):
        pass
    for result in benchmark.results:
        if result.run_name not in seen:
            names.append(result.run_name)
            seen.add(result.run_name)
    return names


def _group_by_dataset(results: list[RunResult]) -> dict[str, list[RunResult]]:
    grouped: dict[str, list[RunResult]] = {}
    for result in results:
        grouped.setdefault(result.dataset, []).append(result)
    return grouped


def _section_id(name: str) -> str:
    slug = name.strip().lower().replace("_", "-").replace(" ", "-")
    safe = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in slug)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "section"


def _dataset_order(benchmark: BenchmarkResult) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    try:
        cfg = load_benchmark_config(benchmark.config_path)
        for dataset in cfg.datasets:
            if dataset not in seen:
                names.append(dataset)
                seen.add(dataset)
    except (OSError, ValueError):
        pass
    for result in benchmark.results:
        if result.dataset not in seen:
            names.append(result.dataset)
            seen.add(result.dataset)
    return names


def _datasets_with_results(benchmark: BenchmarkResult) -> list[str]:
    """Dataset names that appear in this run (config order, then extras)."""
    present = {r.dataset for r in benchmark.results if r.dataset}
    config_order = _dataset_order(benchmark)
    ordered = [name for name in config_order if name in present]
    for name in sorted(present):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _configured_datasets(benchmark: BenchmarkResult) -> list[str]:
    try:
        cfg = load_benchmark_config(benchmark.config_path)
        return list(cfg.datasets)
    except (OSError, ValueError):
        return _datasets_with_results(benchmark)


def _partial_run_notice(benchmark: BenchmarkResult) -> str:
    configured = set(_configured_datasets(benchmark))
    run = set(_datasets_with_results(benchmark))
    skipped = configured - run
    if not skipped:
        return ""
    names = ", ".join(sorted(skipped))
    return (
        f'<p class="notice">Partial run: benchmarked '
        f"<strong>{len(run)}</strong> of <strong>{len(configured)}</strong> "
        f"configured datasets. Not included: <code>{html.escape(names)}</code>. "
        f"Re-run without <code>--datasets</code> to score all corpora.</p>"
    )


def _merge_summaries(target: ScoreSummary, added: ScoreSummary) -> None:
    target.merge(added)


def _aggregate_global_results(results: list[RunResult]) -> list[RunResult]:
    """Pool TP/FP/FN and latency across datasets per backend run."""
    buckets: dict[str, list[RunResult]] = {}
    for result in results:
        buckets.setdefault(result.run_name, []).append(result)

    aggregated: list[RunResult] = []
    for run_name, group in buckets.items():
        ok = [r for r in group if not r.error]
        errors = [r for r in group if r.error]
        if not ok:
            first = errors[0]
            aggregated.append(
                RunResult(
                    run_name=run_name,
                    backend=first.backend,
                    model_id=first.model_id,
                    dataset="(all datasets)",
                    error=first.error,
                )
            )
            continue

        summary = ScoreSummary()
        total_latency_ms = 0.0
        total_load_ms = 0.0
        total_examples = 0
        for result in ok:
            _merge_summaries(summary, result.summary)
            total_latency_ms += result.latency_ms_total
            total_load_ms += result.latency_load_ms
            total_examples += result.summary.n_examples

        template = ok[0]
        weighted_latency = (
            sum(r.latency_ms_per_example * r.summary.n_examples for r in ok) / total_examples
            if total_examples
            else 0.0
        )
        aggregated.append(
            RunResult(
                run_name=run_name,
                backend=template.backend,
                model_id=template.model_id,
                dataset="(all datasets)",
                summary=summary,
                latency_ms_total=total_latency_ms,
                latency_ms_per_example=weighted_latency,
                latency_load_ms=total_load_ms,
            )
        )
    return aggregated


def _latency_display(r: RunResult) -> str:
    if r.error:
        return "—"
    if r.latency_stats is not None:
        text = format_latency_mean_std(
            r.latency_stats.mean,
            r.latency_stats.std,
            n_repeats=r.n_repeats,
        )
        return f"{text} ms"
    return f"{r.latency_ms_per_example:.2f} ms"


def _repeat_stability_section(benchmark: BenchmarkResult) -> str:
    if benchmark.repeats <= 1:
        return ""
    rows: list[str] = []
    for r in benchmark.results:
        if r.error:
            rows.append(
                f"<tr><td><code>{html.escape(r.run_name)}</code></td>"
                f"<td>{html.escape(r.dataset)}</td>"
                f"<td colspan='4' class='err'>ERROR</td></tr>"
            )
            continue
        stable = "yes" if r.scores_reproducible else '<strong class="err">no</strong>'
        f1_vals = ", ".join(_fmt_pct(v) for v in r.strict_f1_samples)
        lat_range = "—"
        if r.latency_stats is not None:
            st = r.latency_stats
            lat_range = f"{st.min:.2f}–{st.max:.2f} ms"
        rows.append(
            f"<tr><td><code>{html.escape(r.run_name)}</code></td>"
            f"<td>{html.escape(r.dataset)}</td>"
            f"<td>{r.n_repeats}</td>"
            f"<td>{stable}</td>"
            f"<td>{f1_vals}</td>"
            f"<td>{lat_range}</td></tr>"
        )
    return f"""
    <section class="block">
      <h2>Repeat stability</h2>
      <p class="notice">Each backend×dataset was run {benchmark.repeats} times; the model cache is cleared between repeat rounds only.
      Scores should match across trials; latency min–max shows CPU/scheduling spread.</p>
      <table>
        <thead>
          <tr>
            <th>Run</th><th>Dataset</th><th>Repeats</th>
            <th>Scores stable</th><th>Strict F1 (trials)</th><th>Latency min–max</th>
          </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </section>
    """


def _leaderboard_table_rows(
    results: list[RunResult],
    benchmark: BenchmarkResult,
    *,
    show_dataset: bool = True,
) -> str:
    """HTML table rows for a result subset."""
    rows_data: list[tuple[RunResult, float]] = []
    for r in results:
        if r.error:
            continue
        _p, _rec, f1 = r.summary.document_prf()
        rows_data.append((r, f1))
    rows_data.sort(key=lambda x: (-x[1], x[0].run_name, x[0].dataset))

    lines: list[str] = []
    for rank, (r, _f1) in enumerate(rows_data, start=1):
        rank_cls = "rank-1" if rank == 1 else ""
        _p, _rec, doc_f1 = r.summary.document_prf()
        _p2, _r2, sf1 = r.summary.strict_prf()
        stable_mark = ""
        if benchmark.repeats > 1 and not r.scores_reproducible:
            stable_mark = ' <span class="err" title="F1 varied across repeats">⚠</span>'
        dataset_cell = f"<td>{html.escape(r.dataset)}</td>" if show_dataset else ""
        lines.append(
            f"<tr class='{rank_cls}'>"
            f"<td>{rank}</td>"
            f"<td><code>{html.escape(r.run_name)}</code></td>"
            f"{dataset_cell}"
            f"<td>{html.escape(r.backend)}</td>"
            f"<td><code>{html.escape(r.model_id or '—')}</code></td>"
            f"<td>{_fmt_pct(doc_f1)}{stable_mark}</td>"
            f"<td>{_fmt_pct(sf1)}</td>"
            f"<td>{_latency_display(r)}</td>"
            f"</tr>"
        )

    err_colspan = "4"
    for r in results:
        if r.error:
            dataset_cell = f"<td>{html.escape(r.dataset)}</td>" if show_dataset else ""
            lines.append(
                "<tr>"
                f"<td>—</td>"
                f"<td><code>{html.escape(r.run_name)}</code></td>"
                f"{dataset_cell}"
                f"<td>{html.escape(r.backend)}</td>"
                f"<td colspan='{err_colspan}' class='err'>ERROR: {html.escape(r.error or '')}</td>"
                f"</tr>"
            )

    return "".join(lines)


def _leaderboard_table_html(
    results: list[RunResult],
    benchmark: BenchmarkResult,
    *,
    show_dataset: bool,
    lat_title: str,
) -> str:
    dataset_header = "<th>Dataset</th>" if show_dataset else ""
    col_count = 8 if show_dataset else 7
    rows = _leaderboard_table_rows(results, benchmark, show_dataset=show_dataset)
    return f"""
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Run</th>
            {dataset_header}
            <th>Backend</th>
            <th>Model</th>
            <th title="Document-level unique string match">Doc F1</th>
            <th title="Exact span + unified label match">Strict F1</th>
            <th title="{html.escape(lat_title)}">Latency</th>
          </tr>
        </thead>
        <tbody>{rows or f"<tr><td colspan='{col_count}'>No results</td></tr>"}</tbody>
      </table>
    """


def _confusion_matrices_html(results: list[RunResult]) -> str:
    """Per-run label confusion tables for a dataset or global section."""
    ok = sorted(
        [r for r in results if not r.error],
        key=lambda r: (-r.summary.strict_prf()[2], r.run_name),
    )
    if not ok:
        return ""
    blocks: list[str] = []
    for result in ok:
        block = render_run_confusion_html(
            result.run_name,
            result.summary.confusion_relaxed,
            result.summary.confusion_strict,
        )
        if block.strip():
            blocks.append(block)
    if not blocks:
        return ""
    return f"""
    <section class="block confusion-section">
      <h3>Label confusion matrices</h3>
      <p class="confusion-note">Rows = gold labels; columns = predicted labels.
      Diagonal (green) = correct label at a paired span. Column <code>{html.escape("∅ missed")}</code>
      = gold entity with no paired prediction. Row <code>{html.escape("∅ spurious")}</code>
      = prediction with no paired gold span.</p>
      {"".join(blocks)}
    </section>
    """


def _radar_for_results(
    results: list[RunResult],
    *,
    dataset_name: str,
    color_map: dict[str, str],
) -> str:
    ok = [r for r in results if not r.error]
    errors = [f"{r.run_name}: {r.error}" for r in results if r.error]
    if not ok:
        return f'<p class="notice">No successful runs for {html.escape(dataset_name)}.</p>'
    leaderboard = [_run_to_leaderboard_row(r) for r in ok]
    return render_radar_section_html(
        leaderboard,
        dataset_name=dataset_name,
        errors=errors or None,
        color_map=color_map,
    )


def _report_index_nav(dataset_names: list[str]) -> str:
    links = ['<a href="#global">Global (all datasets)</a>']
    for name in dataset_names:
        sid = _section_id(name)
        links.append(f'<a href="#dataset-{html.escape(sid)}">{html.escape(name)}</a>')
    return f"""
    <nav class="report-index" aria-label="Report sections">
      <p class="index-label">Sections</p>
      {"".join(links)}
    </nav>
    """


def _global_section(
    benchmark: BenchmarkResult,
    *,
    dataset_names: list[str],
    color_map: dict[str, str],
    lat_title: str,
) -> str:
    global_results = _aggregate_global_results(benchmark.results)
    total_examples = sum(r.summary.n_examples for r in global_results if not r.error)
    table = _leaderboard_table_html(
        global_results,
        benchmark,
        show_dataset=False,
        lat_title=lat_title,
    )
    radar_inner = _radar_for_results(
        global_results,
        dataset_name="Global (all datasets)",
        color_map=color_map,
    )
    return f"""
    <section class="dataset-section" id="global">
      <h2>Global — all datasets</h2>
      <p class="section-sub">Aggregated Document/Strict F1 and latency across
      <strong>{total_examples}</strong> gold examples ({len(dataset_names)} dataset{"s" if len(dataset_names) != 1 else ""}).</p>
      <section class="block">
        <h3>Leaderboard</h3>
        <p class="notice">Ranked by Document-level string overlap F1. Green row = best overall.</p>
        {table}
      </section>
      {radar_inner}
      {_confusion_matrices_html(global_results)}
    </section>
    """


def _dataset_section(
    dataset_name: str,
    group: list[RunResult],
    benchmark: BenchmarkResult,
    *,
    color_map: dict[str, str],
    lat_title: str,
) -> str:
    sid = _section_id(dataset_name)
    ok = [r for r in group if not r.error]
    n_examples = ok[0].summary.n_examples if ok else 0
    n_gold = ok[0].summary.n_gold_spans if ok else 0
    table = _leaderboard_table_html(
        group,
        benchmark,
        show_dataset=False,
        lat_title=lat_title,
    )
    radar_inner = _radar_for_results(
        group,
        dataset_name=dataset_name,
        color_map=color_map,
    )
    return f"""
    <section class="dataset-section" id="dataset-{html.escape(sid)}">
      <h2>{html.escape(dataset_name)}</h2>
      <p class="section-sub"><strong>{n_examples}</strong> examples,
      <strong>{n_gold}</strong> gold spans in this dataset.</p>
      <section class="block">
        <h3>Leaderboard</h3>
        <p class="notice">Ranked by Document-level string overlap F1 on this dataset only. Green row = best for this corpus.</p>
        {table}
      </section>
      {radar_inner}
      {_confusion_matrices_html(group)}
    </section>
    """


def _curves_tab_html(curves: ThresholdCurvesResult | None) -> str | None:
    if curves is None or not curves.records:
        return None
    ok = [r for r in curves.records if not r.error]
    if not ok:
        return None
    intro = """
    <p class="notice">PR/ROC curves use one inference pass at <code>threshold=0</code> per backend×dataset.
    PR recall is micro-averaged over gold units; ROC is proposal-level (ranking correct vs incorrect candidates).</p>
    """
    return intro + render_curves_section_html(curves.to_dict())


def render_html_report(
    benchmark: BenchmarkResult,
    curves: ThresholdCurvesResult | None = None,
) -> str:
    config_path = html.escape(display_repo_path(benchmark.config_path))
    output_dir = html.escape(display_repo_path(benchmark.output_dir))

    run_color_map = build_run_color_map(_run_names_for_color_map(benchmark))
    by_dataset = _group_by_dataset(benchmark.results)
    dataset_names = _datasets_with_results(benchmark)

    repeats_notice = ""
    if benchmark.repeats > 1:
        repeats_notice = (
            f'<p class="notice">Repeats: <strong>{benchmark.repeats}</strong> per backend×dataset '
            "(cache cleared between repeat rounds). Latency = mean ± std ms/example in the table."
        )

    lat_title = (
        "Mean ± std ms/example across repeats"
        if benchmark.repeats > 1
        else "Wall-clock ms per example (inference; load time reported separately in metrics.json)"
    )

    section_blocks = [
        _global_section(
            benchmark,
            dataset_names=dataset_names,
            color_map=run_color_map,
            lat_title=lat_title,
        )
    ]
    for dataset_name in dataset_names:
        section_blocks.append(
            _dataset_section(
                dataset_name,
                by_dataset.get(dataset_name, []),
                benchmark,
                color_map=run_color_map,
                lat_title=lat_title,
            )
        )

    results_html = f"""
    {_partial_run_notice(benchmark)}
    {repeats_notice}
    {_report_index_nav(dataset_names)}
    {"".join(section_blocks)}
    {_repeat_stability_section(benchmark)}
    """

    methodology_html = render_ner_methodology_content(benchmark)
    curves_html = _curves_tab_html(curves)
    tabbed_body = render_report_tabs(
        tab_prefix="ner-bench",
        results_html=results_html,
        methodology_html=methodology_html,
        curves_html=curves_html,
    )
    has_curves = curves_html is not None
    theme_css = load_lab_theme_css()
    footer_note = "ner-detector · benchmark/run_benchmark.py"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NER backend benchmark — ner-detector</title>
  <style>
{theme_css}
    {REPORT_PAGE_EXTRA_CSS}
    {report_tab_styles("ner-bench", has_curves=has_curves)}
    {RADAR_CHART_CSS}
    {curve_chart_css() if has_curves else ""}
  </style>
</head>
<body>
  {site_header(active="projects")}
  <main class="main">
    <header class="page">
      <h1>NER backend benchmark</h1>
      <p class="sub">Config <code>{config_path}</code></p>
      <p class="sub">Output <code>{output_dir}</code></p>
      {"<p class='sub'>Repeats per cell: <strong>" + str(benchmark.repeats) + "</strong></p>" if benchmark.repeats > 1 else ""}
    </header>

    {tabbed_body}
  </main>
  {site_footer(note=footer_note)}
</body>
</html>
"""


def write_html_report(
    benchmark: BenchmarkResult,
    path: Path,
    *,
    curves: ThresholdCurvesResult | None = None,
) -> Path:
    content = render_html_report(benchmark, curves=curves)
    path.write_text(content, encoding="utf-8")
    if path.name != "index.html":
        index_path = path.parent / "index.html"
        index_path.write_text(content, encoding="utf-8")
    return path

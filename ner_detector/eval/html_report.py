"""HTML benchmark report — text-compressor style (light theme, static SVG radar)."""

from __future__ import annotations

import html
from pathlib import Path

from ner_detector.eval.radar_svg import (
    RADAR_CHART_CSS,
    build_run_color_map,
    render_radar_section_html,
)
from ner_detector.eval.runner import BenchmarkResult, RunResult, load_benchmark_config
from ner_detector.eval.repeat_stats import format_latency_mean_std
from ner_detector.eval.report_methodology import (
    render_ner_methodology_content,
    render_report_tabs,
    report_tab_styles,
)

REPORT_PAGE_CSS = """
    :root {
      --bg: #f4f5f7;
      --surface: #fff;
      --text: #14181c;
      --muted: #5f6b7a;
      --border: #dde2e8;
      --accent: #1d4ed8;
      --rank1: #ecfdf5;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }
    header.page {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem 1.75rem;
      margin-bottom: 1.25rem;
    }
    header.page h1 { margin: 0; font-size: 1.45rem; font-weight: 600; }
    header.page .sub { color: var(--muted); font-size: 0.92rem; margin: 0.35rem 0 0; }
    .notice {
      background: #fffbeb;
      border: 1px solid #fde68a;
      border-radius: 8px;
      padding: 0.65rem 0.9rem;
      font-size: 0.88rem;
      margin-bottom: 1rem;
    }
    section.block { margin-bottom: 1.5rem; }
    section.block h2 { font-size: 1.05rem; margin: 0 0 0.65rem; font-weight: 600; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      font-size: 0.88rem;
    }
    th, td {
      padding: 0.55rem 0.75rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }
    th {
      background: #f8fafc;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 600;
    }
    tr:last-child td { border-bottom: none; }
    tr.rank-1 { background: var(--rank1); }
    footer {
      margin-top: 1.5rem;
      text-align: center;
      font-size: 0.78rem;
      color: var(--muted);
    }
    th[title] { cursor: help; border-bottom: 1px dotted var(--border); }
    .report-tabs section.block { margin-bottom: 1.25rem; }
    .report-tabs section.block:last-child { margin-bottom: 0; }
    .report-tabs .notice { margin-bottom: 0.75rem; }
"""


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _speed_score(latency_ms: float, max_latency: float) -> float:
    if max_latency <= 0:
        return 1.0
    return max(0.0, 1.0 - (latency_ms / max_latency))


def _run_to_leaderboard_row(result: RunResult, *, max_latency: float) -> dict[str, object]:
    sp, sr, sf1 = result.summary.strict_prf()
    _rp, _rr, rf1 = result.summary.relaxed_prf()
    return {
        "run_name": result.run_name,
        "backend": result.backend,
        "model_id": result.model_id,
        "dataset": result.dataset,
        "strict_f1": sf1,
        "relaxed_f1": rf1,
        "precision": sp,
        "recall": sr,
        "speed": _speed_score(result.latency_ms_per_example, max_latency),
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
      <p class="notice">Each backend×dataset was run {benchmark.repeats} times with a fresh model cache.
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


def _leaderboard_table_rows(benchmark: BenchmarkResult) -> tuple[str, int]:
    """HTML table rows; return (html, best_rank_run_index for rank-1 styling)."""
    rows_data: list[tuple[RunResult, float]] = []
    for r in benchmark.results:
        if r.error:
            continue
        _p, _rec, f1 = r.summary.strict_prf()
        rows_data.append((r, f1))
    rows_data.sort(key=lambda x: (-x[1], x[0].run_name, x[0].dataset))
    best_f1 = rows_data[0][1] if rows_data else -1.0

    lines: list[str] = []
    rank = 0
    for r, f1 in rows_data:
        rank += 1
        rank_cls = "rank-1" if rank == 1 else ""
        _p, _rec, sf1 = r.summary.strict_prf()
        _p2, _r2, rf1 = r.summary.relaxed_prf()
        stable_mark = ""
        if benchmark.repeats > 1 and not r.scores_reproducible:
            stable_mark = ' <span class="err" title="F1 varied across repeats">⚠</span>'
        lines.append(
            f"<tr class='{rank_cls}'>"
            f"<td>{rank}</td>"
            f"<td><code>{html.escape(r.run_name)}</code></td>"
            f"<td>{html.escape(r.dataset)}</td>"
            f"<td>{html.escape(r.backend)}</td>"
            f"<td><code>{html.escape(r.model_id or '—')}</code></td>"
            f"<td>{_fmt_pct(sf1)}{stable_mark}</td>"
            f"<td>{_fmt_pct(rf1)}</td>"
            f"<td>{_latency_display(r)}</td>"
            f"</tr>"
        )

    for r in benchmark.results:
        if r.error:
            lines.append(
                "<tr>"
                f"<td>—</td>"
                f"<td><code>{html.escape(r.run_name)}</code></td>"
                f"<td>{html.escape(r.dataset)}</td>"
                f"<td>{html.escape(r.backend)}</td>"
                f"<td colspan='4' class='err'>ERROR: {html.escape(r.error or '')}</td>"
                f"</tr>"
            )

    return "".join(lines), int(best_f1 >= 0)


def render_html_report(benchmark: BenchmarkResult) -> str:
    config_path = html.escape(str(benchmark.config_path))
    output_dir = html.escape(str(benchmark.output_dir))
    table_rows, _ = _leaderboard_table_rows(benchmark)

    run_color_map = build_run_color_map(_run_names_for_color_map(benchmark))
    radar_sections: list[str] = []
    for dataset_name, group in sorted(_group_by_dataset(benchmark.results).items()):
        ok = [r for r in group if not r.error]
        errors = [f"{r.run_name}: {r.error}" for r in group if r.error]
        if not ok:
            radar_sections.append(
                f'<section class="block"><h2>{html.escape(dataset_name)}</h2>'
                f'<p class="notice">No successful runs for this dataset.</p></section>'
            )
            continue
        max_lat = max(r.latency_ms_per_example for r in ok)
        leaderboard = [_run_to_leaderboard_row(r, max_latency=max_lat) for r in ok]
        radar_sections.append(
            render_radar_section_html(
                leaderboard,
                dataset_name=dataset_name,
                errors=errors or None,
                color_map=run_color_map,
            )
        )

    radar_html = "\n".join(radar_sections)
    repeats_notice = ""
    if benchmark.repeats > 1:
        repeats_notice = (
            f'<p class="notice">Repeats: <strong>{benchmark.repeats}</strong> per backend×dataset '
            "(cache cleared each time). Latency = mean ± std ms/example; radar Speed uses the mean."
        )

    lat_title = (
        "Mean ± std ms/example across repeats"
        if benchmark.repeats > 1
        else "Wall-clock ms per example for this backend run"
    )

    results_html = f"""
    {repeats_notice}
    <section class="block">
      <h2>Leaderboard</h2>
      <p class="notice">Ranked by strict span F1 (exact start, end, and label). Green row = best overall.</p>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Run</th>
            <th>Dataset</th>
            <th>Backend</th>
            <th>Model</th>
            <th title="Exact span + unified label match">Strict F1</th>
            <th title="Same label, span IoU ≥ 0.5">Relaxed F1</th>
            <th title="{html.escape(lat_title)}">Latency</th>
          </tr>
        </thead>
        <tbody>{table_rows or "<tr><td colspan='8'>No results</td></tr>"}</tbody>
      </table>
    </section>
    {_repeat_stability_section(benchmark)}
    {radar_html}
    """

    methodology_html = render_ner_methodology_content(benchmark)
    tabbed_body = render_report_tabs(
        tab_prefix="ner-bench",
        results_html=results_html,
        methodology_html=methodology_html,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NER backend benchmark — ner-detector</title>
  <style>
    {REPORT_PAGE_CSS}
    {report_tab_styles("ner-bench")}
    {RADAR_CHART_CSS}
    .err {{ color: #b91c1c; font-size: 0.84rem; }}
  </style>
</head>
<body>
  <div class="wrap">
    <header class="page">
      <h1>NER backend benchmark</h1>
      <p class="sub">Config <code>{config_path}</code></p>
      <p class="sub">Output <code>{output_dir}</code></p>
      {"<p class='sub'>Repeats per cell: <strong>" + str(benchmark.repeats) + "</strong></p>" if benchmark.repeats > 1 else ""}
    </header>

    {tabbed_body}

    <footer>ner-detector · benchmark/run_benchmark.py</footer>
  </div>
</body>
</html>
"""


def write_html_report(benchmark: BenchmarkResult, path: Path) -> Path:
    path.write_text(render_html_report(benchmark), encoding="utf-8")
    return path

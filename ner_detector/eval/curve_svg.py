"""SVG line charts for PR and ROC curves."""

from __future__ import annotations

import html
from typing import Any

from ner_detector.eval.radar_svg import build_run_color_map, color_for_run

_CURVE_CSS = """
    .curve-chart-wrap {
      margin: 1rem 0;
      padding: 0.75rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
    }
    .curve-chart-wrap h3 {
      margin: 0 0 0.5rem;
      font-size: 0.95rem;
      font-weight: 600;
    }
    .curve-chart-wrap .sub {
      margin: 0 0 0.65rem;
      font-size: 0.8rem;
      color: var(--muted);
    }
    .curve-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }
    @media (max-width: 720px) {
      .curve-grid { grid-template-columns: 1fr; }
    }
    .curve-panel {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.5rem;
    }
    .curve-panel h4 {
      margin: 0 0 0.35rem;
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .curve-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem 0.75rem;
      margin-top: 0.5rem;
      font-size: 0.75rem;
    }
    .curve-legend span::before {
      content: "";
      display: inline-block;
      width: 0.65rem;
      height: 0.15rem;
      margin-right: 0.25rem;
      vertical-align: middle;
      background: currentColor;
    }
"""


def curve_chart_css() -> str:
    return _CURVE_CSS


def _plot_size() -> tuple[int, int, int, int]:
    """Return width, height, pad, plot_side."""
    w, h, pad = 320, 260, 36
    return w, h, pad, min(w, h) - 2 * pad


def _to_xy(
    values: list[tuple[float, float]],
    *,
    pad: int,
    side: int,
) -> str:
    if not values:
        return ""
    parts: list[str] = []
    for x, y in values:
        px = pad + x * side
        py = pad + (1.0 - y) * side
        parts.append(f"{px:.1f},{py:.1f}")
    return " ".join(parts)


def _axis_svg(*, title: str, x_label: str, y_label: str) -> str:
    w, h, pad, side = _plot_size()
    x1, y1 = pad, pad
    x2, y2 = pad + side, pad + side
    return f"""
    <svg class="curve-svg" viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(title)}">
      <rect x="{x1}" y="{y1}" width="{side}" height="{side}" fill="#fff" stroke="#dde2e8"/>
      <line x1="{x1}" y1="{y2}" x2="{x2}" y2="{y2}" stroke="#94a3b8"/>
      <line x1="{x1}" y1="{y1}" x2="{x1}" y2="{y2}" stroke="#94a3b8"/>
      <text x="{(x1 + x2) / 2}" y="{h - 8}" text-anchor="middle" font-size="10" fill="#5f6b7a">{html.escape(x_label)}</text>
      <text x="10" y="{(y1 + y2) / 2}" text-anchor="middle" font-size="10" fill="#5f6b7a" transform="rotate(-90 10 {(y1 + y2) / 2})">{html.escape(y_label)}</text>
      <text x="{x1 + 4}" y="{y1 + 10}" font-size="9" fill="#94a3b8">1</text>
      <text x="{x1 + 4}" y="{y2 - 2}" font-size="9" fill="#94a3b8">0</text>
      <text x="{x2 - 4}" y="{y2 + 12}" text-anchor="end" font-size="9" fill="#94a3b8">1</text>
    """


def render_curve_panel(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]], str]],
) -> str:
    """Render one chart with multiple polylines (name, points, color)."""
    w, h, pad, side = _plot_size()
    base = _axis_svg(title=title, x_label=x_label, y_label=y_label)
    lines: list[str] = []
    for _name, pts, color in series:
        if len(pts) < 2:
            continue
        path = _to_xy(pts, pad=pad, side=side)
        lines.append(
            f'<polyline fill="none" stroke="{html.escape(color)}" stroke-width="2" points="{path}"/>',
        )
    return base + "\n".join(lines) + "\n</svg>"


def _series_from_pr(
    runs: list[dict[str, Any]],
    *,
    color_map: dict[str, str],
) -> list[tuple[str, list[tuple[float, float]], str]]:
    out: list[tuple[str, list[tuple[float, float]], str]] = []
    for run in runs:
        pts = run.get("pr") or []
        xy = [(float(p["recall"]), float(p["precision"])) for p in pts]
        if len(xy) >= 2:
            name = str(run.get("run_name", "run"))
            out.append((name, xy, color_for_run(name, color_map)))
    return out


def _series_from_roc(
    runs: list[dict[str, Any]],
    *,
    color_map: dict[str, str],
) -> list[tuple[str, list[tuple[float, float]], str]]:
    out: list[tuple[str, list[tuple[float, float]], str]] = []
    for run in runs:
        pts = run.get("roc") or []
        xy = [
            (float(p["fpr"]), float(p["tpr"]))
            for p in pts
            if p.get("fpr") is not None and p.get("tpr") is not None
        ]
        if len(xy) >= 2:
            name = str(run.get("run_name", "run"))
            out.append((name, xy, color_for_run(name, color_map)))
    return out


def render_dataset_curves_html(
    dataset: str,
    runs: list[dict[str, Any]],
    *,
    mode: str,
    color_map: dict[str, str] | None = None,
) -> str:
    """PR + ROC panels for one dataset and match mode."""
    cmap = color_map or build_run_color_map([str(r.get("run_name", "")) for r in runs])
    pr_series = _series_from_pr(runs, color_map=cmap)
    roc_series = _series_from_roc(runs, color_map=cmap)
    mode_label = html.escape(mode)
    ds_label = html.escape(dataset)

    legend = "".join(
        f'<span style="color:{html.escape(color_for_run(str(r.get("run_name", "")), cmap))}">'
        f"{html.escape(str(r.get('run_name', '')))} "
        f"(AUC-PR {float(r.get('auc_pr', 0)):.3f})</span>"
        for r in runs
        if r.get("pr")
    )

    pr_svg = render_curve_panel(
        title=f"PR — {dataset} ({mode})",
        x_label="Recall",
        y_label="Precision",
        series=pr_series,
    )
    roc_svg = render_curve_panel(
        title=f"ROC — {dataset} ({mode})",
        x_label="FPR (among proposals)",
        y_label="TPR (among proposals)",
        series=roc_series,
    )

    return f"""
    <div class="curve-chart-wrap">
      <h3>{ds_label} · {mode_label}</h3>
      <p class="sub">Inference at threshold 0; curves sweep candidate scores high → low.</p>
      <div class="curve-grid">
        <div class="curve-panel"><h4>Precision–Recall</h4>{pr_svg}</div>
        <div class="curve-panel"><h4>ROC (proposal-level)</h4>{roc_svg}</div>
      </div>
      <div class="curve-legend">{legend}</div>
    </div>
    """


def render_curves_section_html(curves_payload: dict[str, Any]) -> str:
    """Full HTML section for threshold curve tab."""
    datasets = curves_payload.get("datasets") or {}
    if not datasets:
        return '<p class="notice">No threshold curves (no threshold backends or empty run).</p>'

    run_names = [str(r.get("run_name", "")) for ds in datasets.values() for r in ds]
    color_map = build_run_color_map(run_names)
    blocks: list[str] = []
    for dataset_name in sorted(datasets.keys()):
        runs = datasets[dataset_name]
        if not runs:
            continue
        for mode in ("document", "strict", "relaxed"):
            mode_runs = [
                {
                    "run_name": r["run_name"],
                    "pr": (r.get("modes") or {}).get(mode, {}).get("pr"),
                    "roc": (r.get("modes") or {}).get(mode, {}).get("roc"),
                    "auc_pr": (r.get("modes") or {}).get(mode, {}).get("auc_pr"),
                }
                for r in runs
            ]
            if any(m.get("pr") for m in mode_runs):
                blocks.append(
                    render_dataset_curves_html(
                        dataset_name,
                        mode_runs,
                        mode=mode,
                        color_map=color_map,
                    ),
                )
    return "\n".join(blocks) if blocks else '<p class="notice">No curve data.</p>'

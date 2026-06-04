"""Static SVG radar charts (text-compressor style)."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any, Callable

_MODEL_COLORS = (
    "#1d4ed8",
    "#c2410c",
    "#047857",
    "#7c3aed",
    "#b45309",
    "#0e7490",
    "#be123c",
    "#4d7c0f",
)


def build_run_color_map(run_names: list[str]) -> dict[str, str]:
    """Stable run_name → color mapping (same backend, same color on every chart)."""
    ordered: list[str] = []
    seen: set[str] = set()
    for name in run_names:
        key = str(name).strip()
        if key and key not in seen:
            ordered.append(key)
            seen.add(key)
    return {
        name: _MODEL_COLORS[i % len(_MODEL_COLORS)] for i, name in enumerate(ordered)
    }


def color_for_run(run_name: str, color_map: dict[str, str] | None) -> str:
    if color_map and run_name in color_map:
        return color_map[run_name]
    return _MODEL_COLORS[hash(run_name) % len(_MODEL_COLORS)]


@dataclass(frozen=True)
class RadarAxis:
    key: str
    label: str
    extract: Callable[[dict[str, Any]], float | None]
    higher_is_better: bool = True


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


NER_RADAR_AXES: tuple[RadarAxis, ...] = (
    RadarAxis("document_f1", "Doc F1", lambda r: _float_or_none(r.get("document_f1"))),
    RadarAxis("strict_f1", "Strict F1", lambda r: _float_or_none(r.get("strict_f1"))),
    RadarAxis("speed", "Speed", lambda r: _float_or_none(r.get("speed"))),
)


def absolute_clamp(value: float | None, *, higher_is_better: bool = True) -> float:
    if value is None:
        return 0.0
    v = float(value)
    if not higher_is_better:
        v = 1.0 - v
    return max(0.0, min(1.0, v))


def radar_polygon_area(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    angle = 2.0 * math.pi / n
    sin_a = math.sin(angle)
    total = sum(values[i] * values[(i + 1) % n] for i in range(n))
    return 0.5 * sin_a * total


def radar_vertex_points(
    values: list[float],
    *,
    cx: float,
    cy: float,
    radius: float,
) -> list[tuple[float, float]]:
    n = len(values)
    if n == 0:
        return []
    start = -math.pi / 2
    step = 2.0 * math.pi / n
    points: list[tuple[float, float]] = []
    for i, r in enumerate(values):
        theta = start + i * step
        dist = max(0.0, min(1.0, r)) * radius
        points.append((cx + dist * math.cos(theta), cy + dist * math.sin(theta)))
    return points


def _points_attr(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def build_radar_series(
    leaderboard: list[dict[str, Any]],
    axes: tuple[RadarAxis, ...] = NER_RADAR_AXES,
) -> list[dict[str, Any]]:
    if not leaderboard:
        return []
    raw_by_axis = [[axis.extract(row) for row in leaderboard] for axis in axes]
    norm_matrix = [
        [absolute_clamp(v, higher_is_better=axis.higher_is_better) for v in raw_col]
        for axis, raw_col in zip(axes, raw_by_axis, strict=True)
    ]
    series: list[dict[str, Any]] = []
    for i, row in enumerate(leaderboard):
        radii = [norm_matrix[j][i] for j in range(len(axes))]
        series.append(
            {
                "run_name": str(row.get("run_name", f"run-{i}")),
                "radii": radii,
                "area": radar_polygon_area(radii),
            }
        )
    return series


def render_radar_section_html(
    leaderboard: list[dict[str, Any]],
    *,
    dataset_name: str,
    errors: list[str] | None = None,
    axes: tuple[RadarAxis, ...] = NER_RADAR_AXES,
    color_map: dict[str, str] | None = None,
) -> str:
    """One radar block: chart + legend (text-compressor layout)."""
    if not leaderboard:
        return ""

    series = build_radar_series(leaderboard, axes=axes)
    if not series:
        return ""

    size = 420
    cx = cy = size / 2
    plot_r = 150
    label_r = plot_r + 28

    grid_lines = []
    for level in (0.25, 0.5, 0.75, 1.0):
        ring = radar_vertex_points([level] * len(axes), cx=cx, cy=cy, radius=plot_r)
        grid_lines.append(f'<polygon points="{_points_attr(ring)}" class="radar-grid" />')

    spokes: list[str] = []
    axis_labels: list[str] = []
    n = len(axes)
    start = -math.pi / 2
    step = 2.0 * math.pi / n
    for i, axis in enumerate(axes):
        theta = start + i * step
        x2 = cx + plot_r * math.cos(theta)
        y2 = cy + plot_r * math.sin(theta)
        spokes.append(
            f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" class="radar-spoke" />'
        )
        lx = cx + label_r * math.cos(theta)
        ly = cy + label_r * math.sin(theta)
        anchor = "middle"
        if math.cos(theta) > 0.35:
            anchor = "start"
        elif math.cos(theta) < -0.35:
            anchor = "end"
        axis_labels.append(
            f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="{anchor}" '
            f'dominant-baseline="middle" class="radar-axis-label">'
            f"{html.escape(axis.label)}</text>"
        )

    polygons: list[str] = []
    legend_items: list[str] = []
    for item in sorted(series, key=lambda s: (-s["area"], str(s["run_name"]))):
        color = color_for_run(str(item["run_name"]), color_map)
        pts = radar_vertex_points(item["radii"], cx=cx, cy=cy, radius=plot_r)
        polygons.append(
            f'<polygon points="{_points_attr(pts)}" class="radar-model" '
            f'fill="{color}" stroke="{color}" />'
        )
        legend_items.append(
            f'<li><span class="radar-swatch" style="background:{color}"></span>'
            f"<code>{html.escape(item['run_name'])}</code> — area "
            f"<strong>{item['area']:.3f}</strong></li>"
        )

    axis_list = "".join(f"<li>{html.escape(a.label)}</li>" for a in axes)
    err_block = ""
    if errors:
        err_block = (
            '<p class="notice">'
            + "Skipped: "
            + "; ".join(html.escape(e) for e in errors)
            + "</p>"
        )

    title = html.escape(dataset_name)
    return f"""
    <section class="block radar-section">
      <h2>Backend radar — {title}</h2>
      <p class="radar-note">Quality axes (F1, precision, recall) are 0–1 (higher is better).
      Speed = 1 − (ms/example ÷ 1000), clamped 0–1 — so 1 s/example scores 0 on Speed.
      Absolute latency is also in the leaderboard table. Polygon area is a quick composite index —
      not a ranking metric.</p>
      {err_block}
      <div class="radar-layout">
        <figure class="radar-figure">
          <svg class="radar-chart" viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img"
               aria-label="Radar chart for {title}">
            {''.join(grid_lines)}
            {''.join(spokes)}
            {''.join(polygons)}
            {''.join(axis_labels)}
          </svg>
        </figure>
        <div class="radar-side">
          <h3 class="radar-subhead">Legend (by area)</h3>
          <ol class="radar-legend">{''.join(legend_items)}</ol>
          <h3 class="radar-subhead">Axes</h3>
          <ul class="radar-axes-list">{axis_list}</ul>
        </div>
      </div>
    </section>
    """


RADAR_CHART_CSS = """
    .radar-section { margin-bottom: 1.5rem; }
    .radar-note {
      font-size: 0.84rem; color: var(--muted); margin: 0 0 0.85rem; line-height: 1.45;
    }
    .radar-layout {
      display: flex; flex-wrap: wrap; gap: 1.25rem; align-items: flex-start;
      background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
      padding: 1rem 1.25rem;
    }
    .radar-figure { margin: 0; flex: 0 0 auto; }
    .radar-chart { display: block; max-width: 100%; height: auto; }
    .radar-grid { fill: none; stroke: var(--border); stroke-width: 1; }
    .radar-spoke { stroke: var(--border); stroke-width: 1; }
    .radar-axis-label {
      font-size: 11px; fill: var(--muted); font-family: "Segoe UI", system-ui, sans-serif;
    }
    .radar-model { fill-opacity: 0.12; stroke-width: 2; stroke-linejoin: round; }
    .radar-side { flex: 1 1 220px; min-width: 200px; }
    .radar-subhead {
      margin: 0 0 0.4rem; font-size: 0.78rem; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted);
    }
    .radar-legend { margin: 0 0 1rem; padding-left: 1.25rem; font-size: 0.84rem; line-height: 1.55; }
    .radar-legend li { margin: 0.2rem 0; }
    .radar-swatch {
      display: inline-block; width: 10px; height: 10px; border-radius: 2px;
      margin-right: 0.35rem; vertical-align: middle;
    }
    .radar-axes-list { margin: 0; padding-left: 1.15rem; font-size: 0.82rem; color: var(--muted); }
"""

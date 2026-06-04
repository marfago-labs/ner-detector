"""Tests for radar SVG helpers."""

from __future__ import annotations

import math

from ner_detector.eval.radar_svg import (
    NER_RADAR_AXES,
    absolute_clamp,
    build_radar_series,
    build_run_color_map,
    color_for_run,
    radar_polygon_area,
    radar_vertex_points,
    render_radar_section_html,
)


def test_absolute_clamp_none_and_inverted() -> None:
    assert absolute_clamp(None) == 0.0
    assert absolute_clamp(0.2, higher_is_better=False) == 0.8


def test_radar_polygon_area_small_n() -> None:
    assert radar_polygon_area([0.5, 0.5]) == 0.0


def test_radar_vertex_points_empty() -> None:
    assert radar_vertex_points([], cx=0, cy=0, radius=10) == []


def test_build_radar_series_empty_leaderboard() -> None:
    assert build_radar_series([]) == []


def test_render_radar_empty_inputs() -> None:
    assert render_radar_section_html([], dataset_name="empty") == ""


def test_color_for_run_fallback_hash() -> None:
    color = color_for_run("unknown-run", None)
    assert color.startswith("#")


def test_build_run_color_map_skips_blanks() -> None:
    cmap = build_run_color_map(["", "a", "a", "  "])
    assert list(cmap.keys()) == ["a"]


def test_render_radar_with_errors_block() -> None:
    rows = [
        {
            "run_name": "x",
            "strict_f1": 0.5,
            "relaxed_f1": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "speed": 0.5,
        }
    ]
    html_out = render_radar_section_html(
        rows,
        dataset_name="ds",
        errors=["run failed"],
    )
    assert "Skipped:" in html_out
    assert "run failed" in html_out


def test_ner_radar_axes_include_speed() -> None:
    keys = [axis.key for axis in NER_RADAR_AXES]
    assert keys == ["document_f1", "strict_f1", "speed"]
    row = {"strict_f1": None}
    for axis in NER_RADAR_AXES:
        assert axis.extract(row) is None or isinstance(axis.extract(row), float)


def test_radar_vertex_clamps_radius() -> None:
    pts = radar_vertex_points([2.0], cx=100, cy=100, radius=50)
    assert len(pts) == 1
    dist = math.hypot(pts[0][0] - 100, pts[0][1] - 100)
    assert dist == 50.0

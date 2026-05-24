"""Tests for repeat-run latency aggregation."""

from __future__ import annotations

from ner_detector.eval.repeat_stats import compute_latency_stats, format_latency_mean_std


def test_compute_latency_stats_single() -> None:
    st = compute_latency_stats([12.5])
    assert st.mean == 12.5
    assert st.std == 0.0


def test_compute_latency_stats_multiple() -> None:
    st = compute_latency_stats([10.0, 20.0, 30.0])
    assert st.mean == 20.0
    assert st.min == 10.0
    assert st.max == 30.0
    assert st.median == 20.0
    assert st.std > 0


def test_format_latency_mean_std() -> None:
    assert format_latency_mean_std(5.0, 0.0, n_repeats=1) == "5.00"
    assert "±" in format_latency_mean_std(10.0, 2.5, n_repeats=3)


def test_compute_latency_stats_empty() -> None:
    st = compute_latency_stats([])
    assert st.mean == 0.0
    assert st.samples == []

"""Latency statistics across repeated benchmark trials."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass
class LatencyStats:
    """Wall-clock ms/example across repeated benchmark trials."""

    mean: float
    std: float
    min: float
    max: float
    median: float
    samples: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "median": round(self.median, 2),
            "samples_ms_per_example": [round(v, 2) for v in self.samples],
        }


def compute_latency_stats(samples: list[float]) -> LatencyStats:
    if not samples:
        return LatencyStats(0.0, 0.0, 0.0, 0.0, 0.0, [])
    if len(samples) == 1:
        v = samples[0]
        return LatencyStats(v, 0.0, v, v, v, list(samples))
    return LatencyStats(
        mean=statistics.mean(samples),
        std=statistics.stdev(samples),
        min=min(samples),
        max=max(samples),
        median=statistics.median(samples),
        samples=list(samples),
    )


def format_latency_mean_std(mean: float, std: float, *, n_repeats: int) -> str:
    if n_repeats <= 1:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"

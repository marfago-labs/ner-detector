"""Types for benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class GoldEntity(BaseModel):
    text: str
    label: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class GoldExample(BaseModel):
    id: str
    text: str
    entities: list[GoldEntity]


@dataclass(frozen=True, slots=True)
class EvalSpan:
    """Normalized span for scoring."""

    start: int
    end: int
    label: str
    text: str

    def as_tuple(self) -> tuple[int, int, str]:
        return (self.start, self.end, self.label)


@dataclass(frozen=True, slots=True)
class BackendRunSpec:
    name: str
    backend: str
    model_id: str | None = None
    labels: list[str] | None = None
    threshold: float = 0.5


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    runs: list[BackendRunSpec]
    datasets: list[str]
    label_map: str = "unified"
    benchmark_root: str | None = None
    repeats: int = 5

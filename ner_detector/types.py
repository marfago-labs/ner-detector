"""Shared types for NER detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NerBackend = Literal["pattern", "transformers", "gliner", "nuner", "generative_ner", "llm"]


@dataclass(frozen=True, slots=True)
class DetectedEntity:
    """A named span extracted from text."""

    text: str
    label: str
    score: float | None = None
    start: int | None = None
    end: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "label": self.label,
            "score": self.score,
            "start": self.start,
            "end": self.end,
        }

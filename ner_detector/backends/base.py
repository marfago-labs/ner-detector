"""Abstract NER backend."""

from __future__ import annotations

from typing import Protocol

from ner_detector.types import DetectedEntity


class NerDetectorBackend(Protocol):
    """Extract entities from raw text."""

    backend: str
    model_id: str

    def detect(
        self,
        text: str,
        *,
        labels: list[str] | None = None,
        threshold: float = 0.5,
    ) -> list[DetectedEntity]: ...

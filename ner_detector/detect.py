"""High-level NER detection API."""

from __future__ import annotations

from ner_detector.registry import create_backend
from ner_detector.types import DetectedEntity, NerBackend


def detect_entities(
    text: str,
    *,
    backend: NerBackend = "pattern",
    model_id: str | None = None,
    labels: list[str] | None = None,
    threshold: float = 0.5,
) -> list[DetectedEntity]:
    """Run NER on ``text`` using the selected backend."""
    detector = create_backend(backend, model_id=model_id)
    return detector.detect(text, labels=labels, threshold=threshold)

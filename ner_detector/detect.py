"""High-level NER detection API."""

from __future__ import annotations

from ner_detector.registry import BackendOptions, create_backend
from ner_detector.types import DetectedEntity, NerBackend


def detect_entities(
    text: str,
    *,
    backend: NerBackend = "pattern",
    model_id: str | None = None,
    labels: list[str] | None = None,
    threshold: float = 0.5,
    provider: str | None = None,
    temperature: float | None = None,
    max_chars: int | None = None,
    max_chunk_chars: int | None = None,
    label_definitions: dict[str, str] | None = None,
    few_shot_examples: list[dict[str, object]] | None = None,
) -> list[DetectedEntity]:
    """Run NER on ``text`` using the selected backend."""
    detector = create_backend(
        backend,
        model_id=model_id,
        options=BackendOptions(
            provider=provider,
            temperature=temperature,
            max_chars=max_chars,
            max_chunk_chars=max_chunk_chars,
            label_definitions=label_definitions,
            few_shot_examples=few_shot_examples,
        ),
    )
    return detector.detect(text, labels=labels, threshold=threshold)

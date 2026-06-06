"""Backend family taxonomy (deterministic, ML, LLM)."""

from __future__ import annotations

from ner_detector.types import NerBackend

BACKEND_FAMILIES: dict[NerBackend, str] = {
    "pattern": "deterministic",
    "transformers": "ml",
    "gliner": "ml",
    "nuner": "ml",
    "generative_ner": "ml",
    "llm": "llm",
}

THRESHOLD_BACKENDS: frozenset[NerBackend] = frozenset(
    {"transformers", "gliner", "nuner"},
)


def backend_family(backend: str) -> str:
    """Return ``deterministic``, ``ml``, or ``llm`` for a backend name."""
    return BACKEND_FAMILIES.get(backend, "unknown")  # type: ignore[arg-type]


def uses_score_threshold(backend: str) -> bool:
    return backend in THRESHOLD_BACKENDS

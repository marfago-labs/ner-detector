"""Backend factory and caching."""

from __future__ import annotations

from ner_detector.backends.base import NerDetectorBackend
from ner_detector.backends.pattern import PatternBackend
from ner_detector.config import default_model_id
from ner_detector.types import NerBackend

_CACHE: dict[str, NerDetectorBackend] = {}


def create_backend(
    backend: NerBackend,
    *,
    model_id: str | None = None,
) -> NerDetectorBackend:
    """Return a cached or new backend instance."""
    resolved_model = (model_id or default_model_id(backend)).strip()
    cache_key = f"{backend}|{resolved_model}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if backend == "pattern":
        instance: NerDetectorBackend = PatternBackend()
    elif backend == "transformers":
        from ner_detector.backends.transformers_backend import TransformersBackend

        instance = TransformersBackend(resolved_model)
    elif backend == "gliner":
        from ner_detector.backends.gliner_backend import GlinerBackend

        instance = GlinerBackend(resolved_model)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    _CACHE[cache_key] = instance
    return instance


def clear_backend_cache() -> None:
    """Drop cached backends (for tests)."""
    _CACHE.clear()

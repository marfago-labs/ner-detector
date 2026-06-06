"""Backend factory and caching."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ner_detector.backends.base import NerDetectorBackend
from ner_detector.backends.pattern import PatternBackend
from ner_detector.config import default_model_id
from ner_detector.types import NerBackend

_CACHE: dict[str, NerDetectorBackend] = {}


@dataclass(frozen=True, slots=True)
class BackendOptions:
    """Optional construction parameters (LLM provider, chunk size, …)."""

    provider: str | None = None
    temperature: float | None = None
    max_chars: int | None = None
    max_chunk_chars: int | None = None
    label_definitions: dict[str, str] | None = None
    few_shot_examples: list[dict[str, Any]] | None = None


def _cache_key(
    backend: NerBackend,
    model_id: str,
    options: BackendOptions,
) -> str:
    parts = [backend, model_id]
    if backend == "llm":
        parts.extend(
            [
                options.provider or "openrouter",
                str(options.temperature if options.temperature is not None else 0.0),
                str(options.max_chars if options.max_chars is not None else 8000),
                json.dumps(options.label_definitions or {}, sort_keys=True),
                json.dumps(options.few_shot_examples or [], sort_keys=True),
            ],
        )
    if backend == "generative_ner":
        parts.append(str(options.max_chunk_chars if options.max_chunk_chars is not None else 4000))
    if backend == "transformers":
        parts.append(str(options.max_chunk_chars if options.max_chunk_chars is not None else 4000))
    return "|".join(parts)


def _build_pattern(_model_id: str, _options: BackendOptions) -> NerDetectorBackend:
    return PatternBackend()


def _build_transformers(model_id: str, options: BackendOptions) -> NerDetectorBackend:
    from ner_detector.backends.transformers_backend import TransformersBackend

    return TransformersBackend(
        model_id,
        max_chunk_chars=options.max_chunk_chars or 4000,
    )


def _build_gliner(model_id: str, _options: BackendOptions) -> NerDetectorBackend:
    from ner_detector.backends.gliner_backend import GlinerBackend

    return GlinerBackend(model_id)


def _build_nuner(model_id: str, _options: BackendOptions) -> NerDetectorBackend:
    from ner_detector.backends.nuner_backend import NunerBackend

    return NunerBackend(model_id)


def _build_generative_ner(model_id: str, options: BackendOptions) -> NerDetectorBackend:
    from ner_detector.backends.generative_ner_backend import GenerativeNerBackend

    return GenerativeNerBackend(
        model_id,
        max_chunk_chars=options.max_chunk_chars or 4000,
    )


def _build_llm(model_id: str, options: BackendOptions) -> NerDetectorBackend:
    from ner_detector.backends.llm_backend import LlmBackend

    return LlmBackend(
        model_id,
        provider=options.provider or "openrouter",
        temperature=options.temperature if options.temperature is not None else 0.0,
        max_chars=options.max_chars if options.max_chars is not None else 8000,
        label_definitions=options.label_definitions,
        few_shot_examples=options.few_shot_examples,
    )


_BUILDERS: dict[NerBackend, Callable[[str, BackendOptions], NerDetectorBackend]] = {
    "pattern": _build_pattern,
    "transformers": _build_transformers,
    "gliner": _build_gliner,
    "nuner": _build_nuner,
    "generative_ner": _build_generative_ner,
    "llm": _build_llm,
}


def create_backend(
    backend: NerBackend,
    *,
    model_id: str | None = None,
    options: BackendOptions | None = None,
) -> NerDetectorBackend:
    """Return a cached or new backend instance."""
    opts = options or BackendOptions()
    resolved_model = (model_id or default_model_id(backend)).strip()
    if backend != "pattern" and not resolved_model:
        raise ValueError(f"model_id is required for backend {backend!r}")

    cache_key = _cache_key(backend, resolved_model, opts)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    builder = _BUILDERS.get(backend)
    if builder is None:
        raise ValueError(f"Unknown backend: {backend!r}")

    instance = builder(resolved_model, opts)
    _CACHE[cache_key] = instance
    return instance


def clear_backend_cache() -> None:
    """Drop cached backends (for tests)."""
    _CACHE.clear()

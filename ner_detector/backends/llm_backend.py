"""LLM NER backend (OpenRouter, mock)."""

from __future__ import annotations

import json
from typing import Any

from ner_detector.backends.chunking import (
    _DEFAULT_CHUNK_OVERLAP,
    chunk_offset,
    chunk_text,
    merge_overlapping_entities,
    shift_entities,
)
from ner_detector.backends.llm.client import OpenRouterChatClient, create_llm_client
from ner_detector.backends.llm.parse import default_llm_labels, parse_entities_response
from ner_detector.types import DetectedEntity

_DEFAULT_MAX_CHARS = 8000
_DEFAULT_CHUNK_ATTEMPTS = 3


class LlmBackend:
    """Zero-shot NER via chat completion + structured JSON."""

    backend = "llm"

    def __init__(
        self,
        model_id: str,
        *,
        provider: str = "openrouter",
        temperature: float = 0.0,
        max_chars: int = _DEFAULT_MAX_CHARS,
        label_definitions: dict[str, str] | None = None,
        few_shot_examples: list[dict[str, Any]] | None = None,
    ) -> None:
        self.model_id = model_id
        self.provider = provider.strip().lower()
        self.temperature = temperature
        self.max_chars = max_chars
        self.label_definitions = label_definitions
        self.few_shot_examples = few_shot_examples
        self._client = create_llm_client(self.provider)

    def detect(
        self,
        text: str,
        *,
        labels: list[str] | None = None,
        threshold: float = 0.5,
    ) -> list[DetectedEntity]:
        del threshold  # LLM uses full model output; threshold reserved for API parity
        if not text.strip():
            return []
        entity_labels = labels if labels else default_llm_labels()
        allowed = {label.strip().lower() for label in entity_labels if label.strip()}
        chunks = chunk_text(
            text,
            max_chars=self.max_chars,
            overlap=_DEFAULT_CHUNK_OVERLAP,
        )
        found: list[DetectedEntity] = []
        for chunk in chunks:
            offset = chunk_offset(text, chunk)
            chunk_entities = self._detect_chunk(chunk, labels=entity_labels, allowed=allowed)
            found.extend(shift_entities(chunk_entities, offset=offset, source=text))
        return merge_overlapping_entities(found, source=text)

    def _detect_chunk(
        self,
        chunk: str,
        *,
        labels: list[str],
        allowed: set[str],
    ) -> list[DetectedEntity]:
        for attempt in range(_DEFAULT_CHUNK_ATTEMPTS):
            raw = self._complete_chunk(chunk, labels=labels)
            entities = parse_entities_response(raw, chunk, allowed_labels=allowed)
            if entities:
                return entities
            if not _response_has_entity_items(raw):
                return entities
            if attempt + 1 >= _DEFAULT_CHUNK_ATTEMPTS:
                break
        return []

    def _complete_chunk(self, chunk: str, *, labels: list[str]) -> str:
        kwargs = {
            "labels": labels,
            "model_id": self.model_id,
            "label_definitions": self.label_definitions,
            "few_shot_examples": self.few_shot_examples,
        }
        if isinstance(self._client, OpenRouterChatClient):
            return self._client.complete_json(
                chunk,
                temperature=self.temperature,
                **kwargs,
            )
        return self._client.complete_json(chunk, **kwargs)


def _response_has_entity_items(raw: str) -> bool:
    """True when JSON declares a non-empty entities array (may still be unusable)."""
    try:
        payload = json.loads(raw.strip())
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    items = payload.get("entities")
    return isinstance(items, list) and len(items) > 0

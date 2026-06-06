"""Generative instruction-tuned NER (UniversalNER / GoLLIE-style)."""

from __future__ import annotations

import json
import re

from ner_detector.backends.chunking import (
    _DEFAULT_CHUNK_OVERLAP,
    chunk_offset,
    chunk_text,
    merge_overlapping_entities,
    shift_entities,
)
from ner_detector.types import DetectedEntity

_DEFAULT_LABELS = ["person", "organization", "location", "date"]
_DEFAULT_MAX_CHUNK_CHARS = 4000
_UNINER_PROMPT = (
    "A virtual assistant answers questions from a user based on the provided text.\n"
    "USER: Text: {text}\n"
    "ASSISTANT: I've read this text.\n"
    "USER: What describes {entity_type} in the text?\n"
    "ASSISTANT:"
)
_JSON_LIST_RE = re.compile(r"\[\s*\(.+?\)\s*\]", re.DOTALL)


def build_uniner_prompt(text: str, entity_type: str) -> str:
    """Return UniversalNER-style instruction prompt for one entity type."""
    return _UNINER_PROMPT.format(text=text.strip(), entity_type=entity_type.strip())


def parse_uniner_response(raw: str, *, source_text: str, entity_type: str) -> list[DetectedEntity]:
    """Parse model output as list of (entity, type) tuples or JSON list."""
    cleaned = raw.strip()
    if not cleaned:
        return []
    payload: object
    if cleaned.startswith("["):
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            match = _JSON_LIST_RE.search(cleaned)
            if match is None:
                return []
            payload = json.loads(match.group(0).replace("(", "[").replace(")", "]"))
    else:
        match = _JSON_LIST_RE.search(cleaned)
        if match is None:
            return []
        inner = match.group(0)
        try:
            payload = json.loads(inner)
        except json.JSONDecodeError:
            try:
                payload = json.loads(inner.replace("(", "[").replace(")", "]"))
            except json.JSONDecodeError:
                return []
    if not isinstance(payload, list):
        return []
    found: list[DetectedEntity] = []
    label = entity_type.strip().lower()
    for item in payload:
        surface: str | None = None
        item_label = label
        if isinstance(item, (list, tuple)) and item:
            surface = str(item[0]).strip()
            if len(item) > 1:
                item_label = str(item[1]).strip().lower()
        elif isinstance(item, dict):
            surface = str(item.get("text") or item.get("entity") or "").strip()
            item_label = str(item.get("label") or label).strip().lower()
        if not surface or item_label != label:
            continue
        idx = source_text.find(surface)
        if idx < 0:
            idx = source_text.lower().find(surface.lower())
        if idx < 0:
            continue
        end = idx + len(surface)
        found.append(
            DetectedEntity(
                text=source_text[idx:end],
                label=item_label,
                score=1.0,
                start=idx,
                end=end,
            ),
        )
    return found


class GenerativeNerBackend:
    """Zero-shot NER via instruction-tuned text generation (UniversalNER template)."""

    backend = "generative_ner"

    def __init__(
        self,
        model_id: str,
        *,
        max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
    ) -> None:
        self.model_id = model_id
        self.max_chunk_chars = max_chunk_chars
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            tokenizer=self.model_id,
            device=device,
        )

    def detect(
        self,
        text: str,
        *,
        labels: list[str] | None = None,
        threshold: float = 0.5,
    ) -> list[DetectedEntity]:
        del threshold
        if not text.strip():
            return []
        entity_labels = labels if labels else list(_DEFAULT_LABELS)
        self._ensure_loaded()
        assert self._pipeline is not None
        chunks = chunk_text(
            text,
            max_chars=self.max_chunk_chars,
            overlap=_DEFAULT_CHUNK_OVERLAP,
        )
        found: list[DetectedEntity] = []
        for chunk in chunks:
            offset = chunk_offset(text, chunk)
            for label in entity_labels:
                prompt = build_uniner_prompt(chunk, label)
                output = self._pipeline(
                    prompt,
                    max_new_tokens=256,
                    do_sample=False,
                    return_full_text=False,
                )
                generated = ""
                if isinstance(output, list) and output:
                    generated = str(output[0].get("generated_text", ""))
                parsed = parse_uniner_response(
                    generated,
                    source_text=chunk,
                    entity_type=label,
                )
                found.extend(shift_entities(parsed, offset=offset, source=text))
        return merge_overlapping_entities(found, source=text)

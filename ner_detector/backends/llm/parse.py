"""Parse LLM JSON responses into DetectedEntity spans."""

from __future__ import annotations

import json
import re

from ner_detector.types import DetectedEntity

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE)
_DEFAULT_LABELS = ["person", "organization", "location", "date"]


class LlmParseError(ValueError):
    """Model output could not be parsed as entity JSON."""


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text)
        if text.endswith("```"):
            text = text[: text.rfind("```")].strip()
    return text


_SURFACE_KEYS = ("text", "entity", "span", "name", "value", "surface")
_ITEM_META_KEYS = frozenset({"label", "score", "type", "category", "confidence"})


def _surface_from_item(item: dict) -> str:
    """Extract entity surface form; tolerate missing ``text`` key from some models."""
    for key in _SURFACE_KEYS:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    for key, value in item.items():
        if key in _ITEM_META_KEYS:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _align_span(text: str, surface: str) -> tuple[int, int] | None:
    stripped = surface.strip()
    if not stripped:
        return None
    idx = text.find(stripped)
    if idx < 0:
        idx = text.lower().find(stripped.lower())
    if idx < 0:
        return None
    return idx, idx + len(stripped)


def parse_entities_response(
    raw: str,
    source_text: str,
    *,
    allowed_labels: set[str] | None = None,
    default_score: float = 1.0,
) -> list[DetectedEntity]:
    """Parse model JSON into aligned ``DetectedEntity`` list."""
    cleaned = _strip_json_fence(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LlmParseError(f"Invalid JSON from LLM: {exc}") from exc

    items = payload.get("entities") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise LlmParseError("Expected JSON object with 'entities' array")

    found: list[DetectedEntity] = []
    seen: set[tuple[int, int, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        surface = _surface_from_item(item)
        label = str(item.get("label", "")).strip().lower()
        if not surface or not label:
            continue
        if allowed_labels is not None and label not in allowed_labels:
            continue
        span = _align_span(source_text, surface)
        if span is None:
            continue
        start, end = span
        key = (start, end, label)
        if key in seen:
            continue
        seen.add(key)
        score_raw = item.get("score")
        score = float(score_raw) if score_raw is not None else default_score
        found.append(
            DetectedEntity(
                text=source_text[start:end],
                label=label,
                score=round(score, 4),
                start=start,
                end=end,
            ),
        )
    found.sort(key=lambda e: (e.start or 0, e.end or 0))
    return found


def default_llm_labels() -> list[str]:
    return list(_DEFAULT_LABELS)

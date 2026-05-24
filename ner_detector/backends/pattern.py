"""Deterministic pattern backend for tests and offline demos."""

from __future__ import annotations

import re

from ner_detector.types import DetectedEntity

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("arxiv_id", re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")),
    ("person", re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")),
    ("acronym", re.compile(r"\b[A-Z]{2,10}\b")),
    ("year", re.compile(r"\b(?:19|20)\d{2}\b")),
    ("number", re.compile(r"\b\d+(?:\.\d+)?%?\b")),
)
_STOP = frozenset({"HTTP", "HTTPS", "THE", "AND", "FOR", "WITH"})


class PatternBackend:
    """Regex-based NER without ML dependencies."""

    backend = "pattern"
    model_id = "builtin/pattern"

    def detect(
        self,
        text: str,
        *,
        labels: list[str] | None = None,
        threshold: float = 0.5,
    ) -> list[DetectedEntity]:
        del threshold  # unused
        allowed = set(labels) if labels else None
        found: list[DetectedEntity] = []
        seen: set[tuple[int, int, str]] = set()
        for label, pattern in _PATTERNS:
            if allowed is not None and label not in allowed:
                continue
            for match in pattern.finditer(text):
                span = match.group(0)
                if span in _STOP:
                    continue
                key = (match.start(), match.end(), label)
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    DetectedEntity(
                        text=span,
                        label=label,
                        score=1.0,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        found.sort(key=lambda e: (e.start or 0, e.end or 0))
        return found

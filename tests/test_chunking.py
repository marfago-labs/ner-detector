"""Tests for document chunking and span merging."""

from __future__ import annotations

from ner_detector.backends.chunking import (
    chunk_offset,
    chunk_text,
    merge_overlapping_entities,
    shift_entities,
)
from ner_detector.types import DetectedEntity


def test_chunk_text_empty() -> None:
    assert chunk_text("   ", max_chars=10, overlap=2) == []


def test_chunk_text_single_short() -> None:
    assert chunk_text("hello", max_chars=100, overlap=10) == ["hello"]


def test_merge_overlapping_same_label() -> None:
    source = "The Transformer model on WMT 2014"
    entities = [
        DetectedEntity(text="Transformer", label="model", start=4, end=15, score=0.9),
        DetectedEntity(text="Transformer model", label="model", start=4, end=21, score=0.8),
    ]
    merged = merge_overlapping_entities(entities, source=source)
    assert len(merged) == 1
    assert merged[0].text == "Transformer model"


def test_shift_entities_maps_to_source() -> None:
    source = "Alice works at OpenAI."
    local = [DetectedEntity(text="Alice", label="person", start=0, end=5, score=1.0)]
    shifted = shift_entities(local, offset=0, source=source)
    assert shifted[0].text == "Alice"
    assert shifted[0].start == 0


def test_chunk_offset_fallback() -> None:
    assert chunk_offset("hello world", "missing") == 0

"""Tests for LLM NER parsing."""

from __future__ import annotations

import pytest

from ner_detector.backends.llm.parse import LlmParseError, parse_entities_response


def test_parse_entities_response_success() -> None:
    text = "Alice works at OpenAI."
    raw = '{"entities": [{"text": "Alice", "label": "person", "score": 0.9}]}'
    entities = parse_entities_response(raw, text, allowed_labels={"person"})
    assert len(entities) == 1
    assert entities[0].text == "Alice"
    assert entities[0].label == "person"
    assert entities[0].start == 0


def test_parse_strips_json_fence() -> None:
    text = "Bob"
    raw = '```json\n{"entities": [{"text": "Bob", "label": "person"}]}\n```'
    entities = parse_entities_response(raw, text, allowed_labels={"person"})
    assert entities[0].text == "Bob"


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(LlmParseError):
        parse_entities_response("not json", "hello")


def test_parse_filters_unknown_labels() -> None:
    text = "OpenAI"
    raw = '{"entities": [{"text": "OpenAI", "label": "company"}]}'
    entities = parse_entities_response(raw, text, allowed_labels={"organization"})
    assert entities == []


def test_parse_missing_entities_key_raises() -> None:
    with pytest.raises(LlmParseError, match="entities"):
        parse_entities_response('{"items": []}', "hello")


def test_parse_skips_unaligned_surface() -> None:
    raw = '{"entities": [{"text": "MISSING", "label": "person"}]}'
    assert parse_entities_response(raw, "hello", allowed_labels={"person"}) == []


def test_parse_accepts_missing_text_key_with_empty_key() -> None:
    text = "GPT-4 was trained by OpenAI."
    raw = (
        '{"entities": [{"":"GPT-4","label":"model"},'
        '{"text":"OpenAI","label":"organization"}]}'
    )
    entities = parse_entities_response(
        raw,
        text,
        allowed_labels={"model", "organization"},
    )
    assert len(entities) == 2
    assert entities[0].text == "GPT-4"
    assert entities[1].text == "OpenAI"

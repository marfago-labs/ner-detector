"""Tests for LLM NER backend."""

from __future__ import annotations

from ner_detector.backends.llm_backend import LlmBackend, _response_has_entity_items
from ner_detector.registry import clear_backend_cache, create_backend


def test_llm_mock_backend_detects_person() -> None:
    clear_backend_cache()
    backend = LlmBackend("mock/ner", provider="mock")
    entities = backend.detect(
        "Alice Smith works at OpenAI.",
        labels=["person", "organization"],
    )
    labels = {e.label for e in entities}
    assert "person" in labels


def test_llm_registry_create() -> None:
    from ner_detector.registry import BackendOptions

    clear_backend_cache()
    backend = create_backend(
        "llm",
        model_id="mock/ner",
        options=BackendOptions(provider="mock"),
    )
    assert backend.backend == "llm"
    entities = backend.detect("Year 2024", labels=["year"])
    assert any(e.label == "year" for e in entities)


def test_llm_empty_text() -> None:
    backend = LlmBackend("mock/ner", provider="mock")
    assert backend.detect("  ") == []


def test_response_has_entity_items() -> None:
    assert _response_has_entity_items('{"entities": [{"text": "x", "label": "y"}]}')
    assert not _response_has_entity_items('{"entities": []}')
    assert not _response_has_entity_items("not json")


def test_detect_chunk_retries_when_items_unparsed(monkeypatch) -> None:
    calls: list[str] = []

    def fake_complete(_self, chunk: str, *, labels: list[str]) -> str:
        del labels
        calls.append(chunk)
        if len(calls) < 3:
            return '{"entities": [{"": ""}]}'
        return '{"entities": [{"text": "Alice", "label": "person"}]}'

    backend = LlmBackend("mock/ner", provider="mock")
    monkeypatch.setattr(backend, "_complete_chunk", fake_complete.__get__(backend, LlmBackend))
    entities = backend._detect_chunk(
        "Alice",
        labels=["person"],
        allowed={"person"},
    )
    assert len(entities) == 1
    assert entities[0].text == "Alice"
    assert len(calls) == 3

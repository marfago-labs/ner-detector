"""Additional tests for SOTA backends and chunking coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ner_detector.backends.chunking import collect_chunked_entities
from ner_detector.backends.generative_ner_backend import (
    GenerativeNerBackend,
    parse_uniner_response,
)
from ner_detector.backends.nuner_backend import NunerBackend
from ner_detector.config import resolve_label_definition_preset, resolve_label_definitions
from ner_detector.types import DetectedEntity


def test_collect_chunked_entities_merges_chunks() -> None:
    source = "Alice Smith works at OpenAI in Boston."

    def detect_chunk(chunk: str) -> list[DetectedEntity]:
        if chunk.startswith("Alice"):
            return [DetectedEntity(text="Alice Smith", label="person", start=0, end=11, score=0.9)]
        return [DetectedEntity(text="OpenAI", label="organization", start=0, end=6, score=0.8)]

    chunks = [source[:20], source[15:]]
    entities = collect_chunked_entities(source, chunks, detect_chunk=detect_chunk)
    labels = {e.label for e in entities}
    assert "person" in labels
    assert "organization" in labels


def test_parse_uniner_response_skips_non_matching_type() -> None:
    text = "GPT-4 was trained by OpenAI."
    raw = '[["GPT-4", "model"], ["OpenAI", "organization"]]'
    entities = parse_uniner_response(raw, source_text=text, entity_type="organization")
    assert len(entities) == 1
    assert entities[0].text == "OpenAI"


def test_parse_uniner_response_invalid_payload() -> None:
    assert parse_uniner_response("not json", source_text="x", entity_type="person") == []


def test_resolve_label_definition_preset_scientific() -> None:
    defs = resolve_label_definition_preset("scientific")
    assert defs is not None
    assert "model" in defs


def test_resolve_label_definitions_only_for_llm() -> None:
    assert (
        resolve_label_definitions(
            backend="gliner",
            label_definitions=None,
            label_definition_preset="scientific",
        )
        is None
    )


def test_nuner_default_labels_when_none() -> None:
    model = MagicMock()
    model.predict_entities.return_value = []
    with patch.dict(
        "sys.modules", {"gliner": MagicMock(GLiNER=MagicMock(from_pretrained=lambda _m: model))}
    ):
        NunerBackend("m").detect("hello")
    labels_arg = model.predict_entities.call_args[0][1]
    assert "person" in labels_arg


def test_generative_empty_text() -> None:
    backend = GenerativeNerBackend("Universal-NER/UniNER-7B-type")
    backend._pipeline = MagicMock()
    assert backend.detect("  ") == []


def test_shift_entities_without_positions() -> None:
    from ner_detector.backends.chunking import shift_entities

    local = [DetectedEntity(text="x", label="person", start=None, end=None, score=1.0)]
    shifted = shift_entities(local, offset=0, source="x")
    assert shifted[0].start is None


def test_parse_uniner_response_dict_items() -> None:
    text = "OpenAI released GPT-4."
    raw = '[{"text": "OpenAI", "label": "organization"}]'
    entities = parse_uniner_response(raw, source_text=text, entity_type="organization")
    assert entities[0].text == "OpenAI"


def test_build_ner_messages_label_without_definition() -> None:
    from ner_detector.backends.llm.prompts import build_ner_messages

    messages = build_ner_messages(
        "text",
        ["person", "organization"],
        label_definitions={"person": "A human"},
    )
    assert "- organization" in messages[1]["content"]


def test_ner_config_inline_label_definitions(tmp_path: Path) -> None:
    from ner_detector.config import load_ner_config, resolve_ner_settings

    path = tmp_path / "ner.yaml"
    path.write_text(
        "backend: llm\nprovider: mock\nlabel_definitions:\n  model: A named model\n",
        encoding="utf-8",
    )
    cfg = load_ner_config(str(path))
    assert cfg.label_definitions == {"model": "A named model"}
    resolved = resolve_ner_settings(config_path=str(path))
    assert resolved.label_definitions == {"model": "A named model"}

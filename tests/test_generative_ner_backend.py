"""Tests for generative UniversalNER backend (mocked pipeline)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import MagicMock, patch

from ner_detector.backends.generative_ner_backend import (
    GenerativeNerBackend,
    build_uniner_prompt,
    parse_uniner_response,
)


def test_build_uniner_prompt_contains_type() -> None:
    prompt = build_uniner_prompt("GPT-4 by OpenAI.", "model")
    assert "model" in prompt
    assert "GPT-4 by OpenAI." in prompt


def test_parse_uniner_response_json_list() -> None:
    text = "GPT-4 was trained by OpenAI."
    raw = '[["GPT-4", "model"], ["OpenAI", "organization"]]'
    entities = parse_uniner_response(raw, source_text=text, entity_type="model")
    assert len(entities) == 1
    assert entities[0].text == "GPT-4"


class _FakePipeline:
    def __call__(self, prompt: str, **kwargs: object) -> list[dict]:
        del kwargs
        if "person" in prompt:
            return [{"generated_text": '[["Jane Smith", "person"]]' }]
        return [{"generated_text": "[]"}]


@contextmanager
def _mock_generative_pipeline() -> Iterator[MagicMock]:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_tf = MagicMock()
    mock_tf.pipeline.return_value = _FakePipeline()
    with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
        yield mock_tf.pipeline


def test_generative_detect_with_mock_pipeline() -> None:
    text = "Jane Smith joined the lab."
    with _mock_generative_pipeline():
        backend = GenerativeNerBackend("Universal-NER/UniNER-7B-type")
        entities = backend.detect(text, labels=["person", "organization"])
    assert any(e.label == "person" for e in entities)

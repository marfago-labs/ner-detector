"""Tests for NuNER backend (mocked model)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from ner_detector.backends.nuner_backend import NunerBackend, merge_token_entities


class _FakeNunerModel:
    def predict_entities(self, text: str, labels: list[str], threshold: float = 0.5) -> list[dict]:
        del text, threshold
        assert labels == ["person"]
        return [
            {"text": "Ali", "label": "person", "score": 0.9, "start": 0, "end": 3},
            {"text": "ce", "label": "person", "score": 0.85, "start": 3, "end": 5},
        ]


@contextmanager
def _mock_gliner(model: object | None = None) -> Iterator[MagicMock]:
    mock_mod = MagicMock()
    mock_mod.GLiNER.from_pretrained.return_value = model or _FakeNunerModel()
    with patch.dict("sys.modules", {"gliner": mock_mod}):
        yield mock_mod.GLiNER


def test_nuner_lowercases_labels() -> None:
    with _mock_gliner() as gliner_cls:
        backend = NunerBackend("numind/NuNER_Zero")
        entities = backend.detect("Alice", labels=["Person"], threshold=0.4)
    assert len(entities) == 1
    assert entities[0].text == "Alice"
    gliner_cls.from_pretrained.assert_called_once_with("numind/NuNER_Zero")


def test_merge_token_entities_joins_adjacent_tokens() -> None:
    source = "Alice"
    merged = merge_token_entities(
        [
            {"text": "Ali", "label": "person", "start": 0, "end": 3},
            {"text": "ce", "label": "person", "start": 3, "end": 5},
        ],
        source=source,
    )
    assert len(merged) == 1
    assert merged[0]["text"] == "Alice"


def test_nuner_empty_text() -> None:
    backend = NunerBackend("m")
    backend._model = _FakeNunerModel()
    assert backend.detect("   ") == []

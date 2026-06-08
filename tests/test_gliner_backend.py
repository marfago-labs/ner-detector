"""Tests for GLiNER backend (mocked model)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from ner_detector.backends.gliner_backend import GlinerBackend


class _FakeGlinerModel:
    def predict_entities(self, text: str, labels: list[str], threshold: float = 0.5) -> list[dict]:
        return [
            {
                "text": "Alice",
                "label": "person",
                "score": 0.88,
                "start": 0,
                "end": 5,
            },
            {
                "text": "unknown",
                "label": "misc",
                "score": None,
                "start": 10,
                "end": 17,
            },
        ]


@contextmanager
def _mock_gliner(model: object | None = None) -> Iterator[MagicMock]:
    mock_mod = MagicMock()
    mock_mod.GLiNER.from_pretrained.return_value = model or _FakeGlinerModel()
    with patch.dict("sys.modules", {"gliner": mock_mod}):
        yield mock_mod.GLiNER


def test_gliner_detect_with_labels() -> None:
    with _mock_gliner() as gliner_cls:
        backend = GlinerBackend("test/gliner")
        entities = backend.detect("Alice works here", labels=["person"], threshold=0.4)
    assert len(entities) == 2
    assert entities[0].text == "Alice"
    assert entities[0].score == 0.88
    assert entities[1].score is None
    gliner_cls.from_pretrained.assert_called_once_with("test/gliner")


def test_gliner_default_labels_when_none() -> None:
    model = MagicMock()
    model.predict_entities.return_value = []
    with _mock_gliner(model):
        GlinerBackend("m").detect("text")
    labels_arg = model.predict_entities.call_args[0][1]
    assert "person" in labels_arg
    assert "organization" in labels_arg


def test_gliner_empty_text() -> None:
    backend = GlinerBackend("m")
    backend._model = _FakeGlinerModel()
    assert backend.detect("   ") == []


def test_gliner_ensure_loaded_is_idempotent() -> None:
    with _mock_gliner() as gliner_cls:
        backend = GlinerBackend("m")
        backend._ensure_loaded()
        backend._ensure_loaded()
    gliner_cls.from_pretrained.assert_called_once()

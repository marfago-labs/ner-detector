"""Tests for transformers backend (mocked pipeline)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from ner_detector.backends.chunking import chunk_text
from ner_detector.backends.transformers_backend import TransformersBackend


def test_chunk_text_empty() -> None:
    assert chunk_text("   ", max_chars=10, overlap=2) == []


def test_chunk_text_single_short() -> None:
    assert chunk_text("hello", max_chars=100, overlap=10) == ["hello"]


def test_chunk_text_splits_long_text() -> None:
    text = "a" * 100
    chunks = chunk_text(text, max_chars=40, overlap=10)
    assert len(chunks) > 1
    assert chunks[0]


def test_chunk_text_progresses_when_overlap_exceeds_max() -> None:
    chunks = chunk_text("abcdefghij", max_chars=4, overlap=10)
    assert len(chunks) >= 2


class _FakePipeline:
    def __init__(self, spans: list[dict] | None = None) -> None:
        self._spans = spans or []

    def __call__(self, chunk: str) -> list[dict]:
        return list(self._spans)


@contextmanager
def _mock_transformers(
    spans: list[dict] | None = None,
    *,
    cuda: bool = False,
) -> Iterator[MagicMock]:
    fake = _FakePipeline(spans)
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = cuda
    mock_tf = MagicMock()
    mock_tf.pipeline.return_value = fake
    with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
        yield mock_tf.pipeline


def test_transformers_detect_empty() -> None:
    backend = TransformersBackend("test/model")
    assert backend.detect("  ") == []


def test_transformers_loads_on_cuda() -> None:
    spans = [{"word": "Alice", "score": 0.9, "start": 0, "end": 5, "entity_group": "PER"}]
    with _mock_transformers(spans, cuda=True) as pipeline_ctor:
        entities = TransformersBackend("test/model").detect("Alice Smith")
    assert len(entities) == 1
    assert entities[0].label == "PER"
    assert entities[0].score == 0.9
    pipeline_ctor.assert_called_once()
    assert pipeline_ctor.call_args.kwargs["device"] == 0


def test_transformers_cpu_device() -> None:
    with _mock_transformers([], cuda=False) as pipeline_ctor:
        TransformersBackend("m").detect("x")
    assert pipeline_ctor.call_args.kwargs["device"] == -1


def test_transformers_skips_low_score_and_empty_word() -> None:
    spans = [
        {"word": "", "score": 0.9, "start": 0, "end": 0, "entity": "PER"},
        {"word": "Bob", "score": 0.1, "start": 0, "end": 3, "entity": "PER"},
    ]
    with _mock_transformers(spans):
        assert TransformersBackend("m").detect("Bob", threshold=0.5) == []


def test_transformers_dedupes_spans() -> None:
    span = {"word": "OpenAI", "score": 0.8, "start": 0, "end": 6, "entity": "ORG"}
    with _mock_transformers([span, span]):
        entities = TransformersBackend("m").detect("OpenAI")
    assert len(entities) == 1


def test_transformers_chunking_and_fallback_offset() -> None:
    spans = [{"word": "x", "score": 0.9, "start": 0, "end": 1, "entity_group": "MISC"}]
    with _mock_transformers(spans):
        long_text = "word " + ("y" * 8000)
        entities = TransformersBackend("m", max_chunk_chars=3000).detect(long_text)
    assert entities


def test_transformers_ensure_loaded_is_idempotent() -> None:
    with _mock_transformers([]) as pipeline_ctor:
        backend = TransformersBackend("m")
        backend._ensure_loaded()
        backend._ensure_loaded()
    pipeline_ctor.assert_called_once()


def test_transformers_chunk_offset_fallback_when_not_in_text() -> None:
    spans = [{"word": "y", "score": 0.9, "start": 0, "end": 1, "entity_group": "MISC"}]
    with _mock_transformers(spans):
        padded = "   " + ("y" * 8000)
        entities = TransformersBackend("m", max_chunk_chars=3000).detect(padded)
    assert entities


def test_transformers_span_without_start_still_emits() -> None:
    spans = [{"word": "Org", "score": 0.95, "entity_group": "ORG"}]
    with _mock_transformers(spans):
        entities = TransformersBackend("m").detect("Org")
    assert len(entities) == 1
    assert entities[0].start is None


def test_transformers_prefers_source_slice_over_wordpiece() -> None:
    text = "Experiments on CIFAR-10 show gains."
    start = text.index("CIFAR-10")
    end = start + len("CIFAR-10")
    spans = [
        {
            "word": "##IFAR - 10",
            "score": 0.95,
            "start": start,
            "end": end,
            "entity_group": "MISC",
        }
    ]
    with _mock_transformers(spans):
        entities = TransformersBackend("m").detect(text)
    assert len(entities) == 1
    assert entities[0].text == "CIFAR-10"

"""Registry tests for ML backends (mocked constructors)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ner_detector.registry import clear_backend_cache, create_backend


def test_create_transformers_backend() -> None:
    clear_backend_cache()
    with patch("ner_detector.backends.transformers_backend.TransformersBackend") as mock_cls:
        mock_cls.return_value = MagicMock(backend="transformers", model_id="custom/ner")
        backend = create_backend("transformers", model_id="custom/ner")
        mock_cls.assert_called_once_with("custom/ner")
        assert backend is mock_cls.return_value


def test_create_gliner_backend() -> None:
    clear_backend_cache()
    with patch("ner_detector.backends.gliner_backend.GlinerBackend") as mock_cls:
        mock_cls.return_value = MagicMock(backend="gliner", model_id="test/gliner")
        backend = create_backend("gliner")
        mock_cls.assert_called_once()
        assert backend is mock_cls.return_value

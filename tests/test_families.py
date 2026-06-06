"""Tests for backend family helpers."""

from __future__ import annotations

from ner_detector.backends.families import backend_family, uses_score_threshold


def test_backend_families() -> None:
    assert backend_family("pattern") == "deterministic"
    assert backend_family("transformers") == "ml"
    assert backend_family("gliner") == "ml"
    assert backend_family("nuner") == "ml"
    assert backend_family("generative_ner") == "ml"
    assert backend_family("llm") == "llm"


def test_uses_score_threshold() -> None:
    assert uses_score_threshold("gliner")
    assert uses_score_threshold("transformers")
    assert uses_score_threshold("nuner")
    assert not uses_score_threshold("pattern")
    assert not uses_score_threshold("llm")
    assert not uses_score_threshold("generative_ner")

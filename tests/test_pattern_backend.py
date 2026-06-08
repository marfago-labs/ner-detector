"""Tests for pattern backend and public API."""

from __future__ import annotations

import json
from io import StringIO

import pytest

from ner_detector import detect_entities
from ner_detector.backends.pattern import PatternBackend
from ner_detector.cli import _parse_labels, main
from ner_detector.config import default_model_id, load_model_config, resolve_ner_settings
from ner_detector.registry import clear_backend_cache, create_backend
from ner_detector.types import DetectedEntity


def test_pattern_detects_person_and_year() -> None:
    text = "Alice Smith published work in 2024."
    entities = detect_entities(text, backend="pattern")
    labels = {e.label for e in entities}
    assert "person" in labels
    assert "year" in labels


def test_pattern_respects_label_filter() -> None:
    entities = detect_entities(
        "Alice Smith and 2024",
        backend="pattern",
        labels=["year"],
    )
    assert all(e.label == "year" for e in entities)


def test_pattern_skips_stop_acronyms() -> None:
    entities = detect_entities("HTTP and HTTPS", backend="pattern", labels=["acronym"])
    texts = {e.text for e in entities}
    assert "HTTP" not in texts
    assert "HTTPS" not in texts


def test_pattern_empty_text() -> None:
    assert PatternBackend().detect("   ") == []


def test_detected_entity_to_dict() -> None:
    entity = DetectedEntity(text="OpenAI", label="ORG", score=0.9, start=0, end=6)
    assert entity.to_dict()["label"] == "ORG"


def test_default_model_ids() -> None:
    assert default_model_id("transformers") == "dslim/bert-base-NER"
    assert "gliner" in default_model_id("gliner")


def test_load_model_config_has_backends() -> None:
    cfg = load_model_config()
    assert "transformers" in cfg
    assert "gliner" in cfg


def test_registry_caches_backends() -> None:
    clear_backend_cache()
    a = create_backend("pattern")
    b = create_backend("pattern")
    assert a is b
    clear_backend_cache()


def test_cli_json_output(capsys) -> None:
    code = main(["Alice Smith works at OpenAI in 2024.", "--backend", "pattern"])
    assert code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["backend"] == "pattern"
    assert data["entity_count"] >= 1


def test_cli_text_format(capsys) -> None:
    code = main(["2024", "--format", "text", "--labels", "year"])
    assert code == 0
    assert "2024" in capsys.readouterr().out


def test_cli_list_models(capsys) -> None:
    code = main(["--list-models"])
    assert code == 0
    assert "transformers" in capsys.readouterr().out


def test_cli_empty_input_stderr(capsys) -> None:
    code = main(["   "])
    assert code == 1
    assert "No input" in capsys.readouterr().err


def test_cli_missing_ml_extra(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import(*_a, **_k):
        raise ImportError("torch not installed")

    monkeypatch.setattr("ner_detector.cli.detect_entities", _raise_import)
    code = main(["text", "--backend", "transformers"])
    assert code == 1
    err = capsys.readouterr().err
    assert "uv sync" in err
    assert "ml" in err


def test_cli_stdin(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", StringIO("Paper 2402.15343."))
    code = main([])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    labels = {e["label"] for e in data["entities"]}
    assert "arxiv_id" in labels


def test_cli_file_mode(tmp_path, capsys) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("Alice Smith in 2024.", encoding="utf-8")
    code = main([str(path), "--file"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["entity_count"] >= 1


def test_cli_missing_file(capsys) -> None:
    code = main(["/no/such/file.txt", "--file"])
    assert code == 1
    assert "File not found" in capsys.readouterr().err


def test_cli_gliner_uses_label_presets(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_detect(_text: str, **kwargs: object) -> list:
        captured["labels"] = kwargs.get("labels")
        return []

    monkeypatch.setattr("ner_detector.cli.detect_entities", _fake_detect)
    code = main(["Alice Smith in Paris.", "--backend", "gliner", "--config", "nope.yaml"])
    assert code == 0
    labels = captured.get("labels")
    assert isinstance(labels, list)
    assert "person" in labels


def test_resolve_ner_settings_from_repo_default() -> None:
    resolved = resolve_ner_settings()
    assert resolved.backend == "pattern"
    assert resolved.threshold == 0.5


def test_cli_value_error(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_value(*_a, **_k):
        raise ValueError("bad backend config")

    monkeypatch.setattr("ner_detector.cli.detect_entities", _raise_value)
    code = main(["text", "--backend", "pattern"])
    assert code == 1
    assert "bad backend" in capsys.readouterr().err


def test_cli_gliner_import_error(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import(*_a, **_k):
        raise ImportError("gliner missing")

    monkeypatch.setattr("ner_detector.cli.detect_entities", _raise_import)
    code = main(["text", "--backend", "gliner"])
    assert code == 1
    assert "gliner" in capsys.readouterr().err


def test_cli_text_format_without_score(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    from ner_detector.types import DetectedEntity

    fake = [DetectedEntity(text="Acme", label="ORG", score=None)]
    monkeypatch.setattr(
        "ner_detector.cli.detect_entities",
        lambda *_a, **_k: fake,
    )
    code = main(["ignored", "--format", "text"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Acme" in out
    assert "ORG" in out
    assert "(" not in out


def test_cli_oserror_on_read(capsys, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "broken.txt"
    path.write_text("x", encoding="utf-8")

    def _boom(_self, *_a, **_k):
        raise OSError("read failed")

    monkeypatch.setattr(type(path), "read_text", _boom)
    code = main([str(path), "--file"])
    assert code == 1
    assert "read failed" in capsys.readouterr().err


def test_parse_labels_empty_commas() -> None:
    assert _parse_labels(" , , ") is None


def test_pattern_skips_duplicate_span(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = PatternBackend()
    match = type(
        "M",
        (),
        {
            "group": lambda self, _i=0: "Dup",
            "start": lambda self: 0,
            "end": lambda self: 3,
        },
    )()
    fake_pattern = type("P", (), {"finditer": lambda self, _t: iter([match, match])})()
    monkeypatch.setattr(
        "ner_detector.backends.pattern._PATTERNS",
        (("person", fake_pattern),),
    )
    entities = backend.detect("Dup", labels=["person"])
    assert len(entities) == 1

"""CLI for ner-detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ner_detector.env import load_project_env

load_project_env()

from ner_detector.config import (
    default_ner_config_path,
    load_model_config,
    resolve_ner_settings,
)
from ner_detector.detect import detect_entities
from ner_detector.types import NerBackend


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ner-detect",
        description="Extract named entities from text or files.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Text to analyze, or path to a .txt file when --file is set",
    )
    parser.add_argument(
        "--file",
        "-f",
        action="store_true",
        help="Treat input as a file path",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to ner.yaml (default: config/ner.yaml or NER_CONFIG_PATH)",
    )
    parser.add_argument(
        "--backend",
        "-b",
        choices=("pattern", "transformers", "gliner"),
        default=None,
        help="NER backend (overrides config file)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Hugging Face model id (overrides config file)",
    )
    parser.add_argument(
        "--labels",
        "-l",
        default=None,
        help="Comma-separated entity labels (overrides config file)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Minimum confidence score (overrides config file)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print model catalog (default_models.yaml) and exit",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print resolved settings from config file and exit",
    )
    return parser


def _read_input(args: argparse.Namespace) -> str:
    if args.input is None:
        return sys.stdin.read()
    if args.file:
        path = Path(args.input)
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")
        return path.read_text(encoding="utf-8")
    return args.input


def _parse_labels(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    labels = [part.strip() for part in raw.split(",") if part.strip()]
    return labels or None


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        print(json.dumps(load_model_config(), indent=2))
        return 0

    settings = resolve_ner_settings(
        config_path=args.config,
        backend=args.backend,
        model_id=args.model,
        labels=_parse_labels(args.labels),
        threshold=args.threshold,
    )

    if args.show_config:
        payload = {
            "config_path": str(settings.config_path or default_ner_config_path()),
            "backend": settings.backend,
            "model_id": settings.model_id,
            "threshold": settings.threshold,
            "labels": settings.labels,
        }
        print(json.dumps(payload, indent=2))
        return 0

    try:
        text = _read_input(args)
    except SystemExit as exc:
        message = str(exc) if exc.args else ""
        if message:
            print(message, file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error reading input: {exc}", file=sys.stderr)
        return 1

    if not text.strip():
        print("No input text.", file=sys.stderr)
        return 1

    backend: NerBackend = settings.backend

    try:
        entities = detect_entities(
            text,
            backend=backend,
            model_id=settings.model_id,
            labels=settings.labels,
            threshold=settings.threshold,
        )
    except ImportError as exc:
        extra = "ml" if backend == "transformers" else "gliner"
        print(
            f"Missing dependency for backend {backend!r}: {exc}\n"
            f"Install with: uv sync --extra {extra}",
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.format == "json":
        payload = {
            "backend": backend,
            "model_id": settings.model_id,
            "config_path": str(settings.config_path) if settings.config_path else None,
            "entity_count": len(entities),
            "entities": [e.to_dict() for e in entities],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for entity in entities:
            score = f" ({entity.score:.2f})" if entity.score is not None else ""
            print(f"{entity.text}\t{entity.label}{score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

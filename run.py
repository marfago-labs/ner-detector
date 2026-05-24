#!/usr/bin/env python3
"""Convenience entrypoint: ``uv run python run.py ...``"""

from ner_detector.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

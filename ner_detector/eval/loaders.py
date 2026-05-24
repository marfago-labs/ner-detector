"""Load gold benchmark datasets."""

from __future__ import annotations

import json
from pathlib import Path

from ner_detector.eval.types import GoldEntity, GoldExample

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BENCHMARK_ROOT = _PACKAGE_ROOT / "benchmark"


def benchmark_root(path: str | None = None) -> Path:
    return Path(path) if path else _DEFAULT_BENCHMARK_ROOT


def dataset_path(name: str, *, root: Path | None = None) -> Path:
    base = root or _DEFAULT_BENCHMARK_ROOT
    return base / "datasets" / f"{name}.jsonl"


def load_gold_jsonl(path: Path) -> list[GoldExample]:
    if not path.is_file():
        raise FileNotFoundError(f"Gold dataset not found: {path}")
    examples: list[GoldExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
                examples.append(GoldExample.model_validate(raw))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid gold JSONL at {path}:{line_no}: {exc}") from exc
    return examples


def load_dataset(
    name: str,
    *,
    root: Path | None = None,
    max_examples: int | None = None,
) -> list[GoldExample]:
    path = dataset_path(name, root=root)
    examples = load_gold_jsonl(path)
    if max_examples is not None:
        return examples[:max_examples]
    return examples

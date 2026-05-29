"""Load gold benchmark datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ner_detector.eval.types import GoldEntity, GoldExample

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BENCHMARK_ROOT = _PACKAGE_ROOT / "benchmark"


def benchmark_root(path: str | None = None) -> Path:
    return Path(path) if path else _DEFAULT_BENCHMARK_ROOT


def _dataset_candidate_paths(name: str) -> list[Path]:
    """Search paths for ``{name}.jsonl`` (sibling ner-dataset first, then bundled)."""
    filename = f"{name}.jsonl"
    candidates: list[Path] = []
    env = os.environ.get("NER_DATASET_DIR", "").strip()
    if env:
        candidates.append(Path(env) / filename)
    sibling = _PACKAGE_ROOT.parent / "ner-dataset" / "datasets" / filename
    if sibling not in candidates:
        candidates.append(sibling)
    bundled = _DEFAULT_BENCHMARK_ROOT / "datasets" / filename
    if bundled not in candidates:
        candidates.append(bundled)
    return candidates


def dataset_path(name: str, *, root: Path | None = None) -> Path:
    """Resolve a gold JSONL path by dataset name.

    When *root* is omitted, returns the first existing file among:

    1. ``NER_DATASET_DIR/{name}.jsonl`` (if env set)
    2. ``../ner-dataset/datasets/{name}.jsonl`` (sibling repo)
    3. ``benchmark/datasets/{name}.jsonl`` (shipped with ner-detector)

    If none exist, returns the bundled path so :func:`load_gold_jsonl` raises a
    clear error.
    """
    if root is not None:
        return root / "datasets" / f"{name}.jsonl"
    for path in _dataset_candidate_paths(name):
        if path.is_file():
            return path
    return _DEFAULT_BENCHMARK_ROOT / "datasets" / f"{name}.jsonl"


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

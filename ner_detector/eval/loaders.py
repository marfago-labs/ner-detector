"""Load gold benchmark datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ner_detector.eval.types import GoldEntity, GoldExample

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
_SIBLING_NER_DATASET = _PACKAGE_ROOT.parent / "ner-dataset"


def sibling_ner_dataset_root() -> Path:
    """Path to the canonical ner-dataset repo next to ner-detector."""
    return _SIBLING_NER_DATASET


def benchmark_root(path: str | None = None) -> Path:
    """Return the gold dataset repo root (``…/datasets/`` lives beneath it)."""
    return Path(path) if path else _SIBLING_NER_DATASET


def resolve_benchmark_root(
    raw: str | None,
    *,
    config_path: Path | None = None,
) -> Path:
    """Resolve gold dataset root directory.

    Tries, in order:

    1. Absolute ``raw`` path
    2. ``raw`` relative to the benchmark config file (when *config_path* set)
    3. ``raw`` relative to the current working directory
    4. Sibling ``../ner-dataset`` next to the ner-detector package
    """
    if not raw:
        sibling = _SIBLING_NER_DATASET.resolve()
        if (sibling / "datasets").is_dir():
            return sibling
        env = os.environ.get("NER_DATASET_DIR", "").strip()
        if env:
            datasets_dir = Path(env).resolve()
            if datasets_dir.is_dir():
                return datasets_dir.parent
        return sibling

    path = Path(raw)
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    if config_path is not None:
        candidates.append((config_path.parent / path).resolve())
    candidates.append((Path.cwd() / path).resolve())
    sibling = _SIBLING_NER_DATASET.resolve()
    if sibling not in candidates:
        candidates.append(sibling)

    for candidate in candidates:
        if (candidate / "datasets").is_dir():
            return candidate
    return candidates[0]


def _dataset_candidate_paths(name: str) -> list[Path]:
    """Search paths for ``{name}.jsonl`` (env override, then sibling ner-dataset)."""
    filename = f"{name}.jsonl"
    candidates: list[Path] = []
    env = os.environ.get("NER_DATASET_DIR", "").strip()
    if env:
        candidates.append(Path(env) / filename)
    sibling = _SIBLING_NER_DATASET / "datasets" / filename
    if sibling not in candidates:
        candidates.append(sibling)
    return candidates


def dataset_path(name: str, *, root: Path | None = None) -> Path:
    """Resolve a gold JSONL path by dataset name.

    When *root* is omitted, returns the first existing file among:

    1. ``NER_DATASET_DIR/{name}.jsonl`` (if env set)
    2. ``../ner-dataset/datasets/{name}.jsonl`` (sibling repo)

    If none exist, returns the sibling path so :func:`load_gold_jsonl` raises a
    clear error.
    """
    if root is not None:
        return root / "datasets" / f"{name}.jsonl"
    for path in _dataset_candidate_paths(name):
        if path.is_file():
            return path
    return _SIBLING_NER_DATASET / "datasets" / f"{name}.jsonl"


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

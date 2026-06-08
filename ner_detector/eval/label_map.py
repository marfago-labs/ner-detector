"""Label normalization across backends and gold data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MAPS_PATH = _PACKAGE_ROOT / "benchmark" / "config" / "label_maps.yaml"

# BERT / CoNLL style entity_group values
_TRANSFORMERS_ALIASES: dict[str, str] = {
    "per": "person",
    "person": "person",
    "org": "organization",
    "organization": "organization",
    "loc": "location",
    "location": "location",
    "misc": "miscellaneous",
    "miscellaneous": "miscellaneous",
    "date": "date",
    "product": "product",
}


@lru_cache(maxsize=4)
def load_label_maps(path: str | None = None) -> dict[str, dict[str, str]]:
    maps_path = Path(path) if path else _DEFAULT_MAPS_PATH
    if not maps_path.is_file():
        return {"unified": dict(_TRANSFORMERS_ALIASES)}
    with maps_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        return {"unified": dict(_TRANSFORMERS_ALIASES)}
    out: dict[str, dict[str, str]] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out[str(key)] = {str(k).lower(): str(v).lower() for k, v in value.items()}
    return out


def normalize_label(raw: str, map_name: str = "unified") -> str:
    """Map backend-specific label to unified scoring label."""
    key = raw.strip().lower()
    if not key:
        return "unknown"
    maps = load_label_maps()
    mapping = maps.get(map_name, {})
    if key in mapping:
        return mapping[key]
    if key in _TRANSFORMERS_ALIASES:
        return _TRANSFORMERS_ALIASES[key]
    return key.replace(" ", "_")


def clear_label_map_cache() -> None:
    load_label_maps.cache_clear()

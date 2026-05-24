"""Load model catalog and runtime NER settings from YAML."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from ner_detector.types import NerBackend

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_MODELS_PATH = _PACKAGE_ROOT / "config" / "default_models.yaml"
_DEFAULT_NER_PATH = _PACKAGE_ROOT / "config" / "ner.yaml"
_ENV_CONFIG_PATH = "NER_CONFIG_PATH"


class NerRuntimeConfig(BaseModel):
    """Runtime NER settings from ``config/ner.yaml`` (or a custom path)."""

    backend: NerBackend = "pattern"
    model_id: str | None = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    labels: list[str] | None = None
    label_preset: str = "general_en"

    @field_validator("model_id", mode="before")
    @classmethod
    def _empty_model_id_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("labels", mode="before")
    @classmethod
    def _normalize_labels(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            return parts or None
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()] or None
        return value


class ResolvedNerSettings(BaseModel):
    """Fully resolved settings after merging config file and CLI overrides."""

    backend: NerBackend
    model_id: str | None
    threshold: float
    labels: list[str] | None
    config_path: Path | None = None


@lru_cache(maxsize=1)
def load_model_config() -> dict[str, Any]:
    if not _MODELS_PATH.is_file():
        return {}
    with _MODELS_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def default_model_id(backend: str) -> str:
    section = load_model_config().get(backend, {})
    if isinstance(section, dict):
        value = section.get("default")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if backend == "transformers":
        return "dslim/bert-base-NER"
    if backend == "gliner":
        return "urchade/gliner_medium-v2.1"
    return ""


def default_ner_config_path() -> Path:
    env_path = os.environ.get(_ENV_CONFIG_PATH, "").strip()
    if env_path:
        return Path(env_path)
    return _DEFAULT_NER_PATH


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=8)
def load_ner_config(path: str | None = None) -> NerRuntimeConfig:
    """Load runtime settings from ``config/ner.yaml`` or ``path``."""
    config_path = Path(path) if path else default_ner_config_path()
    raw = _load_yaml_mapping(config_path)
    return NerRuntimeConfig.model_validate(raw)


def resolve_label_preset(preset_name: str) -> list[str] | None:
    presets = load_model_config().get("label_presets", {})
    if not isinstance(presets, dict):
        return None
    preset = presets.get(preset_name)
    if isinstance(preset, list):
        return [str(item) for item in preset]
    return None


def resolve_labels(
    *,
    backend: NerBackend,
    labels: list[str] | None,
    label_preset: str,
) -> list[str] | None:
    if labels:
        return labels
    if backend == "gliner":
        return resolve_label_preset(label_preset) or resolve_label_preset("general_en")
    return None


def resolve_ner_settings(
    *,
    config_path: str | None = None,
    backend: NerBackend | None = None,
    model_id: str | None = None,
    labels: list[str] | None = None,
    threshold: float | None = None,
) -> ResolvedNerSettings:
    """Merge YAML config with CLI overrides (CLI wins)."""
    path = Path(config_path) if config_path else default_ner_config_path()
    file_cfg = load_ner_config(str(path)) if path.is_file() else NerRuntimeConfig()

    resolved_backend = backend if backend is not None else file_cfg.backend
    resolved_threshold = threshold if threshold is not None else file_cfg.threshold
    resolved_labels = labels if labels is not None else file_cfg.labels

    resolved_model = model_id if model_id is not None else file_cfg.model_id
    if resolved_model is None and resolved_backend != "pattern":
        fallback = default_model_id(resolved_backend)
        resolved_model = fallback or None

    resolved_labels = resolve_labels(
        backend=resolved_backend,
        labels=resolved_labels,
        label_preset=file_cfg.label_preset,
    )

    return ResolvedNerSettings(
        backend=resolved_backend,
        model_id=resolved_model,
        threshold=resolved_threshold,
        labels=resolved_labels,
        config_path=path if path.is_file() else None,
    )


def clear_config_caches() -> None:
    """Clear cached config (for tests)."""
    load_model_config.cache_clear()
    load_ner_config.cache_clear()

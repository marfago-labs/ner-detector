"""Environment-backed settings for LLM NER."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class LlmEnvSettings:
    openrouter_api_key: str
    openrouter_base_url: str
    mock_llm: bool


def load_llm_env() -> LlmEnvSettings:
    return LlmEnvSettings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", "").strip(),
        openrouter_base_url=os.environ.get(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ).strip(),
        mock_llm=_env_bool("MOCK_LLM", default=False),
    )

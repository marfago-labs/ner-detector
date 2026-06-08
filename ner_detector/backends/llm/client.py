"""Chat clients for LLM NER (mock and OpenRouter)."""

from __future__ import annotations

import json
from typing import Any, Protocol

from ner_detector.backends.llm.prompts import build_ner_messages
from ner_detector.backends.llm.settings import LlmEnvSettings, load_llm_env
from ner_detector.backends.pattern import PatternBackend


class LlmChatClient(Protocol):
    def complete_json(
        self,
        text: str,
        *,
        labels: list[str],
        model_id: str,
        temperature: float = 0.0,
        label_definitions: dict[str, str] | None = None,
        few_shot_examples: list[dict[str, object]] | None = None,
    ) -> str: ...


class OpenRouterApiError(ValueError):
    """OpenRouter returned a non-success HTTP status."""


# Keep chain-of-thought out of `content` so NER JSON parses cleanly.
# `exclude: true` works for reasoning-mandatory models (e.g. gpt-oss); `effort: none` does not.
_OPENROUTER_REASONING: dict[str, bool] = {"exclude": True}


def _openrouter_error_message(response: Any) -> str:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
        if isinstance(err, str):
            return err
    return response.text.strip() or f"HTTP {response.status_code}"


class MockLlmClient:
    """Deterministic offline client for CI and benchmarks without API keys."""

    def complete_json(
        self,
        text: str,
        *,
        labels: list[str],
        model_id: str,
        temperature: float = 0.0,
        label_definitions: dict[str, str] | None = None,
        few_shot_examples: list[dict[str, object]] | None = None,
    ) -> str:
        del model_id, temperature, label_definitions, few_shot_examples
        entities = PatternBackend().detect(text, labels=labels, threshold=0.0)
        payload = {
            "entities": [
                {"text": e.text, "label": e.label, "score": e.score or 1.0} for e in entities
            ],
        }
        return json.dumps(payload, ensure_ascii=False)


class OpenRouterChatClient:
    """OpenRouter chat completions with JSON response body."""

    def __init__(self, settings: LlmEnvSettings | None = None) -> None:
        self._settings = settings or load_llm_env()

    def complete_json(
        self,
        text: str,
        *,
        labels: list[str],
        model_id: str,
        temperature: float = 0.0,
        label_definitions: dict[str, str] | None = None,
        few_shot_examples: list[dict[str, object]] | None = None,
    ) -> str:
        api_key = self._settings.openrouter_api_key
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Use provider: mock, MOCK_LLM=true, "
                "or set a key in .env for live OpenRouter runs."
            )
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "LLM backend requires httpx. Install with: uv sync --extra llm"
            ) from exc

        messages = build_ner_messages(
            text,
            labels,
            label_definitions=label_definitions,
            few_shot_examples=few_shot_examples,
        )
        url = f"{self._settings.openrouter_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, object] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
            "reasoning": dict(_OPENROUTER_REASONING),
        }
        with httpx.Client(timeout=180.0) as client:
            response = client.post(url, headers=headers, json=body)
            if response.status_code >= 400:
                detail = _openrouter_error_message(response)
                raise OpenRouterApiError(
                    f"OpenRouter request failed ({response.status_code}): {detail}"
                )
            data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected OpenRouter response shape: {data!r}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenRouter returned empty message content")
        return content


def create_llm_client(provider: str, settings: LlmEnvSettings | None = None) -> LlmChatClient:
    """Return a chat client for ``mock`` or ``openrouter``."""
    normalized = provider.strip().lower()
    env = settings or load_llm_env()
    if normalized == "mock" or env.mock_llm:
        return MockLlmClient()
    if normalized == "openrouter":
        return OpenRouterChatClient(env)
    raise ValueError(f"Unknown LLM provider {provider!r}. Use 'mock' or 'openrouter'.")

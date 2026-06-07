"""Tests for LLM chat clients."""

from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest

from ner_detector.backends.llm.client import MockLlmClient, OpenRouterChatClient, create_llm_client
from ner_detector.backends.llm.settings import LlmEnvSettings


def _install_fake_httpx(monkeypatch: pytest.MonkeyPatch, client_cls: type) -> None:
    """Inject a fake httpx module so OpenRouter tests run without the llm extra."""
    mod = ModuleType("httpx")
    mod.Client = client_cls
    monkeypatch.setitem(sys.modules, "httpx", mod)


def test_mock_client_returns_json_entities() -> None:
    raw = MockLlmClient().complete_json(
        "Alice Smith joined OpenAI in 2024.",
        labels=["person", "organization", "year"],
        model_id="mock/ner",
    )
    data = json.loads(raw)
    assert "entities" in data
    assert len(data["entities"]) >= 1


def test_create_llm_client_mock_provider() -> None:
    client = create_llm_client("mock")
    assert isinstance(client, MockLlmClient)


def test_create_llm_client_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm_client("unknown")


def test_openrouter_requires_api_key() -> None:
    client = OpenRouterChatClient(
        LlmEnvSettings(openrouter_api_key="", openrouter_base_url="https://x", mock_llm=False),
    )
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        client.complete_json("hello", labels=["person"], model_id="m")


def test_openrouter_http_error_raises_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 400

        @staticmethod
        def json() -> dict:
            return {"error": {"message": "not a valid model ID"}}

        text = ""

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, *args: object, **kwargs: object) -> FakeResponse:
            return FakeResponse()

    _install_fake_httpx(monkeypatch, FakeClient)
    client = OpenRouterChatClient(
        LlmEnvSettings(openrouter_api_key="secret", openrouter_base_url="https://x/v1", mock_llm=False),
    )
    from ner_detector.backends.llm.client import OpenRouterApiError

    with pytest.raises(OpenRouterApiError, match="not a valid model ID"):
        client.complete_json("hello", labels=["person"], model_id="bad/model")


def test_openrouter_complete_json(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[dict] = []

    class FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": '{"entities": []}'}}],
            }

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, *args: object, **kwargs: object) -> FakeResponse:
            posted.append(kwargs["json"])
            return FakeResponse()

    _install_fake_httpx(monkeypatch, FakeClient)
    client = OpenRouterChatClient(
        LlmEnvSettings(openrouter_api_key="secret", openrouter_base_url="https://x/v1", mock_llm=False),
    )
    raw = client.complete_json("hello", labels=["person"], model_id="m")
    assert "entities" in raw
    assert posted[0]["reasoning"] == {"exclude": True}


def test_create_llm_client_respects_mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM", "true")
    client = create_llm_client("openrouter")
    assert isinstance(client, MockLlmClient)

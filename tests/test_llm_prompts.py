"""Tests for LLM prompts."""

from __future__ import annotations

from ner_detector.backends.llm.prompts import build_ner_messages


def test_build_ner_messages_includes_labels_and_text() -> None:
    messages = build_ner_messages("Alice at OpenAI.", ["person", "organization"])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "text and label" in messages[0]["content"]
    assert "person" in messages[1]["content"]
    assert "Alice at OpenAI." in messages[1]["content"]


def test_build_ner_messages_with_definitions_and_few_shot() -> None:
    messages = build_ner_messages(
        "GPT-4 by OpenAI.",
        ["model", "organization"],
        label_definitions={"model": "A named AI system", "organization": "A company"},
        few_shot_examples=[
            {
                "text": "Example text",
                "entities": [{"text": "Example text", "label": "model"}],
            },
        ],
    )
    user = messages[1]["content"]
    assert "Entity type definitions" in user
    assert "Few-shot examples" in user
    assert "A named AI system" in user

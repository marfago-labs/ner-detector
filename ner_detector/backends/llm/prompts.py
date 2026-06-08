"""Prompt templates for LLM NER."""

from __future__ import annotations

import json
from typing import Any


def _format_label_definitions(labels: list[str], definitions: dict[str, str]) -> str:
    lines: list[str] = []
    for label in labels:
        definition = definitions.get(label) or definitions.get(label.lower())
        if definition:
            lines.append(f"- {label}: {definition}")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


def _format_few_shot_examples(examples: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, example in enumerate(examples, start=1):
        text = str(example.get("text", "")).strip()
        entities = example.get("entities")
        if not text or not isinstance(entities, list):
            continue
        payload = {
            "entities": [
                {"text": str(item.get("text", "")), "label": str(item.get("label", ""))}
                for item in entities
                if isinstance(item, dict)
            ],
        }
        blocks.append(
            f"Example {idx} text:\n{text}\n"
            f"Example {idx} response:\n{json.dumps(payload, ensure_ascii=False)}",
        )
    return "\n\n".join(blocks)


def build_ner_messages(
    text: str,
    labels: list[str],
    *,
    label_definitions: dict[str, str] | None = None,
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Return chat messages asking for JSON entity extraction."""
    label_list = ", ".join(labels)
    example = json.dumps(
        {
            "entities": [
                {"text": "GPT-4", "label": "model"},
                {"text": "OpenAI", "label": "organization"},
            ],
        },
        ensure_ascii=False,
    )
    system = (
        "You extract named entities from text. "
        "Return ONLY valid JSON with an entities array. "
        "Each entity object MUST use exactly these keys: text and label. "
        "The text value must be an exact substring from the source text. "
        "Do not invent entities. Allowed labels: "
        f"{label_list}."
    )
    user_parts = [f"Allowed labels: {label_list}"]
    if label_definitions:
        user_parts.append(
            "Entity type definitions:\n" + _format_label_definitions(labels, label_definitions),
        )
    if few_shot_examples:
        formatted = _format_few_shot_examples(few_shot_examples)
        if formatted:
            user_parts.append(f"Few-shot examples:\n{formatted}")
    user_parts.append(f"Text:\n{text}")
    user_parts.append(
        f"Example response shape (replace values with entities from the text above):\n{example}",
    )
    user = "\n\n".join(user_parts)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

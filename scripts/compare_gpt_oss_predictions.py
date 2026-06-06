#!/usr/bin/env python3
"""Print gold vs LLM predictions for a small sample."""

from __future__ import annotations

from pathlib import Path

from ner_detector.detect import detect_entities
from ner_detector.env import load_project_env
from ner_detector.eval.loaders import load_dataset, resolve_benchmark_root

load_project_env()

MODEL = "openai/gpt-oss-120b:free"
LABELS = [
    "person",
    "organization",
    "location",
    "date",
    "number",
    "product",
    "model",
    "dataset",
    "benchmark",
    "metric",
    "method",
]
MAX_EXAMPLES = 2
CONFIG = Path("benchmark/config/llm_gpt_oss_smoke.yaml")


def _fmt_ent(label: str, text: str, start: int | None, end: int | None) -> str:
    span = f"[{start}:{end}]" if start is not None else ""
    return f"{label:14} {span:12} {text!r}"


def _entity_row(entity: object) -> tuple[str, str, int, int]:
    label = str(getattr(entity, "label"))
    text = str(getattr(entity, "text"))
    start = int(getattr(entity, "start"))
    end = int(getattr(entity, "end"))
    return label, text, start, end


def main() -> None:
    root = resolve_benchmark_root(None, config_path=CONFIG)

    for dataset in ("synthetic_news_100", "arxiv_gold"):
        print("=" * 72)
        print(f"DATASET: {dataset}  |  model: {MODEL}")
        print("=" * 72)

        examples = load_dataset(dataset, root=root, max_examples=MAX_EXAMPLES)
        for idx, example in enumerate(examples, 1):
            preds = detect_entities(
                example.text,
                backend="llm",
                model_id=MODEL,
                labels=LABELS,
                provider="openrouter",
                temperature=0,
                max_chars=8000,
            )

            print()
            print(f"--- Example {idx}/{len(examples)}  id={example.id} ---")
            preview = example.text.replace("\n", " ")
            suffix = "..." if len(preview) > 220 else ""
            print(f"Text: {preview[:220]}{suffix}")
            print()
            print("EXPECTED (gold):")
            gold_rows = sorted((_entity_row(g) for g in example.entities), key=lambda r: (r[2], r[3]))
            if gold_rows:
                for label, text, start, end in gold_rows:
                    print("  ", _fmt_ent(label, text, start, end))
            else:
                print("   (none)")

            print()
            print("PREDICTED (model):")
            pred_rows = sorted(
                (
                    (
                        p.label,
                        p.text,
                        p.start if p.start is not None else -1,
                        p.end if p.end is not None else -1,
                    )
                    for p in preds
                ),
                key=lambda r: (r[2], r[3]),
            )
            if pred_rows:
                for label, text, start, end in pred_rows:
                    print("  ", _fmt_ent(label, text, start, end))
            else:
                print("   (none)")

            gold_set = {(label, text.lower(), start, end) for label, text, start, end in gold_rows}
            pred_set = {(label, text.lower(), start, end) for label, text, start, end in pred_rows}
            tp = gold_set & pred_set
            fn = gold_set - pred_set
            fp = pred_set - gold_set

            print()
            print("MATCHES (exact span + label):")
            if tp:
                for label, _text, start, end in sorted(tp, key=lambda x: (x[2], x[3])):
                    print(f"  TP  {label:14} [{start}:{end}] {example.text[start:end]!r}")
            else:
                print("  (none)")
            if fn:
                print("MISSED (gold not predicted):")
                for label, _text, start, end in sorted(fn, key=lambda x: (x[2], x[3])):
                    print(f"  FN  {label:14} [{start}:{end}] {example.text[start:end]!r}")
            if fp:
                print("EXTRA (predicted not in gold):")
                for label, text, start, end in sorted(fp, key=lambda x: (x[2], x[3])):
                    surface = example.text[start:end] if 0 <= start < end <= len(example.text) else text
                    print(f"  FP  {label:14} [{start}:{end}] {surface!r}")
            print()


if __name__ == "__main__":
    main()

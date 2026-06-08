"""Hugging Face transformers token-classification NER."""

from __future__ import annotations

from ner_detector.backends.chunking import (
    _DEFAULT_CHUNK_OVERLAP,
    chunk_offset,
    chunk_text,
    merge_overlapping_entities,
)
from ner_detector.types import DetectedEntity

_DEFAULT_MAX_CHUNK_CHARS = 4000


class TransformersBackend:
    """BERT/RoBERTa-style fixed-label NER via ``pipeline('ner')``."""

    backend = "transformers"

    def __init__(
        self,
        model_id: str,
        *,
        max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
    ) -> None:
        self.model_id = model_id
        self.max_chunk_chars = max_chunk_chars
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            "ner",
            model=self.model_id,
            tokenizer=self.model_id,
            aggregation_strategy="simple",
            device=device,
        )

    def detect(
        self,
        text: str,
        *,
        labels: list[str] | None = None,
        threshold: float = 0.5,
    ) -> list[DetectedEntity]:
        del labels  # fixed label set from model head
        if not text.strip():
            return []
        self._ensure_loaded()
        assert self._pipeline is not None
        found: list[DetectedEntity] = []
        for chunk in chunk_text(
            text,
            max_chars=self.max_chunk_chars,
            overlap=_DEFAULT_CHUNK_OVERLAP,
        ):
            chunk_start = chunk_offset(text, chunk)
            for span in self._pipeline(chunk):
                score = float(span.get("score", 0.0))
                if score < threshold:
                    continue
                word = str(span.get("word", "")).strip()
                start = span.get("start")
                end = span.get("end")
                abs_start = chunk_start + int(start) if start is not None else None
                abs_end = chunk_start + int(end) if end is not None else None
                if (
                    abs_start is not None
                    and abs_end is not None
                    and 0 <= abs_start < abs_end <= len(text)
                ):
                    surface = text[abs_start:abs_end].strip()
                else:
                    surface = word.replace("##", "").strip()
                if not surface:
                    continue
                entity_group = str(span.get("entity_group", span.get("entity", "MISC")))
                found.append(
                    DetectedEntity(
                        text=surface,
                        label=entity_group,
                        score=round(score, 4),
                        start=abs_start,
                        end=abs_end,
                    ),
                )
        deduped: list[DetectedEntity] = []
        seen: set[tuple[int, int, str]] = set()
        for ent in found:
            if ent.start is None or ent.end is None:
                deduped.append(ent)
                continue
            key = (ent.start, ent.end, ent.label)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                DetectedEntity(
                    text=text[ent.start : ent.end],
                    label=ent.label,
                    score=ent.score,
                    start=ent.start,
                    end=ent.end,
                ),
            )
        return merge_overlapping_entities(deduped, source=text)

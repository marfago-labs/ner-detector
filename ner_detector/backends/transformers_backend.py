"""Hugging Face transformers token-classification NER."""

from __future__ import annotations

from ner_detector.types import DetectedEntity

_DEFAULT_MAX_CHUNK_CHARS = 4000
_CHUNK_OVERLAP_CHARS = 200


def _chunk_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= max_chars:
        return [stripped]
    chunks: list[str] = []
    start = 0
    while start < len(stripped):
        end = min(len(stripped), start + max_chars)
        chunks.append(stripped[start:end])
        if end >= len(stripped):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


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
        seen: set[tuple[int, int]] = set()
        for chunk in _chunk_text(
            text,
            max_chars=self.max_chunk_chars,
            overlap=_CHUNK_OVERLAP_CHARS,
        ):
            chunk_offset = text.find(chunk)
            if chunk_offset < 0:
                chunk_offset = 0
            for span in self._pipeline(chunk):
                score = float(span.get("score", 0.0))
                if score < threshold:
                    continue
                word = str(span.get("word", "")).strip()
                start = span.get("start")
                end = span.get("end")
                abs_start = chunk_offset + int(start) if start is not None else None
                abs_end = chunk_offset + int(end) if end is not None else None
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
                dedupe_key = (abs_start or 0, abs_end or 0)
                if dedupe_key in seen and abs_start is not None:
                    continue
                if abs_start is not None:
                    seen.add(dedupe_key)
                entity_group = str(span.get("entity_group", span.get("entity", "MISC")))
                found.append(
                    DetectedEntity(
                        text=surface,
                        label=entity_group,
                        score=round(score, 4),
                        start=abs_start,
                        end=abs_end,
                    )
                )
        return found

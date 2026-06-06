"""NuNER Zero zero-shot NER backend (token-classifier GLiNER variant)."""

from __future__ import annotations

from ner_detector.types import DetectedEntity

_DEFAULT_LABELS = ["person", "organization", "location", "date"]


def merge_token_entities(
    entities: list[dict],
    *,
    source: str,
) -> list[dict]:
    """Merge adjacent token-level spans with the same label (NuNER Zero convention)."""
    if not entities:
        return []
    ordered = sorted(entities, key=lambda item: (item.get("start", 0), item.get("end", 0)))
    merged: list[dict] = [dict(ordered[0])]
    for nxt in ordered[1:]:
        current = merged[-1]
        same_label = nxt.get("label") == current.get("label")
        cur_end = int(current.get("end", 0))
        nxt_start = int(nxt.get("start", 0))
        if same_label and nxt_start <= cur_end + 1:
            nxt_end = int(nxt.get("end", cur_end))
            start = int(current.get("start", 0))
            current["end"] = nxt_end
            current["text"] = source[start:nxt_end].strip()
            score = current.get("score")
            nxt_score = nxt.get("score")
            if score is not None and nxt_score is not None:
                current["score"] = max(float(score), float(nxt_score))
            continue
        merged.append(dict(nxt))
    return merged


class NunerBackend:
    """Open-vocabulary NER via NuNER Zero (arbitrary-length token classification)."""

    backend = "nuner"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from gliner import GLiNER

        self._model = GLiNER.from_pretrained(self.model_id)

    def detect(
        self,
        text: str,
        *,
        labels: list[str] | None = None,
        threshold: float = 0.5,
    ) -> list[DetectedEntity]:
        if not text.strip():
            return []
        self._ensure_loaded()
        assert self._model is not None
        entity_labels = labels if labels else list(_DEFAULT_LABELS)
        query_labels = [label.strip().lower() for label in entity_labels if label.strip()]
        raw = self._model.predict_entities(text, query_labels, threshold=threshold)
        merged = merge_token_entities(raw, source=text)
        found: list[DetectedEntity] = []
        for item in merged:
            score = item.get("score")
            label = str(item.get("label", "unknown"))
            found.append(
                DetectedEntity(
                    text=str(item.get("text", "")),
                    label=label,
                    score=round(float(score), 4) if score is not None else None,
                    start=item.get("start"),
                    end=item.get("end"),
                ),
            )
        return found

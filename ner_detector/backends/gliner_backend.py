"""GLiNER zero-shot NER backend."""

from __future__ import annotations

from ner_detector.types import DetectedEntity

_DEFAULT_LABELS = ["person", "organization", "location", "date"]


class GlinerBackend:
    """Open-vocabulary NER with user-supplied labels."""

    backend = "gliner"

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
        raw = self._model.predict_entities(text, entity_labels, threshold=threshold)
        found: list[DetectedEntity] = []
        for item in raw:
            score = item.get("score")
            found.append(
                DetectedEntity(
                    text=str(item.get("text", "")),
                    label=str(item.get("label", "unknown")),
                    score=round(float(score), 4) if score is not None else None,
                    start=item.get("start"),
                    end=item.get("end"),
                )
            )
        return found

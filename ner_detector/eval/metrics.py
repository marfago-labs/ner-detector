"""Span-level precision, recall, and F1 for NER evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ner_detector.eval.confusion import LabelConfusionMatrix, label_confusion_matrix
from ner_detector.eval.label_map import normalize_label
from ner_detector.eval.types import EvalSpan, GoldEntity, GoldExample
from ner_detector.types import DetectedEntity


def _overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter == 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def gold_to_span(entity: GoldEntity, *, label_map: str) -> EvalSpan:
    return EvalSpan(
        start=entity.start,
        end=entity.end,
        label=normalize_label(entity.label, label_map),
        text=entity.text,
    )


def prediction_to_span(
    text: str,
    entity: DetectedEntity,
    *,
    label_map: str,
) -> EvalSpan | None:
    surface = entity.text.strip()
    if not surface:
        return None
    start = entity.start
    end = entity.end
    if start is None or end is None:
        idx = text.find(surface)
        if idx < 0:
            idx = text.lower().find(surface.lower())
        if idx < 0:
            return None
        start = idx
        end = idx + len(surface)
    label = normalize_label(entity.label, label_map)
    return EvalSpan(start=start, end=end, label=label, text=surface)


def match_prediction_to_gold(
    pred: EvalSpan,
    gold: list[EvalSpan],
    matched_gold: set[int],
    *,
    mode: str,
) -> int | None:
    """Return index of first unmatched gold span that matches ``pred``, or None."""
    for i, g in enumerate(gold):
        if i in matched_gold:
            continue
        if mode == "document":
            if (g.label, g.text.strip().lower()) == (pred.label, pred.text.strip().lower()):
                return i
            continue
        if g.label != pred.label:
            continue
        if mode == "strict" and g.as_tuple() == pred.as_tuple():
            return i
        if mode == "relaxed" and _overlap_ratio(g.start, g.end, pred.start, pred.end) >= 0.5:
            return i
    return None


def count_gold_units(gold: list[EvalSpan], *, mode: str) -> int:
    """Number of gold units for recall denominator (spans or unique document keys)."""
    if mode == "document":
        return len({(g.label, g.text.strip().lower()) for g in gold})
    return len(gold)


def _match_counts(
    gold: list[EvalSpan],
    pred: list[EvalSpan],
    *,
    relaxed: bool,
) -> tuple[int, int, int]:
    """Return (tp, fp, fn) for one example."""
    matched_gold: set[int] = set()
    tp = 0
    for p in pred:
        found = False
        for i, g in enumerate(gold):
            if i in matched_gold:
                continue
            if g.label != p.label:
                continue
            if relaxed:
                if _overlap_ratio(g.start, g.end, p.start, p.end) >= 0.5:
                    found = True
                    matched_gold.add(i)
                    break
            elif g.as_tuple() == p.as_tuple():
                found = True
                matched_gold.add(i)
                break
        if found:
            tp += 1
    fp = len(pred) - tp
    fn = len(gold) - len(matched_gold)
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _match_document_counts(
    gold: list[EvalSpan],
    pred: list[EvalSpan],
) -> tuple[int, int, int]:
    """Calculate TP, FP, FN based on unique (label, lower(text)) sets per document."""
    gold_set = {(g.label, g.text.strip().lower()) for g in gold}
    pred_set = {(p.label, p.text.strip().lower()) for p in pred}
    
    tp = len(gold_set.intersection(pred_set))
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


@dataclass
class EntityScores:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: EntityScores) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def precision_recall_f1(self) -> tuple[float, float, float]:
        return _prf(self.tp, self.fp, self.fn)


@dataclass
class ScoreSummary:
    strict: EntityScores = field(default_factory=EntityScores)
    relaxed: EntityScores = field(default_factory=EntityScores)
    document: EntityScores = field(default_factory=EntityScores)
    n_examples: int = 0
    n_gold_spans: int = 0
    n_pred_spans: int = 0
    skipped_predictions: int = 0
    confusion_strict: LabelConfusionMatrix = field(default_factory=LabelConfusionMatrix.empty)
    confusion_relaxed: LabelConfusionMatrix = field(default_factory=LabelConfusionMatrix.empty)

    def merge(self, other: ScoreSummary) -> None:
        """Accumulate counts and confusion matrices from another summary."""
        self.strict.add(other.strict)
        self.relaxed.add(other.relaxed)
        self.document.add(other.document)
        self.n_examples += other.n_examples
        self.n_gold_spans += other.n_gold_spans
        self.n_pred_spans += other.n_pred_spans
        self.skipped_predictions += other.skipped_predictions
        self.confusion_strict = self.confusion_strict.merge(other.confusion_strict)
        self.confusion_relaxed = self.confusion_relaxed.merge(other.confusion_relaxed)

    def strict_prf(self) -> tuple[float, float, float]:
        return self.strict.precision_recall_f1()

    def relaxed_prf(self) -> tuple[float, float, float]:
        return self.relaxed.precision_recall_f1()

    def document_prf(self) -> tuple[float, float, float]:
        return self.document.precision_recall_f1()


def score_example(
    example: GoldExample,
    predictions: list[DetectedEntity],
    *,
    label_map: str = "unified",
) -> ScoreSummary:
    gold_spans = [gold_to_span(e, label_map=label_map) for e in example.entities]
    pred_spans: list[EvalSpan] = []
    skipped = 0
    for ent in predictions:
        span = prediction_to_span(example.text, ent, label_map=label_map)
        if span is None:
            skipped += 1
            continue
        pred_spans.append(span)

    strict = EntityScores(*_match_counts(gold_spans, pred_spans, relaxed=False))
    relaxed = EntityScores(*_match_counts(gold_spans, pred_spans, relaxed=True))
    document = EntityScores(*_match_document_counts(gold_spans, pred_spans))
    return ScoreSummary(
        strict=strict,
        relaxed=relaxed,
        document=document,
        n_examples=1,
        n_gold_spans=len(gold_spans),
        n_pred_spans=len(pred_spans),
        skipped_predictions=skipped,
        confusion_strict=label_confusion_matrix(gold_spans, pred_spans, mode="strict"),
        confusion_relaxed=label_confusion_matrix(gold_spans, pred_spans, mode="relaxed"),
    )

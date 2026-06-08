"""Label confusion matrices for NER benchmark error analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ner_detector.eval.types import EvalSpan

MISSED_COL = "∅ missed"
SPURIOUS_ROW = "∅ spurious"
SpanPairMode = Literal["strict", "relaxed"]


def _overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter == 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


@dataclass
class LabelConfusionMatrix:
    """Gold label (row) × predicted label (column) counts at paired spans."""

    counts: dict[tuple[str, str], int] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> LabelConfusionMatrix:
        return cls()

    def is_empty(self) -> bool:
        return not self.counts

    def total(self) -> int:
        return sum(self.counts.values())

    def increment(self, gold_label: str, pred_label: str, *, amount: int = 1) -> None:
        key = (gold_label, pred_label)
        self.counts[key] = self.counts.get(key, 0) + amount

    def merge(self, other: LabelConfusionMatrix) -> LabelConfusionMatrix:
        if other.is_empty():
            return self
        if self.is_empty():
            return LabelConfusionMatrix(counts=dict(other.counts))
        merged = dict(self.counts)
        for key, value in other.counts.items():
            merged[key] = merged.get(key, 0) + value
        return LabelConfusionMatrix(counts=merged)

    def row_labels(self) -> list[str]:
        rows = {g for g, _ in self.counts}
        ordered = sorted(r for r in rows if r not in {SPURIOUS_ROW})
        if SPURIOUS_ROW in rows:
            ordered.append(SPURIOUS_ROW)
        return ordered

    def col_labels(self) -> list[str]:
        cols = {p for _, p in self.counts}
        ordered = sorted(c for c in cols if c not in {MISSED_COL})
        if MISSED_COL in cols:
            ordered.append(MISSED_COL)
        return ordered

    def get(self, gold_label: str, pred_label: str) -> int:
        return self.counts.get((gold_label, pred_label), 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.row_labels(),
            "cols": self.col_labels(),
            "counts": [{"gold": g, "pred": p, "n": n} for (g, p), n in sorted(self.counts.items())],
            "total": self.total(),
        }


def _best_gold_match(
    pred: EvalSpan,
    gold: list[EvalSpan],
    matched_gold: set[int],
    *,
    mode: SpanPairMode,
) -> int | None:
    best_i: int | None = None
    best_iou = 0.0
    for i, g in enumerate(gold):
        if i in matched_gold:
            continue
        if mode == "strict":
            if g.start == pred.start and g.end == pred.end:
                return i
            continue
        iou = _overlap_ratio(g.start, g.end, pred.start, pred.end)
        if iou >= 0.5 and iou > best_iou:
            best_i = i
            best_iou = iou
    return best_i


def label_confusion_matrix(
    gold: list[EvalSpan],
    pred: list[EvalSpan],
    *,
    mode: SpanPairMode,
) -> LabelConfusionMatrix:
    """Pair spans and count gold×pred label co-occurrences.

    * ``relaxed`` — greedy one-to-one pairing by max IoU (≥ 0.5); labels may differ.
    * ``strict`` — pair only when ``start``/``end`` match exactly; labels may differ.

    Unpaired gold → column :data:`MISSED_COL`. Unpaired predictions → row
    :data:`SPURIOUS_ROW`.
    """
    matrix = LabelConfusionMatrix.empty()
    matched_gold: set[int] = set()

    for p in pred:
        match_i = _best_gold_match(p, gold, matched_gold, mode=mode)
        if match_i is None:
            matrix.increment(SPURIOUS_ROW, p.label)
            continue
        matched_gold.add(match_i)
        matrix.increment(gold[match_i].label, p.label)

    for i, g in enumerate(gold):
        if i not in matched_gold:
            matrix.increment(g.label, MISSED_COL)

    return matrix

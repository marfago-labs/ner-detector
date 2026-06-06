"""PR and ROC curves from scored NER predictions (threshold backends)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from ner_detector.eval.metrics import (
    count_gold_units,
    gold_to_span,
    match_prediction_to_gold,
    prediction_to_span,
)
from ner_detector.eval.types import EvalSpan, GoldExample
from ner_detector.types import DetectedEntity

MatchMode = Literal["strict", "relaxed", "document"]
MATCH_MODES: tuple[MatchMode, ...] = ("strict", "relaxed", "document")

DEFAULT_INFERENCE_THRESHOLD = 0.0


@dataclass(frozen=True, slots=True)
class RankedPrediction:
    example_idx: int
    span: EvalSpan
    score: float


@dataclass(frozen=True, slots=True)
class CurvePoint:
    recall: float
    precision: float
    threshold: float | None = None
    fpr: float | None = None
    tpr: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {
            "recall": round(self.recall, 6),
            "precision": round(self.precision, 6),
            "threshold": round(self.threshold, 6) if self.threshold is not None else None,
        }
        if self.fpr is not None:
            out["fpr"] = round(self.fpr, 6)
        if self.tpr is not None:
            out["tpr"] = round(self.tpr, 6)
        return out


from ner_detector.backends.families import uses_score_threshold as uses_threshold_backend


def _default_score(entity: DetectedEntity) -> float:
    return float(entity.score) if entity.score is not None else 0.0


def build_ranked_predictions(
    examples: list[GoldExample],
    predictions_per_example: list[list[DetectedEntity]],
    *,
    label_map: str = "unified",
) -> list[RankedPrediction]:
    """Flatten per-example predictions into score-sorted candidates."""
    if len(examples) != len(predictions_per_example):
        raise ValueError("examples and predictions_per_example length mismatch")
    ranked: list[RankedPrediction] = []
    for idx, (example, preds) in enumerate(zip(examples, predictions_per_example, strict=True)):
        for ent in preds:
            span = prediction_to_span(example.text, ent, label_map=label_map)
            if span is None:
                continue
            ranked.append(RankedPrediction(example_idx=idx, span=span, score=_default_score(ent)))
    return ranked


def _gold_spans_for_examples(
    examples: list[GoldExample],
    *,
    label_map: str,
) -> list[list]:
    return [
        [gold_to_span(e, label_map=label_map) for e in ex.entities] for ex in examples
    ]


def total_gold_units(
    examples: list[GoldExample],
    *,
    label_map: str,
    mode: MatchMode,
) -> int:
    gold_by_ex = _gold_spans_for_examples(examples, label_map=label_map)
    return sum(count_gold_units(g, mode=mode) for g in gold_by_ex)


def pr_curve_points(
    ranked: list[RankedPrediction],
    examples: list[GoldExample],
    *,
    label_map: str = "unified",
    mode: MatchMode = "strict",
) -> list[CurvePoint]:
    """
    Micro-averaged PR curve by sweeping score thresholds (high → low).

    Recall uses all gold units in the corpus; precision uses accepted predictions so far.
    """
    gold_by_ex = _gold_spans_for_examples(examples, label_map=label_map)
    total_gold = sum(count_gold_units(g, mode=mode) for g in gold_by_ex)
    matched: dict[int, set[int]] = defaultdict(set)
    ordered = sorted(
        ranked,
        key=lambda r: (-r.score, r.example_idx, r.span.start, r.span.end),
    )

    points: list[CurvePoint] = [CurvePoint(recall=0.0, precision=1.0, threshold=None)]
    tp = 0
    fp = 0
    for pred in ordered:
        gold = gold_by_ex[pred.example_idx]
        idx = match_prediction_to_gold(
            pred.span,
            gold,
            matched[pred.example_idx],
            mode=mode,
        )
        if idx is not None:
            matched[pred.example_idx].add(idx)
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / total_gold if total_gold else 0.0
        points.append(
            CurvePoint(recall=recall, precision=precision, threshold=pred.score),
        )
    return points


def roc_curve_points(
    ranked: list[RankedPrediction],
    examples: list[GoldExample],
    *,
    label_map: str = "unified",
    mode: MatchMode = "strict",
) -> list[CurvePoint]:
    """
    Proposal-level ROC: each scored candidate span is one trial (1 = matches gold).

    TPR = TP / (TP + FN) among proposals accepted so far is NOT used; instead
    TPR = TP / total_proposal_positives where total_proposal_positives is the count
    of proposals that match gold when the full ranked list is traversed in order.
    FPR = FP / total_proposal_negatives among accepted proposals.
    """
    gold_by_ex = _gold_spans_for_examples(examples, label_map=label_map)
    matched: dict[int, set[int]] = defaultdict(set)
    ordered = sorted(
        ranked,
        key=lambda r: (-r.score, r.example_idx, r.span.start, r.span.end),
    )

    labels: list[int] = []
    for pred in ordered:
        gold = gold_by_ex[pred.example_idx]
        idx = match_prediction_to_gold(
            pred.span,
            gold,
            matched[pred.example_idx],
            mode=mode,
        )
        if idx is not None:
            matched[pred.example_idx].add(idx)
            labels.append(1)
        else:
            labels.append(0)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    points: list[CurvePoint] = [
        CurvePoint(recall=0.0, precision=1.0, threshold=None, fpr=0.0, tpr=0.0),
    ]
    if not labels:
        return points

    tp = 0
    fp = 0
    for label, pred in zip(labels, ordered, strict=True):
        if label:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        points.append(
            CurvePoint(
                recall=0.0,
                precision=0.0,
                threshold=pred.score,
                fpr=fpr,
                tpr=tpr,
            ),
        )
    return points


def trapezoid_auc(xs: list[float], ys: list[float]) -> float:
    """Area under curve via trapezoidal rule on sorted x."""
    if len(xs) < 2:
        return 0.0
    pairs = sorted(zip(xs, ys, strict=True), key=lambda p: p[0])
    area = 0.0
    for i in range(1, len(pairs)):
        x0, y0 = pairs[i - 1]
        x1, y1 = pairs[i]
        area += (x1 - x0) * (y0 + y1) / 2.0
    return max(0.0, min(1.0, area))


def auc_pr(points: list[CurvePoint]) -> float:
    """Area under PR curve (recall on x-axis, precision on y-axis)."""
    pr = [(p.recall, p.precision) for p in points if p.recall is not None]
    return trapezoid_auc([r for r, _ in pr], [p for _, p in pr])


def auc_roc(points: list[CurvePoint]) -> float:
    """Area under ROC curve (FPR on x-axis, TPR on y-axis)."""
    roc = [(p.fpr, p.tpr) for p in points if p.fpr is not None and p.tpr is not None]
    if len(roc) < 2:
        return 0.0
    return trapezoid_auc([x for x, _ in roc], [y for _, y in roc])


def point_at_threshold(
    points: list[CurvePoint],
    threshold: float,
) -> CurvePoint | None:
    """Last curve point with ``threshold >=`` operating threshold (predictions kept)."""
    eligible = [p for p in points if p.threshold is not None and p.threshold >= threshold]
    return eligible[-1] if eligible else None


def curves_for_run(
    examples: list[GoldExample],
    predictions_per_example: list[list[DetectedEntity]],
    *,
    label_map: str = "unified",
    operating_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute PR/ROC curves and AUCs for all match modes."""
    ranked = build_ranked_predictions(
        examples, predictions_per_example, label_map=label_map
    )
    modes_out: dict[str, Any] = {}
    for mode in MATCH_MODES:
        pr_pts = pr_curve_points(ranked, examples, label_map=label_map, mode=mode)
        roc_pts = roc_curve_points(ranked, examples, label_map=label_map, mode=mode)
        op = point_at_threshold(pr_pts, operating_threshold)
        modes_out[mode] = {
            "pr": [p.to_dict() for p in pr_pts],
            "roc": [p.to_dict() for p in roc_pts],
            "auc_pr": round(auc_pr(pr_pts), 6),
            "auc_roc": round(auc_roc(roc_pts), 6),
            "n_candidates": len(ranked),
            "n_gold": total_gold_units(examples, label_map=label_map, mode=mode),
            "operating_point": op.to_dict() if op else None,
        }
    return modes_out

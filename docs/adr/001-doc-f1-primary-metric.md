# ADR 001: Document F1 as primary benchmark metric

**Status:** Accepted
**Date:** 2026-06
**Context:** [benchmarks.md](../benchmarks.md), [metrics.py](../../ner_detector/eval/metrics.py)

## Context

NER benchmarks in this repo use **salient-entity gold**: annotators mark important entities only, not every possible mention. Predictions that extract valid but unlabeled spans are penalized as false positives under span-level metrics. Leaderboards need a single primary score that reflects *useful extraction* on this gold style without over-penalizing surface-form variation.

## Decision

Use **document-level F1** (Doc F1) as the **primary** leaderboard metric in reports and portfolio copy.

Doc F1 treats each document as a set of `(label, normalized_text)` pairs (text lowercased). A prediction matches gold if label and normalized surface agree—**character offsets are not required**.

## Alternatives considered

| Metric | When to use | Why not primary |
|--------|-------------|-----------------|
| **Strict span F1** | Offset-sensitive tasks, CoNLL-style fully labeled corpora | Salient gold + unlabeled mentions inflate FP; small offset drift counts as miss |
| **Relaxed span F1** (IoU ≥ 0.5) | Boundary-tolerant span evaluation | Still span-pair based; Doc F1 is simpler for salient sets and cross-backend comparison |
| **Micro span F1 only** | Fully annotated corpora | Misleading on sparse `arxiv_gold` (~4 entities/abstract) |

Relaxed and strict span F1 remain in reports for diagnosis; Doc F1 drives the headline comparison.

## Consequences

- HTML/Markdown reports sort by Doc F1 first ([report.py](../../ner_detector/eval/report.py)).
- Backend choice narrative (LLM vs BERT vs GLiNER) uses Doc F1 + latency ([README.md](../../README.md)).
- Readers must understand Doc F1 semantics before comparing to external benchmarks that use strict span F1 on fully labeled data.

## References

- Implementation: `document_f1` in [metrics.py](../../ner_detector/eval/metrics.py)
- IoU threshold for relaxed F1: `RELAXED_SPAN_IOU_THRESHOLD = 0.5`

# NER backend benchmark report

Config: `benchmark/config/compare_backends.yaml`
Output: `benchmark/results/latest`

## Summary (document-level string overlap F1)

| Run | Dataset | Backend | Model | Doc F1 | P | R | Latency (ms/ex) |
|-----|---------|---------|-------|--------|---|---|-----------------|
| pattern | synthetic_news_100 | pattern | — | 24.2% | 21.0% | 28.6% | 0.17 |
| pattern | arxiv_gold | pattern | — | 2.2% | 2.0% | 2.4% | 0.89 |

## Summary (strict span F1)

| Run | Dataset | Backend | Model | F1 | P | R | Latency (ms/ex) |
|-----|---------|---------|-------|-----|---|---|-----------------|
| pattern | synthetic_news_100 | pattern | — | 23.9% | 20.5% | 28.6% | 0.17 |
| pattern | arxiv_gold | pattern | — | 1.7% | 1.3% | 2.4% | 0.89 |

## Summary (relaxed span F1, ≥50% overlap)

| Run | Dataset | F1 | P | R | TP | FP | FN |
|-----|---------|-----|---|---|----|----|-----|
| pattern | arxiv_gold | 5.0% | 3.8% | 7.3% | 3 | 76 | 38 |
| pattern | synthetic_news_100 | 28.2% | 24.2% | 33.8% | 169 | 530 | 331 |

## Notes

- **Document (Doc F1)**: per example, build sets of `(label, lowercased text)` for gold and predictions; TP = intersection, FP = predictions not in gold, FN = gold not in predictions; micro-averaged P/R/F1. Duplicate spans with the same label and string count once. Primary leaderboard metric — suited to salient gold and concept-style extraction (e.g. recommendation tags).
- **Strict**: exact `(start, end, label)` after label normalization; one span per match.
- **Relaxed**: same label and span overlap ratio ≥ 0.5.
- Compare backends only on datasets whose gold labels match the run (see `benchmark/config/label_maps.yaml`).
- `pattern` is intended for domain gold with arxiv IDs and years (e.g. synthetic corpora), not CoNLL entity types.

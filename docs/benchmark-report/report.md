# NER backend benchmark report

Config: `benchmark\config\compare_backends.yaml`
Output: `benchmark\results\run-20260606T131135Z`

## Summary (document-level string overlap F1)

| Run | Dataset | Backend | Model | Doc F1 | P | R | Latency (ms/ex) |
|-----|---------|---------|-------|--------|---|---|-----------------|
| llm-gpt-oss | synthetic_news_100 | llm | openai/gpt-oss-120b:free | 83.9% | 75.3% | 94.6% | 6905.76 |
| nuner-zero | synthetic_news_100 | nuner | numind/NuNER_Zero | 78.0% | 70.4% | 87.4% | 1860.95 |
| bert-conll | synthetic_news_100 | transformers | dslim/bert-base-NER | 72.5% | 78.4% | 67.4% | 79.60 |
| gliner-medium | synthetic_news_100 | gliner | urchade/gliner_medium-v2.1 | 69.6% | 58.6% | 85.8% | 197.21 |
| gliner-bi-large | synthetic_news_100 | gliner | knowledgator/gliner-bi-large-v2.0 | 59.2% | 88.8% | 44.4% | 364.32 |
| llm-gpt-oss | arxiv_gold | llm | openai/gpt-oss-120b:free | 47.2% | 38.5% | 61.0% | 9052.61 |
| gliner-bi-large | arxiv_gold | gliner | knowledgator/gliner-bi-large-v2.0 | 39.1% | 37.0% | 41.5% | 664.38 |
| gliner-medium | arxiv_gold | gliner | urchade/gliner_medium-v2.1 | 36.2% | 32.1% | 41.5% | 364.31 |
| llm-mock | synthetic_news_100 | llm | mock/ner | 33.1% | 35.1% | 31.2% | 0.21 |
| nuner-zero | arxiv_gold | nuner | numind/NuNER_Zero | 32.7% | 28.1% | 39.0% | 840.31 |
| pattern | synthetic_news_100 | pattern | — | 24.2% | 21.0% | 28.6% | 0.14 |
| llm-mock | arxiv_gold | llm | mock/ner | 2.9% | 3.4% | 2.4% | 0.21 |
| pattern | arxiv_gold | pattern | — | 2.2% | 2.0% | 2.4% | 0.72 |

## Summary (strict span F1)

| Run | Dataset | Backend | Model | F1 | P | R | Latency (ms/ex) |
|-----|---------|---------|-------|-----|---|---|-----------------|
| llm-gpt-oss | synthetic_news_100 | llm | openai/gpt-oss-120b:free | 83.9% | 75.3% | 94.6% | 6905.76 |
| nuner-zero | synthetic_news_100 | nuner | numind/NuNER_Zero | 77.9% | 70.3% | 87.4% | 1860.95 |
| bert-conll | synthetic_news_100 | transformers | dslim/bert-base-NER | 72.5% | 78.4% | 67.4% | 79.60 |
| gliner-medium | synthetic_news_100 | gliner | urchade/gliner_medium-v2.1 | 69.6% | 58.5% | 85.8% | 197.21 |
| gliner-bi-large | synthetic_news_100 | gliner | knowledgator/gliner-bi-large-v2.0 | 59.2% | 88.8% | 44.4% | 364.32 |
| llm-gpt-oss | arxiv_gold | llm | openai/gpt-oss-120b:free | 45.3% | 36.9% | 58.5% | 9052.61 |
| llm-mock | synthetic_news_100 | llm | mock/ner | 33.1% | 35.1% | 31.2% | 0.21 |
| gliner-bi-large | arxiv_gold | gliner | knowledgator/gliner-bi-large-v2.0 | 27.5% | 22.1% | 36.6% | 664.38 |
| gliner-medium | arxiv_gold | gliner | urchade/gliner_medium-v2.1 | 27.1% | 20.8% | 39.0% | 364.31 |
| pattern | synthetic_news_100 | pattern | — | 23.9% | 20.5% | 28.6% | 0.14 |
| nuner-zero | arxiv_gold | nuner | numind/NuNER_Zero | 22.6% | 16.9% | 34.1% | 840.31 |
| llm-mock | arxiv_gold | llm | mock/ner | 2.9% | 3.4% | 2.4% | 0.21 |
| pattern | arxiv_gold | pattern | — | 1.7% | 1.3% | 2.4% | 0.72 |

## Summary (relaxed span F1, ≥50% overlap)

| Run | Dataset | F1 | P | R | TP | FP | FN |
|-----|---------|-----|---|---|----|----|-----|
| pattern | arxiv_gold | 5.0% | 3.8% | 7.3% | 3 | 76 | 38 |
| pattern | synthetic_news_100 | 28.2% | 24.2% | 33.8% | 169 | 530 | 331 |
| bert-conll | synthetic_news_100 | 78.5% | 84.9% | 73.0% | 365 | 65 | 135 |
| gliner-medium | arxiv_gold | 32.2% | 24.7% | 46.3% | 19 | 58 | 22 |
| gliner-medium | synthetic_news_100 | 72.2% | 60.7% | 89.0% | 445 | 288 | 55 |
| gliner-bi-large | arxiv_gold | 34.9% | 27.9% | 46.3% | 19 | 49 | 22 |
| gliner-bi-large | synthetic_news_100 | 59.2% | 88.8% | 44.4% | 222 | 28 | 278 |
| nuner-zero | arxiv_gold | 29.0% | 21.7% | 43.9% | 18 | 65 | 23 |
| nuner-zero | synthetic_news_100 | 80.4% | 72.5% | 90.2% | 451 | 171 | 49 |
| llm-mock | arxiv_gold | 8.6% | 10.3% | 7.3% | 3 | 26 | 38 |
| llm-mock | synthetic_news_100 | 36.0% | 38.3% | 34.0% | 170 | 274 | 330 |
| llm-gpt-oss | arxiv_gold | 54.7% | 44.6% | 70.7% | 29 | 36 | 12 |
| llm-gpt-oss | synthetic_news_100 | 84.2% | 75.6% | 95.0% | 475 | 153 | 25 |

## Notes

- **Document (Doc F1)**: per example, build sets of `(label, lowercased text)` for gold and predictions; TP = intersection, FP = predictions not in gold, FN = gold not in predictions; micro-averaged P/R/F1. Duplicate spans with the same label and string count once. Primary leaderboard metric — suited to salient gold and concept-style extraction (e.g. recommendation tags).
- **Strict**: exact `(start, end, label)` after label normalization; one span per match.
- **Relaxed**: same label and span overlap ratio ≥ 0.5.
- Compare backends only on datasets whose gold labels match the run (see `benchmark/config/label_maps.yaml`).
- `pattern` is intended for domain gold with arxiv IDs and years (e.g. synthetic corpora), not CoNLL entity types.

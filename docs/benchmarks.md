# Benchmarks

Compare **pattern**, **transformers**, and **gliner** backends on gold JSONL datasets. Produces `metrics.json` and `report.md`.

Copy `.env.example` to `.env` (or use the repo `.env`) to set `TRANSFORMERS_VERBOSITY=error` and reduce Hugging Face load noise during ML benchmarks. **No keys** are required for `--pattern-only`; live LLM runs need `OPENROUTER_API_KEY` — see [configuration.md](configuration.md#which-keys-you-need).

## Quick run (no ML download)

```bash
cd ner-detector
uv sync --extra dev
uv run python benchmark/run_benchmark.py --pattern-only
```

Requires the sibling **[ner-dataset](https://github.com/marfago-labs/ner-dataset)** repo at `../ner-dataset/` (default in the marfago-labs monorepo), or set `NER_DATASET_DIR` to a `datasets/` directory.

Output: `benchmark/results/run-<timestamp>/`

- `report.md` — tables
- `report.html` — **static SVG radar charts**, **label confusion matrices**, and leaderboard tables per section
- `metrics.json` — machine-readable scores

## Full comparison (downloads models)

```bash
uv sync --extra dev --extra ml --extra gliner
uv run python benchmark/run_benchmark.py
```

## Configuration

| File | Role |
|------|------|
| `benchmark/config/compare_backends.yaml` | Which backends and datasets to run |
| `benchmark/config/compare_generated.yaml` | All six gold files from `ner-dataset` |
| `benchmark/config/label_maps.yaml` | Label normalization for scoring |
| `../ner-dataset/datasets/*.jsonl` | **Canonical gold** from [ner-gold-generator](https://github.com/marfago-labs/ner-gold-generator) |

Benchmark YAML may set `benchmark_root: ../ner-dataset` (default resolution when omitted). Override with `NER_DATASET_DIR` pointing at a `datasets/` folder (used in CI when `ner-dataset` is checked out into the workspace).

### Gold JSONL format

```json
{
  "id": "ex-001",
  "text": "Alice Smith joined OpenAI in 2024.",
  "entities": [
    {"text": "Alice Smith", "label": "person", "start": 0, "end": 11}
  ]
}
```

### Datasets (ner-dataset)

| Name | Description |
|------|-------------|
| `arxiv_gold` | 10 ML paper abstracts (models, datasets, benchmarks, metrics, methods) |
| `synthetic_news_100` | LLM-generated news-style text (person, organization, location, date) |
| `synthetic_blog_100` | LLM-generated blog-style corpus |
| `synthetic_scientific_100` | LLM-generated scientific-style corpus |
| `synthetic_transcript_100` | LLM-generated transcript-style corpus |
| `synthetic_mixed_100` | LLM-generated mixed-domain corpus |

`arxiv_gold` is built by the sibling repo **[ner-gold-generator](https://github.com/marfago-labs/ner-gold-generator)** into **[ner-dataset](https://github.com/marfago-labs/ner-dataset)** by default:

```bash
cd ../ner-gold-generator
uv sync --extra dev
uv run build-arxiv-gold   # → ../ner-dataset/datasets/arxiv_gold.jsonl
```

`load_dataset("arxiv_gold")` checks `NER_DATASET_DIR`, then `../ner-dataset/datasets/`.

The same tool can build gold from **YouTube** transcripts, a **folder** of `.txt`/`.md`/`.json` (blogs, docs), or a **JSONL** corpus (`uv run build-gold --source …`). See [ner-gold-generator docs](https://github.com/marfago-labs/ner-gold-generator/blob/main/docs/README.md).

Add more gold files under `ner-dataset/datasets/`. For CoNLL-2003 exports, use Hugging Face [`eriktks/conll2003`](https://huggingface.co/datasets/eriktks/conll2003) and convert to the JSONL schema above.

Small fixture corpora used only by unit tests live under `tests/fixtures/datasets/`.

## CLI

```bash
uv run python benchmark/run_benchmark.py --help
```

| Flag | Description |
|------|-------------|
| `--config` | Benchmark YAML |
| `--output` | Results directory |
| `--datasets` | Subset of datasets (comma-separated) |
| `--runs` | Subset of run names |
| `--max-examples` | Cap examples per dataset |
| `--pattern-only` | Skip ML backends |
| `--repeats` | Run each backend×dataset N times (default **1**; use N>1 for latency variance) |

## Metrics

The benchmark reports three F1 scores (micro-averaged over all examples). Full definitions also appear in the HTML report **Metrics & methodology** tab.

| Metric | Match rule |
|--------|------------|
| **Doc F1** (primary leaderboard) | Same unified `(label, lowercased text)`; span offsets ignored |
| **Strict F1** | Same label and exact `(start, end)` character offsets |
| **Relaxed F1** | Same label and span **IoU ≥ 0.5** |

- **Latency**: ms per example (wall clock per trial). With `--repeats N`, reports **mean ± std** and min–max across trials (model cache cleared between repeat rounds, not between datasets).
- **Score stability**: strict F1 must match across repeats; unstable runs are flagged in the report.

### Relaxed span F1 (IoU ≥ 0.5)

**IoU** (intersection over union) compares gold and predicted spans as half-open intervals `[start, end)` on the document text:

```
IoU = length(overlap) / length(union)
```

A prediction counts as a true positive when it has the **same normalized label** as gold and `IoU ≥ 0.5`. That threshold is the usual partial-credit bar in span NER eval: boundaries may differ slightly (whitespace, tokenization) but most of the entity must align.

**Example** — text `"OpenAI"` (6 characters):

| Gold | Prediction | IoU | Strict | Relaxed |
|------|------------|-----|--------|---------|
| `[0, 6)` org | `[0, 5)` org | 5/6 ≈ 0.83 | miss | match |
| `[0, 6)` org | `[1, 6)` org | 5/6 ≈ 0.83 | miss | match |
| `[0, 6)` org | `[0, 3)` org | 3/6 = 0.50 | miss | match |
| `[0, 6)` org | `[0, 2)` org | 2/6 ≈ 0.33 | miss | miss |

Pairing is **greedy one-to-one**: each prediction matches at most one gold span (first eligible gold in list order). Unmatched predictions are false positives; unmatched gold is false negatives. Label confusion matrices use the same IoU threshold; relaxed matrices pair by **maximum IoU** per prediction instead of first-fit (see `ner_detector/eval/confusion.py`).

Implementation: `ner_detector/eval/metrics.py` (`RELAXED_SPAN_IOU_THRESHOLD`, `_overlap_ratio`).

## Interpreting results

- **`pattern`** on synthetic corpora: regex baseline (arxiv_id, years, capitalized names where present).
- **`bert-conll`** on `synthetic_news_100`: strong on person/organization/location-style gold.
- **`gliner-medium`**: zero-shot; sensitive to `threshold` and `labels` in config.
- **`gliner-bi-large`**: same `gliner` backend and `predict_entities` API; use a **lower threshold** (e.g. `0.3`) than uni-encoder models — bi-encoder logits are calibrated lower, so `0.5` collapses recall.
- **`llm-mock`**: deterministic LLM backend (`provider: mock`) for offline benchmarks; uses pattern-style extraction wrapped as JSON. No PR/ROC curves (no score threshold sweep).
- **`llm-nemotron-super`**: live OpenRouter with **`nvidia/nemotron-3-super-120b-a12b:free`** (Nemotron 3 Super 120B, free tier). Requires `OPENROUTER_API_KEY` in `.env` and `uv sync --extra llm`. (`nvidia/nemotron-3-ultra:free` is listed on the site but not yet a valid API model ID.)

**`arxiv_gold` label schema:** gold uses scientific types (`model`, `dataset`, `benchmark`, `metric`, `method`, `number`, …), not CoNLL PER/ORG/LOC. `dslim/bert-base-NER` is **not benchmarked on `arxiv_gold`** (see per-run `datasets` in YAML); on synthetic corpora it scores normally. GLiNER must include the scientific label strings in config (see `compare_generated.yaml`). Expect modest strict F1 on `arxiv_gold` even when GLiNER is configured correctly — gold spans are often longer than model predictions.

Do not rank backends on a single F1 number without matching label schemes and datasets.

## Threshold curves (PR / ROC)

For backends that filter by score (`transformers`, `gliner`), each benchmark run also writes:

- `curves.json` — PR/ROC points and AUC per run×dataset (`strict`, `relaxed`, `document`)
- `curves/section.html` — SVG chart fragment
- HTML report tab **Threshold curves** (unless `--no-curves`)

Inference uses **`threshold: 0`** once per backend×dataset; curves sweep candidate scores from high to low. **PR** recall is micro-averaged over gold units. **ROC** is proposal-level (how well scores rank correct vs incorrect *proposed* spans). Use PR to pick an operating threshold; bi-encoder models typically need a lower cutoff than uni-encoder GLiNER.

## Python API

```python
from pathlib import Path
from ner_detector.eval import run_benchmark
from ner_detector.eval.report import write_report

benchmark = run_benchmark(
    Path("benchmark/config/compare_backends.yaml"),
    Path("benchmark/results/my-run"),
    run_names=["pattern"],
)
write_report(benchmark)  # metrics.json, report.md, report.html
```

Open `report.html` (or `index.html`, same content) in a browser — light theme, static SVG charts, no JavaScript required.
The results tab includes a **section index**: **Global (all datasets)** plus one block per gold file (leaderboard + radar + confusion matrices each).
Radar charts plot **F1, precision, recall, and Speed** (Speed = `1 − ms/example÷1000`, clamped 0–1, with **1 s/example as the zero point**). Absolute ms/example is also in the leaderboard table.
Use the **Metrics & methodology** tab for scoring definitions, benchmark process, and caveats (same tab pattern as text-compressor compare reports).

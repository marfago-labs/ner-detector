# Benchmarks

Compare **pattern**, **transformers**, and **gliner** backends on gold JSONL datasets. Produces `metrics.json` and `report.md`.

Copy `.env.example` to `.env` (or use the repo `.env`) to set `TRANSFORMERS_VERBOSITY=error` and reduce Hugging Face load noise during ML benchmarks.

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
| `synthetic_news_100` | Procedural news-style text (person, organization, location, date) |
| `synthetic_blog_100` | Blog-style synthetic corpus |
| `synthetic_scientific_100` | Scientific-style synthetic corpus |
| `synthetic_transcript_100` | Transcript-style synthetic corpus |
| `synthetic_mixed_100` | Mixed-domain synthetic corpus |

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

- **Strict F1**: exact character span + unified label
- **Relaxed F1**: ≥50% span overlap + same label
- **Latency**: ms per example (wall clock per trial). With `--repeats N`, reports **mean ± std** and min–max across trials (model cache cleared between repeat rounds, not between datasets).
- **Score stability**: strict F1 must match across repeats; unstable runs are flagged in the report.

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

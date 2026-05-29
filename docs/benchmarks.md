# Benchmarks

Compare **pattern**, **transformers**, and **gliner** backends on gold JSONL datasets. Produces `metrics.json` and `report.md`.

Copy `.env.example` to `.env` (or use the repo `.env`) to set `TRANSFORMERS_VERBOSITY=error` and reduce Hugging Face load noise during ML benchmarks.

## Quick run (no ML download)

```bash
cd ner-detector
uv sync --extra dev
uv run python benchmark/run_benchmark.py --pattern-only
```

Output: `benchmark/results/run-<timestamp>/`

- `report.md` — tables
- `report.html` — **static SVG radar charts** (text-compressor style) per dataset + leaderboard table
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
| `benchmark/config/label_maps.yaml` | Label normalization for scoring |
| `benchmark/datasets/*.jsonl` | Shipped gold annotations (CI / offline) |
| `../ner-dataset/datasets/*.jsonl` | Canonical gold from [ner-gold-generator](https://github.com/marfago-labs/ner-gold-generator) (preferred when present) |

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

### Datasets included

| Name | Description |
|------|-------------|
| `marfago_gold` | Domain snippets (person, org, location, arxiv_id, year, …) |
| `conll_dev_sample` | Short CoNLL-style English news sentences |
| `arxiv_gold` | 10 ML paper abstracts (models, datasets, benchmarks, metrics, methods) |

`arxiv_gold` is built by the sibling repo **[ner-gold-generator](https://github.com/marfago-labs/ner-gold-generator)** into **[ner-dataset](https://github.com/marfago-labs/ner-dataset)** by default:

```bash
cd ../ner-gold-generator
uv sync --extra dev
uv run build-arxiv-gold   # → ../ner-dataset/datasets/arxiv_gold.jsonl
```

`load_dataset("arxiv_gold")` checks `NER_DATASET_DIR`, then `../ner-dataset/datasets/`, then `benchmark/datasets/` (shipped copy).

The same tool can build gold from **YouTube** transcripts, a **folder** of `.txt`/`.md`/`.json` (blogs, docs), or a **JSONL** corpus (`uv run build-gold --source …`). See [ner-gold-generator docs](https://github.com/marfago-labs/ner-gold-generator/blob/main/docs/README.md).

Add more gold files under `ner-dataset/datasets/` (or `benchmark/datasets/` for small bundled sets). For CoNLL-2003 exports, use Hugging Face [`eriktks/conll2003`](https://huggingface.co/datasets/eriktks/conll2003) and convert to the JSONL schema above.

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
| `--repeats` | Run each backend×dataset N times (default **5**; mean±std latency, score stability) |

## Metrics

- **Strict F1**: exact character span + unified label
- **Relaxed F1**: ≥50% span overlap + same label
- **Latency**: ms per example (wall clock per trial). With `--repeats N`, reports **mean ± std** and min–max across trials (cache cleared each repeat).
- **Score stability**: strict F1 must match across repeats; unstable runs are flagged in the report.

## Interpreting results

- **`pattern`** on `marfago_gold`: regex baseline (arxiv_id, years, capitalized names).
- **`bert-conll`** on `conll_dev_sample`: strong on PER/ORG/LOC/MISC-style gold.
- **`gliner-medium`**: zero-shot; sensitive to `threshold` and `labels` in config.

Do not rank backends on a single F1 number without matching label schemes and datasets.

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

Open `report.html` in a browser — light theme, static SVG charts, no JavaScript required.
Use the **Metrics & methodology** tab for scoring definitions, benchmark process, and caveats (same tab pattern as text-compressor compare reports).

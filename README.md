# ner-detector

Named entity recognition (NER) with **pluggable backends**: deterministic regex (`pattern`), fixed-label BERT (`transformers`), zero-shot GLiNER (`gliner`), and chat LLM (`llm` via OpenRouter or mock).

**[📊 View the Live Benchmark Report](https://marfago-labs.github.io/ner-detector/)**

Part of [marfago-labs](https://github.com/marfago-labs). Standalone experiment for entity extraction and evaluation pipelines.

## Architecture & Benchmark Trade-offs

This framework rigorously evaluates multiple NLP paradigms on both synthetic procedural data and dense salient-concept scientific abstracts. Key findings from our benchmarking:

- **LLMs (e.g., GPT-4 class via OpenRouter):** Achieved the highest extraction quality (84% Doc F1 on synthetic news, ~50% on sparse scientific abstracts). However, inference latency is extremely high (~7–9 seconds per document), making them ideal for offline batched extraction where accuracy outweighs speed.
- **Transformers (e.g., BERT-base-NER):** Extremely fast (~80ms per document) with strong accuracy (72% Doc F1) on standard entity types (PER, ORG, LOC). This remains the optimal choice for real-time processing on standard schemas.
- **Zero-shot (e.g., GLiNER):** Offers the best middle ground for custom schemas (e.g., extracting scientific concepts like datasets and metrics) without needing retraining. `gliner-medium` provides solid extraction at ~200-400ms per document.

## Quick start

```bash
cd ner-detector
uv sync --extra dev
# Uses config/ner.yaml (default backend: pattern)
uv run python run.py "Alice Smith works at OpenAI in San Francisco."

# Or set backend/model in config/ner.yaml, then:
uv run python run.py --show-config
uv run python run.py "Alice Smith works at OpenAI."
```

Edit **`config/ner.yaml`** to choose backend and model; CLI flags override the file.

Install ML backends as needed:

```bash
uv sync --extra dev --extra ml          # transformers + torch
uv sync --extra dev --extra ml --extra gliner
uv sync --extra dev --extra llm         # OpenRouter HTTP client (mock works without API key)
```

## Configuration

Set backend and model in **`config/ner.yaml`** (or `NER_CONFIG_PATH`). CLI flags override the file.

```yaml
backend: transformers
model_id: dslim/bert-base-NER
threshold: 0.5
label_preset: general_en   # GLiNER labels when labels: is omitted
```

```bash
uv run python run.py --show-config    # inspect resolved settings
uv run python run.py -c ./my.yaml "text"   # custom profile path
```

See [docs/configuration.md](docs/configuration.md).

## CLI

| Flag | Description |
|------|-------------|
| `--config` / `-c` | Runtime profile YAML (default: `config/ner.yaml`) |
| `--show-config` | Print resolved backend/model/labels and exit |
| `--backend` | Override `backend` from profile |
| `--model` | Override `model_id` from profile |
| `--labels` | Override labels (comma-separated) |
| `--threshold` | Override minimum score |
| `--file` | Read input from a text file |
| `--format` | `json` or `text` |
| `--list-models` | Print model catalog (`default_models.yaml`) |

```bash
# Profile-driven (edit config/ner.yaml first)
uv run python run.py "Bill Gates founded Microsoft in Redmond."

# One-off overrides
uv run python run.py -b gliner -l "person,company,city" "Bill Gates founded Microsoft."

# Stdin
echo "Paper 2402.15343 by NuMind." | uv run python run.py
```

## Python API

```python
from ner_detector import detect_entities

entities = detect_entities(
    "Alice Smith joined OpenAI in 2024.",
    backend="pattern",
)
for e in entities:
    print(e.text, e.label, e.score)
```

## Model choice

See **[docs/models.md](docs/models.md)** for a researched comparison (Hugging Face, GitHub, Context7) and recommendations by use case.

**Catalog defaults** (used when `model_id` is omitted in `ner.yaml`; see `config/default_models.yaml`):

| Backend | Default model | Best for |
|---------|---------------|----------|
| `transformers` | [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) | English PER/ORG/LOC/MISC, production baseline |
| `gliner` | [urchade/gliner_medium-v2.1](https://huggingface.co/urchade/gliner_medium-v2.1) | Custom entity types without retraining |
| `pattern` | built-in | Tests, CI, offline demos |

## Tests & CI

```bash
uv sync --extra dev
uv run pytest tests/ -q --cov=ner_detector --cov-fail-under=95
```

Coverage gate: **≥95%** on `ner_detector` (enforced in `pyproject.toml`). ML backends are tested with mocks — no model download in CI.

GitHub Actions:

- **CI** (`.github/workflows/ci.yml`) — runs on every push/PR.
- **Benchmark report** (`.github/workflows/benchmark-pages.yml`) — publishes `report.html` to GitHub Pages; configure repeats and backends via [repository variables and secrets](docs/ci.md).

After Pages is enabled, the live report URL is shown under **Settings → Pages** (link it from the README when ready).

## Documentation

**[docs/README.md](docs/README.md)** — architecture, Python API, configuration, CLI, model survey.

| Guide | Topic |
|-------|--------|
| [docs/architecture.md](docs/architecture.md) | Backends, registry, chunking |
| [docs/cli.md](docs/cli.md) | Flags, output, exit codes |
| [docs/python-api.md](docs/python-api.md) | `detect_entities`, types |
| [docs/configuration.md](docs/configuration.md) | YAML defaults, `uv` extras |
| [docs/models.md](docs/models.md) | Model picks (HF / GitHub) |
| [docs/benchmarks.md](docs/benchmarks.md) | Compare backends on gold data |
| [docs/ci.md](docs/ci.md) | GitHub Actions, Pages, variables/secrets |

## Benchmark

Gold datasets live in the sibling **[ner-dataset](https://github.com/marfago-labs/ner-dataset)** repo (`../ner-dataset/datasets/`). See [docs/benchmarks.md](docs/benchmarks.md).

```bash
uv run python benchmark/run_benchmark.py --pattern-only
# → benchmark/results/run-*/report.html (radar charts) + report.md
```

Console script: `ner-detect` (same as `run.py`).

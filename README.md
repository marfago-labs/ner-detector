# ner-detector

Named entity recognition (NER) with **pluggable backends**: deterministic regex (`pattern`), fixed-label BERT (`transformers`), zero-shot GLiNER (`gliner`), NuNER (`nuner`), and chat LLM (`llm` via OpenRouter or mock).

**Benchmark report:** [GitHub Pages](https://marfago-labs.github.io/ner-detector/) · [Run locally](docs/benchmarks.md)

Part of [marfago-labs](https://github.com/marfago-labs). MIT licensed. Pairs with [ner-gold-generator](https://github.com/marfago-labs/ner-gold-generator) and [ner-dataset](https://github.com/marfago-labs/ner-dataset).

> **Pages setup:** the live report URL works after enabling GitHub Pages (Settings → Pages → GitHub Actions) and a successful **Benchmark report (Pages)** workflow run. See [docs/ci.md](docs/ci.md).

## Architecture & Benchmark Trade-offs

Evidence-driven comparison on shared gold data (`synthetic_news_100`, salient-entity `arxiv_gold`). Latest local run: `compare_backends.yaml` (2026-06-09), Doc F1 as primary metric. Full tables: `benchmark/results/latest/report.md`.

- **NuNER (`numind/NuNER_Zero`):** ~78% Doc F1 on synthetic news (~0.8 s/doc) — best measured quality on standard entity types in the latest run.
- **Transformers (`dslim/bert-base-NER`):** ~135 ms/document, ~73% Doc F1 on synthetic news — best real-time baseline on fixed PER/ORG/LOC schemas.
- **Zero-shot GLiNER:** Custom labels without retraining (~36–39% Doc F1 on arxiv scientific types, ~0.5–2 s/doc depending on model) — practical option for domain-specific entity types.
- **LLMs (`openai/gpt-oss-120b:free` via OpenRouter):** not scored in the 2026-06-09 run (OpenRouter 401 — check `OPENROUTER_API_KEY`). Re-run after fixing the key for LLM vs classical comparison.

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

## API keys and `.env`

Most flows need **no API keys** (pattern backend, tests, `agent_smoke.py`, `--pattern-only` benchmarks).

| What you run | Keys needed |
|--------------|-------------|
| Pattern / mock LLM | None |
| ML backends (BERT, GLiNER, NuNER) | None (optional [`HF_TOKEN`](https://huggingface.co/settings/tokens) for gated models or rate limits) |
| Live LLM NER (`backend: llm`, OpenRouter) | [`OPENROUTER_API_KEY`](https://openrouter.ai/keys) + `uv sync --extra llm` |

Local setup:

```bash
cp .env.example .env
# Edit .env — set OPENROUTER_API_KEY only if using live OpenRouter
```

Never commit `.env`. Full variable list: [docs/configuration.md](docs/configuration.md#environment). GitHub Actions secrets for the published benchmark report: [docs/ci.md](docs/ci.md#repository-secrets).

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

**Secret scanning:** Gitleaks runs in CI (`secrets` job) and via pre-commit (see [docs/ci.md](docs/ci.md#secret-scanning-gitleaks)). Keep API keys in `.env` only.

GitHub Actions:

- **CI** (`.github/workflows/ci.yml`) — runs on every push/PR.
- **Benchmark report** (`.github/workflows/benchmark-pages.yml`) — publishes `report.html` to GitHub Pages; configure via [docs/ci.md](docs/ci.md) (secrets, variables, first-time Pages enablement).

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
| [docs/for-agents.md](docs/for-agents.md) | Coding-agent tasks, contracts, smoke workflow |
| [docs/adr/001-doc-f1-primary-metric.md](docs/adr/001-doc-f1-primary-metric.md) | Why Doc F1 is the primary metric |

## Limitations

- **Corpus size:** ~510 documents total (`arxiv_gold` = 10 salient-annotated abstracts; synthetics = 100 each). Regression and methodology gold—not a SOTA leaderboard dataset.
- **Doc F1 primary:** Salient-entity gold uses document-level F1 as the headline metric; read [docs/benchmarks.md](docs/benchmarks.md) and [ADR 001](docs/adr/001-doc-f1-primary-metric.md) before comparing to external span-F1 benchmarks.
- **CI scope:** Fast CI uses mocks (`--extra dev` only); full ML/LLM benchmarks run separately (see [docs/ci.md](docs/ci.md)).
- **Live LLM runs:** OpenRouter model ids (especially `:free` tiers) can change or retire; pin models in benchmark YAML for reproducibility.

## Benchmark

Gold datasets live in the sibling **[ner-dataset](https://github.com/marfago-labs/ner-dataset)** repo (`../ner-dataset/datasets/`). See [docs/benchmarks.md](docs/benchmarks.md).

```bash
uv run python benchmark/run_benchmark.py --pattern-only
# → benchmark/results/run-*/report.html (radar charts) + report.md
```

Console script: `ner-detect` (same as `run.py`).

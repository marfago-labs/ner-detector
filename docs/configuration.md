# Configuration

Settings use two YAML files:

| File | Purpose |
|------|---------|
| `config/ner.yaml` | **Active profile** — backend, model, threshold, labels |
| `config/default_models.yaml` | **Model catalog** — per-backend defaults and label presets |

**Precedence:** CLI flags → `config/ner.yaml` → hardcoded fallbacks.

Override the profile path with `--config` / `-c` or `NER_CONFIG_PATH`.

## `config/ner.yaml`

```yaml
backend: pattern          # pattern | transformers | gliner | llm
# model_id: dslim/bert-base-NER   # omit = use catalog default for backend
threshold: 0.5            # ML backends only
# labels: [person, organization, location]
label_preset: general_en  # gliner / llm when labels omitted
# LLM-only (ignored by other backends):
provider: openrouter      # mock | openrouter
temperature: 0
max_chars: 8000
```

Inspect merged settings without running NER:

```bash
uv run python run.py --show-config
uv run python run.py --show-config --config ./my-ner.yaml
```

## `config/default_models.yaml`

Defines default model ids per backend and optional label presets.

```yaml
transformers:
  default: dslim/bert-base-NER
  alternatives: [...]

gliner:
  default: urchade/gliner_medium-v2.1
  alternatives: [...]

llm:
  default: nvidia/nemotron-3-super-120b-a12b:free
  alternatives: [...]
  providers: [mock, openrouter]

label_presets:
  general_en: [person, organization, location, date, product]
  scientific: [...]
  pii: [...]
```

Override at runtime with CLI `--model` or API `model_id=`.

## Optional dependencies

| Extra | Packages | Enables |
|-------|----------|---------|
| `ml` | `transformers`, `torch`, `accelerate` | `transformers` backend |
| `gliner` | `gliner` | `gliner` backend |
| `llm` | `httpx` | `llm` backend (OpenRouter HTTP) |
| `dev` | `pytest`, `pytest-cov` | Tests and coverage gate |

```bash
uv sync --extra dev --extra ml --extra gliner --extra llm
```

## Example profiles

**Offline / CI** (`config/ner.yaml` as shipped):

```yaml
backend: pattern
threshold: 0.5
```

**English BERT NER:**

```yaml
backend: transformers
model_id: dslim/bert-base-NER
threshold: 0.5
```

**Zero-shot GLiNER:**

```yaml
backend: gliner
model_id: urchade/gliner_medium-v2.1
threshold: 0.5
label_preset: general_en
```

**LLM (offline mock for benchmarks):**

```yaml
backend: llm
provider: mock
model_id: mock/ner
label_preset: scientific
temperature: 0
```

**LLM (live OpenRouter):**

```yaml
backend: llm
provider: openrouter
model_id: nvidia/nemotron-3-super-120b-a12b:free
labels: [person, organization, model, dataset, method]
temperature: 0
```

Requires `uv sync --extra llm` and `OPENROUTER_API_KEY` in `.env`.

## Environment

| Variable | Purpose |
|----------|---------|
| `NER_CONFIG_PATH` | Path to runtime profile YAML (default: `./config/ner.yaml`) |
| `HF_TOKEN` | Optional Hugging Face token for gated models (see `.env.example`) |
| `TRANSFORMERS_VERBOSITY` | Transformers log level; default in `.env.example` is `error` |
| `OPENROUTER_API_KEY` | OpenRouter auth for `llm` + `provider: openrouter` |
| `OPENROUTER_BASE_URL` | API base (default `https://openrouter.ai/api/v1`) |
| `MOCK_LLM` | When `true`, `provider: openrouter` uses deterministic mock client |

## Testing

Coverage gate **≥95%** on package `ner_detector` (`pyproject.toml` → `[tool.coverage.report] fail_under = 95`). ML backends are exercised with mocks in CI — no model download required for `uv run pytest`.

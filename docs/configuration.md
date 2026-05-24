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
backend: pattern          # pattern | transformers | gliner
# model_id: dslim/bert-base-NER   # omit = use catalog default for backend
threshold: 0.5
# labels: [person, organization, location]
label_preset: general_en  # used for gliner when labels omitted
```

Inspect merged settings without running NER:

```bash
uv run python run.py --show-config
uv run python run.py --show-config --config ./my-ner.yaml
```

## `config/default_models.yaml`

Defines default Hugging Face model ids per backend and optional label presets.

```yaml
transformers:
  default: dslim/bert-base-NER
  alternatives: [...]

gliner:
  default: urchade/gliner_medium-v2.1
  alternatives: [...]

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
| `dev` | `pytest`, `pytest-cov` | Tests and coverage gate |

```bash
uv sync --extra dev --extra ml --extra gliner
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
# labels: [person, company, city]   # or set explicitly
```

Requires `uv sync --extra ml` (transformers) and/or `--extra gliner`.

## Environment

| Variable | Purpose |
|----------|---------|
| `NER_CONFIG_PATH` | Path to runtime profile YAML (default: `./config/ner.yaml`) |
| `HF_TOKEN` | Optional Hugging Face token for gated models (see `.env.example`) |
| `TRANSFORMERS_VERBOSITY` | Transformers log level; default in `.env.example` is `error` (quieter benchmark runs) |

## Testing

Coverage gate **≥95%** on package `ner_detector` (`pyproject.toml` → `[tool.coverage.report] fail_under = 95`). ML backends are exercised with mocks in CI — no model download required for `uv run pytest`.

# ner-detector documentation

Pluggable **named entity recognition**: pattern (offline), Hugging Face token-classification, and GLiNER zero-shot.

## Guides

| Document | Contents |
|----------|----------|
| [Configuration](configuration.md) | `.env` setup, API keys (`OPENROUTER_API_KEY`, `HF_TOKEN`), env vars, extras |
| [CLI reference](cli.md) | Flags, config precedence, output, exit codes |
| [Python API](python-api.md) | `detect_entities`, `resolve_ner_settings` |
| [Architecture](architecture.md) | Backends, registry cache, config flow |
| [Model survey](models.md) | Best NER models (Hugging Face, GitHub, benchmarks) |
| [Benchmarks](benchmarks.md) | Gold datasets, backend comparison reports |
| [CI & Pages](ci.md) | GitHub Actions, published benchmark report, variables/secrets |
| [For coding agents](for-agents.md) | Monorepo layout, contracts, smoke workflow |
| [ADR 001](adr/001-doc-f1-primary-metric.md) | Document F1 as primary metric |

## Quick start

```bash
uv sync --extra dev

# 1. Choose backend/model in config/ner.yaml (default: pattern)
# 2. Run
uv run python run.py "Alice Smith works at OpenAI in 2024."

uv run python run.py --show-config
uv run pytest tests/ -q --cov=ner_detector --cov-fail-under=95
```

**Precedence:** CLI flags → `config/ner.yaml` → catalog defaults in `config/default_models.yaml`.

See the [root README](../README.md) for a one-page overview.

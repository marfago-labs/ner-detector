# Guide for coding agents

This repo is **documentation- and schema-first** so humans and coding agents can operate from `docs/`, YAML configs, and JSON outputs without reverse-engineering every module.

## Monorepo layout

```
marfago-labs/
  ner-gold-generator/   # build gold JSONL
  ner-dataset/          # committed benchmark artifacts (datasets/*.jsonl)
  ner-detector/         # this repo — backends + benchmarks
```

**Dataset path resolution** (in order): absolute `benchmark_root` in YAML → `NER_DATASET_DIR` env (folder containing `*.jsonl` or its parent `datasets/`) → sibling `../ner-dataset` → in-repo checkout `ner-dataset/` (CI layout).

## Credentials (`.env`)

Never commit `.env`. Copy `.env.example` → `.env` only when using live OpenRouter.

| Task | Variables |
|------|-----------|
| Smoke benchmark, pattern NER, PR CI | **None** |
| ML benchmark (local) | Optional `HF_TOKEN` |
| Live LLM NER / LLM benchmark runs | `OPENROUTER_API_KEY` (+ `uv sync --extra llm`) |

```bash
cp .env.example .env
# OPENROUTER_API_KEY=...   # https://openrouter.ai/keys
```

Details: [configuration.md](configuration.md#environment). GitHub Actions: [ci.md](ci.md#repository-secrets).

## Safety

- Never commit `.env` or API keys. Gitleaks runs in CI and pre-commit.
- **Offline / no API:** use `backend: pattern`, `provider: mock`, `MOCK_LLM=true`, or `scripts/agent_smoke.py`.
- **ML download:** full benchmarks need `uv sync --extra ml --extra gliner`; CI does not download models on every PR.

## Key contracts

| Artifact | Location |
|----------|----------|
| Gold JSONL schema | [ner-gold-generator gold-schema.md](https://github.com/marfago-labs/ner-gold-generator/blob/master/docs/gold-schema.md) |
| Benchmark config | [benchmark/config/compare_backends.yaml](../benchmark/config/compare_backends.yaml) |
| Label normalization | [benchmark/config/label_maps.yaml](../benchmark/config/label_maps.yaml) |
| Metrics output | `{output_dir}/metrics.json` after `run_benchmark.py` |
| Doc F1 semantics | [ADR 001](adr/001-doc-f1-primary-metric.md), [benchmarks.md](benchmarks.md) |

## Config precedence (detection)

CLI flags → `config/ner.yaml` → `config/default_models.yaml` catalog defaults.

## Agent task recipe 1 — smoke benchmark (recommended)

No API keys, no ML download:

```bash
cd ner-detector
uv sync --extra dev
uv run python scripts/agent_smoke.py
```

Prints JSON to stdout: `{"ok": true, "output_dir": "...", "metrics_json": "..."}`.

Equivalent manual command:

```bash
uv run python benchmark/run_benchmark.py --pattern-only --datasets arxiv_gold --max-examples 2 --no-curves
```

## Agent task recipe 2 — regenerate smoke gold (optional)

Requires sibling [ner-gold-generator](https://github.com/marfago-labs/ner-gold-generator):

```bash
cd ner-gold-generator
uv sync --extra dev
uv run build-gold --source synthetic \
  --synthetic-config configs/synthetic_smoke.yaml \
  --output /tmp/synthetic_smoke.jsonl
```

Point `NER_DATASET_DIR` at the output directory (or copy into `ner-dataset/datasets/`), then run recipe 1.

## CI vs local

| Context | Behavior |
|---------|----------|
| PR CI ([ci.yml](../.github/workflows/ci.yml)) | Gitleaks + pytest with mocks; checks out `ner-dataset` into workspace |
| Benchmark Pages ([benchmark-pages.yml](../.github/workflows/benchmark-pages.yml)) | Full ML/LLM run on `master`; optional `OPENROUTER_API_KEY` |

## Sibling docs

- [ner-gold-generator for-agents.md](https://github.com/marfago-labs/ner-gold-generator/blob/master/docs/for-agents.md)
- [ner-dataset for-agents.md](https://github.com/marfago-labs/ner-dataset/blob/master/docs/for-agents.md)
- Repo root [llms.txt](../llms.txt)

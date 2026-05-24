# Architecture

`ner-detector` is a **library + CLI** (not a server). Text goes in; a list of `DetectedEntity` spans comes out. Backends are selected at runtime and cached by `(backend, model_id)`.

## Flow

```mermaid
flowchart LR
  CFG[config/ner.yaml] --> RES[resolve_ner_settings]
  CLI[CLI flags] --> RES
  CAT[default_models.yaml] --> RES
  RES --> DET[detect_entities]
  IN[Text or file] --> CLIMAIN[cli.main]
  CLIMAIN --> RES
  DET --> REG[registry.create_backend]
  REG --> BE[Backend.detect]
  BE --> OUT[List of DetectedEntity]
```

## Settings resolution

| Source | Provides |
|--------|----------|
| `config/ner.yaml` | `backend`, `model_id`, `threshold`, `labels`, `label_preset` |
| CLI (`--backend`, `--model`, …) | Overrides profile fields |
| `config/default_models.yaml` | Default `model_id` per backend; `label_presets` for GLiNER |

`resolve_ner_settings()` in `ner_detector.config` performs the merge. See [configuration.md](configuration.md).

## Backends

| Backend | Module | Labels | Dependencies |
|---------|--------|--------|----------------|
| `pattern` | `backends/pattern.py` | Fixed regex types (`person`, `year`, `arxiv_id`, …) | core only |
| `transformers` | `backends/transformers_backend.py` | Model head (e.g. PER/ORG/LOC/MISC) | `--extra ml` |
| `gliner` | `backends/gliner_backend.py` | User-supplied strings (zero-shot) | `--extra gliner` |

Long inputs for `transformers` are split with `_chunk_text()` (default 4000 chars, 200 overlap). If `overlap >= max_chars`, the chunker advances by `end` to avoid stalling.

## Entry points

- `uv run python run.py` — same as `ner-detect` console script (`pyproject.toml` → `ner_detector.cli:main`)
- `python -m ner_detector.cli`

Config path: `--config` / `-c`, or env `NER_CONFIG_PATH` (default `./config/ner.yaml`).

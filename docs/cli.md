# CLI reference

Program name: **`ner-detect`** (also `uv run python run.py`).

## Usage

```bash
ner-detect [input] [options]
```

If `input` is omitted, text is read from **stdin**.

## Arguments and flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `input` | — | — | Inline text, or file path when `--file` is set |
| `--file` | `-f` | off | Treat `input` as a path to a UTF-8 text file |
| `--backend` | `-b` | from `ner.yaml` | Override profile backend |
| `--model` | `-m` | from `ner.yaml` / catalog | Override Hugging Face model id |
| `--labels` | `-l` | from `ner.yaml` | Comma-separated; overrides profile |
| `--threshold` | `-t` | from `ner.yaml` | Minimum confidence score |
| `--format` | — | `json` | `json` or `text` (tab-separated) |
| `--config` | `-c` | `config/ner.yaml` or `NER_CONFIG_PATH` | Runtime profile (backend, model, …) |
| `--list-models` | — | off | Print `config/default_models.yaml` and exit |
| `--show-config` | — | off | Print resolved settings and exit |

CLI flags override values from `config/ner.yaml`.

## GLiNER labels

When the resolved backend is `gliner` and no labels are set in the profile or CLI:

1. Use `labels` from `config/ner.yaml` if present.
2. Else resolve `label_preset` from `ner.yaml` (default `general_en`) against `label_presets` in `config/default_models.yaml`.

## Output formats

**JSON** (default):

```json
{
  "backend": "pattern",
  "model_id": null,
  "config_path": "config/ner.yaml",
  "entity_count": 2,
  "entities": [
    {"text": "Alice Smith", "label": "person", "score": 1.0, "start": 0, "end": 11}
  ]
}
```

**Text**: one entity per line — `TEXT\tLABEL (score)` (score omitted when null).

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Missing input, file not found, read error, missing optional dependency, or `ValueError` from detection |

## Examples

```bash
# Profile-driven (edit config/ner.yaml)
uv run python run.py --show-config
uv run python run.py "Paper 2402.15343 by Alice Smith."

# Custom profile file
uv run python run.py -c profiles/gliner.yaml "Bill Gates founded Microsoft."

# One-off CLI overrides (ignore ner.yaml for that run)
uv sync --extra ml --extra gliner
uv run python run.py -b transformers -m dslim/bert-base-NER "Alice works at OpenAI."
uv run python run.py -b gliner -l "person,company,city" "Bill Gates founded Microsoft."

# Model catalog (not the active profile)
uv run python run.py --list-models
```

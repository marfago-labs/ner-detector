# Python API

## `detect_entities`

```python
from ner_detector import detect_entities, DetectedEntity

entities: list[DetectedEntity] = detect_entities(
    "Alice Smith joined OpenAI in 2024.",
    backend="pattern",           # "pattern" | "transformers" | "gliner"
    model_id=None,               # HF id; backend default if None
    labels=None,                 # GLiNER / pattern filter
    threshold=0.5,
)
```

## `DetectedEntity`

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Surface form |
| `label` | `str` | Entity type |
| `score` | `float \| None` | Confidence when available |
| `start` | `int \| None` | Character offset in source text |
| `end` | `int \| None` | Exclusive end offset |

`to_dict()` returns a JSON-serializable mapping for CLI output.

## Registry (advanced)

```python
from ner_detector.registry import create_backend, clear_backend_cache

backend = create_backend("transformers", model_id="dslim/bert-base-NER")
entities = backend.detect(text, threshold=0.5)
clear_backend_cache()  # tests / long-running processes
```

Backends are singleton-cached per `(backend, model_id)`.

## Configuration

```python
from ner_detector.config import load_ner_config, resolve_ner_settings

# From config/ner.yaml (or NER_CONFIG_PATH)
profile = load_ner_config()

# Merge file + overrides (CLI-style)
settings = resolve_ner_settings(
    backend="gliner",
    model_id=None,
    threshold=0.6,
)
entities = detect_entities(
    text,
    backend=settings.backend,
    model_id=settings.model_id,
    labels=settings.labels,
    threshold=settings.threshold,
)
```

See [configuration.md](configuration.md).

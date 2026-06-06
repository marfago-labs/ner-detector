# Python API

## `detect_entities`

```python
from ner_detector import detect_entities, DetectedEntity

entities: list[DetectedEntity] = detect_entities(
    "Alice Smith joined OpenAI in 2024.",
    backend="pattern",           # "pattern" | "transformers" | "gliner" | "llm"
    model_id=None,               # catalog default if None
    labels=None,                 # GLiNER / LLM label list
    threshold=0.5,               # ML backends only
    provider=None,               # LLM: "mock" | "openrouter"
    temperature=None,            # LLM sampling temperature
    max_chars=None,              # LLM chunk size
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
from ner_detector.registry import BackendOptions, create_backend, clear_backend_cache

backend = create_backend(
    "llm",
    model_id="mock/ner",
    options=BackendOptions(provider="mock", temperature=0.0),
)
entities = backend.detect(text, labels=["person", "organization"])
clear_backend_cache()  # tests / long-running processes
```

Backends are singleton-cached per `(backend, model_id, …)`; LLM cache key includes `provider`, `temperature`, and `max_chars`.

## Configuration

```python
from ner_detector.config import load_ner_config, resolve_ner_settings

profile = load_ner_config()

settings = resolve_ner_settings(
    backend="llm",
    provider="mock",
    model_id="mock/ner",
)
entities = detect_entities(
    text,
    backend=settings.backend,
    model_id=settings.model_id,
    labels=settings.labels,
    provider=settings.provider,
    temperature=settings.temperature,
    max_chars=settings.max_chars,
)
```

See [configuration.md](configuration.md).

# NER model survey (2025–2026)

Research sources: [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=token-classification&sort=downloads), [GitHub](https://github.com/urchade/GLiNER), [Transformers docs](https://huggingface.co/docs/transformers/tasks/token_classification) (via Context7), and recent model cards.

## How to choose

| Need | Recommended approach | Example model |
|------|---------------------|---------------|
| Fixed English labels (PER, ORG, LOC, MISC) | **Transformers** token-classification | `dslim/bert-base-NER` |
| Custom entity types at runtime | **GLiNER** zero-shot | `urchade/gliner_medium-v2.1` |
| Salient / scientific concepts via prompt | **LLM** (OpenRouter or mock) | `nvidia/nemotron-3-super-120b-a12b:free` |
| Many labels, production throughput | **GLiNER bi-encoder** | `knowledgator/gliner-bi-base-v2.0` |
| Multilingual (20+ langs) | GLiNER-X or `gliner_multi` | `knowledgator/gliner-x-large` |
| Clinical / biomedical | Domain fine-tunes | `IEETA/MultiClinNER-MIXED`, OpenMed NER |
| SOTA English encoder (research) | NuNER embeddings | `numind/NuNER-v2.0` |
| Unified NER + classification + extraction | GLiNER2 | `fastino/gliner2-base-v1` |
| Tests / CI without GPU | **pattern** backend | built-in |

## Tier 1 — production defaults

### 1. `dslim/bert-base-NER` (transformers)

- **Downloads:** ~90M on Hugging Face
- **Labels:** PER, ORG, LOC, MISC (CoNLL-2003)
- **Size:** ~108M parameters
- **Why:** Most battle-tested English NER checkpoint; works with `pipeline("ner")`; used in marfago-labs `text-compressor` for entity F1 scoring
- **Link:** https://huggingface.co/dslim/bert-base-NER

```python
from transformers import pipeline
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
ner("Apple was founded by Steve Jobs in California.")
```

### 2. `urchade/gliner_medium-v2.1` (GLiNER)

- **Type:** Zero-shot — you pass label strings at inference
- **Size:** Medium (~200M class); also `small`, `large`, `multi`
- **Why:** Best balance of accuracy and speed for open-vocabulary NER; official [urchade/GLiNER](https://github.com/urchade/GLiNER) repo (4k+ stars)
- **Link:** https://huggingface.co/urchade/gliner_medium-v2.1

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
model.predict_entities(text, ["person", "company", "location"], threshold=0.5)
```

### 3. `knowledgator/gliner-bi-base-v2.0` (GLiNER bi-encoder)

- **Type:** Scalable zero-shot; pre-compute label embeddings
- **Benchmark:** ~60% avg CrossNER; ~98% of large model at 2.6× speed (per model card)
- **Why:** When you need **hundreds or thousands** of entity types without linear slowdown
- **Link:** https://huggingface.co/knowledgator/gliner-bi-base-v2.0
- **In this repo:** benchmarked as `gliner-bi-large` (`knowledgator/gliner-bi-large-v2.0`) with **`threshold: 0.3`** (not `0.5` like uni-encoder GLiNER)

## Tier 2 — specialized / newer

| Model | Strength | Notes |
|-------|----------|-------|
| `knowledgator/gliner-x-large` | Multilingual (20+ languages) | MT5 backbone; beats `gliner_multi-v2.1` on several langs |
| `fastino/gliner2-base-v1` | NER + text classification + structured JSON | 453k+ downloads; single 205M model |
| `numind/NuNER-v2.0` | SOTA English entity **embeddings** | Not a drop-in span tagger; contrastive encoder for similarity / few-shot |
| `nvidia/gliner-PII` | PII / PHI detection | Privacy-focused labels |
| `urchade/gliner_multi_pii-v1` | Multilingual PII | EN, FR, DE, ES, PT, IT |
| `Jean-Baptiste/camembert-ner-with-dates` | French NER + dates | 5.7M downloads |
| `IEETA/MultiClinNER-MIXED` | Clinical, 7 languages | XLM-RoBERTa + multi-head CRF |
| OpenMed `*-NER-*` family | Biomedical anatomy, drugs, etc. | Domain SOTA on clinical benchmarks |

## GitHub ecosystems

| Repository | Role |
|------------|------|
| [urchade/GLiNER](https://github.com/urchade/GLiNER) | Original GLiNER training & inference |
| [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2) | Next-gen unified extraction |
| [Knowledgator/GLiNER.cpp](https://github.com/Knowledgator/GLiNER.cpp) | C++ inference for edge |
| [dslim/NER](https://huggingface.co/spaces/dslim/NER) | Demo Space for bert-base-NER |
| [explosion/spaCy](https://github.com/explosion/spaCy) | Classic industrial NER (`en_core_web_*`); not zero-shot |

## What we ship in this repo

| Backend | Default model | Install extra |
|---------|---------------|---------------|
| `pattern` | regex rules | (core only) |
| `transformers` | `dslim/bert-base-NER` | `--extra ml` |
| `gliner` | `urchade/gliner_medium-v2.1` | `--extra gliner` (+ `ml` recommended) |
| `llm` | `nvidia/nemotron-3-super-120b-a12b:free` | `--extra llm`; `OPENROUTER_API_KEY` for live OpenRouter |

Set the active model in **`config/ner.yaml`** (`model_id`, `provider`), or override per run with CLI `--model` / `--provider` / Python kwargs. Catalog defaults are in `config/default_models.yaml`.

## References

- GLiNER paper: [arXiv:2311.08526](https://arxiv.org/abs/2311.08526)
- NuNER: [arXiv:2402.15343](https://arxiv.org/abs/2402.15343)
- GLiNER bi-encoder: [arXiv:2602.18487](https://arxiv.org/abs/2602.18487)
- Transformers token classification: https://huggingface.co/docs/transformers/tasks/token_classification

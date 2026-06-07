# Portfolio brief (recruiters & LinkedIn)

Copy-paste snippets for profiles, posts, and project descriptions. Update benchmark numbers after each major run (see `benchmark/results/`).

## Elevator pitch (2 sentences)

Open-source NER evaluation platform that compares regex, BERT, zero-shot GLiNER, NuNER, and LLM backends on shared gold data—with Doc F1, strict/relaxed span metrics, latency, and interactive benchmark reports. Built to answer *when an LLM is worth the latency cost* versus classical or zero-shot models.

## LinkedIn Featured — title

**Enterprise NER Evaluation Framework — LLMs vs GLiNER vs BERT (benchmarked)**

## LinkedIn Featured — body

I built **ner-detector** as part of [marfago-labs](https://github.com/marfago-labs): a reproducible pipeline to compare Named Entity Recognition backends on the same gold datasets.

**What it does**

- Pluggable backends: pattern (regex), BERT (`transformers`), zero-shot GLiNER / NuNER, and LLM extraction via OpenRouter
- Rigorous evaluation: document-level F1, strict span F1, relaxed span F1 (IoU ≥ 0.5), label confusion matrices, latency per document
- Engineering bar: ≥95% test coverage, typed Python, CI, auto-generated HTML benchmark reports

**Latest findings** (canonical gold, `compare_backends.yaml`)

| Setting | Best quality | Best latency |
|---------|--------------|--------------|
| Synthetic news (100 docs) | LLM ~84% Doc F1 (~7s/doc) | BERT ~73% Doc F1 (~80ms/doc) |
| ML paper abstracts (salient gold) | LLM ~47% Doc F1 (~9s/doc) | GLiNER ~36–39% Doc F1 (~0.4s/doc) |

**Takeaway:** LLMs win on extraction quality where latency is acceptable; BERT remains the real-time baseline on standard entity types; GLiNER is the practical zero-shot option for custom schemas (e.g. scientific concepts) without retraining.

**Ecosystem:** [ner-gold-generator](https://github.com/marfago-labs/ner-gold-generator) → [ner-dataset](https://github.com/marfago-labs/ner-dataset) → **ner-detector** (this repo).

**Links:** [GitHub — ner-detector](https://github.com/marfago-labs/ner-detector) · [Live benchmark report](https://marfago-labs.github.io/ner-detector/) *(after GitHub Pages is enabled)*

## ArXplorer project blurb

**ArXplorer** uses AI agents to search and analyze arXiv papers. Extraction quality for scientific entities (models, datasets, benchmarks, metrics) is backed by the marfago-labs NER stack: gold built with **ner-gold-generator**, stored in **ner-dataset**, and backends compared with **ner-detector** so we choose LLM vs zero-shot vs classical models based on measured accuracy and latency—not hype.

## Short post hook (optional)

> Shipped an open-source NER benchmark harness: same gold, four paradigms (regex / BERT / GLiNER / LLM), full latency + F1 report. Spoiler: LLMs are best on quality, BERT on speed, GLiNER on custom labels without training. Code + methodology in the repo.

## Publish checklist

Before sharing widely with recruiters:

1. **Push** local commits to `marfago-labs/ner-detector` on GitHub.
2. **Visibility:** repo is currently **private** — set to **public** if you want open portfolio links.
3. **GitHub Pages:** Settings → Pages → Source **GitHub Actions**; add `OPENROUTER_API_KEY` secret if the workflow should run the LLM backend; re-run **Benchmark report (Pages)**. See [ci.md](ci.md).
4. **Verify** [https://marfago-labs.github.io/ner-detector/](https://marfago-labs.github.io/ner-detector/) loads after deploy.
5. **LinkedIn Featured:** paste title + body above; link repo and live report.

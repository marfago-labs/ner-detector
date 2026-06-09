# Portfolio brief (recruiters & LinkedIn)

Copy-paste snippets for profiles, posts, and project descriptions. Update benchmark numbers after each major run (see `benchmark/results/`). Last refreshed: **2026-06-09** (`benchmark/results/latest/`).

## Elevator pitch (2 sentences)

As an AI leader and Principal Engineer, I built this open-source NER evaluation platform to rigorously compare regex, BERT, zero-shot GLiNER, NuNER, and LLM backends on shared gold data. It provides enterprise-grade metrics (Doc F1, latency, interactive reports) to definitively answer *when an LLM is worth the latency cost* in production systems versus classical or zero-shot models.

## LinkedIn Featured — title

**Enterprise NER Evaluation Framework — LLMs vs GLiNER vs BERT (benchmarked)**

## LinkedIn Featured — body

In my work leading AI solutions, I constantly evaluate the trade-offs between model quality and production latency. I built **ner-detector** as part of [marfago-labs](https://github.com/marfago-labs) to provide a reproducible, enterprise-grade pipeline for comparing Named Entity Recognition backends on the same gold datasets.

**What it does**

- Pluggable backends: pattern (regex), BERT (`transformers`), zero-shot GLiNER / NuNER, and LLM extraction via OpenRouter
- Rigorous evaluation: document-level F1, strict span F1, relaxed span F1 (IoU ≥ 0.5), label confusion matrices, latency per document
- Engineering bar: ≥95% test coverage, typed Python, CI, auto-generated HTML benchmark reports
- Agent-legible: schema-documented gold, YAML-driven benchmarks, [for-agents.md](for-agents.md) + offline `scripts/agent_smoke.py` for coding agents

**Latest findings** (canonical gold, `compare_backends.yaml`, run 2026-06-09)

| Setting | Best quality | Best latency (usable quality) |
|---------|--------------|-------------------------------|
| Synthetic news (100 docs) | NuNER ~78% Doc F1 (~0.8s/doc) | BERT ~73% Doc F1 (~135ms/doc) |
| ML paper abstracts (salient gold, 10 docs) | GLiNER bi-large ~39% Doc F1 (~2.1s/doc) | GLiNER medium ~36% Doc F1 (~525ms/doc) |

**LLM note:** `llm-gpt-oss` (`openai/gpt-oss-120b:free`) returned **OpenRouter 401 (User not found)** in this run — refresh `OPENROUTER_API_KEY` and re-run for live LLM scores. Prior successful runs showed higher Doc F1 at ~7–9s/doc when the key was valid.

**Takeaway:** On standard PER/ORG/LOC news gold, NuNER leads Doc F1 while BERT stays the fast baseline; on sparse scientific abstracts, zero-shot GLiNER is the practical choice without retraining. LLM quality/latency trade-offs need a valid OpenRouter key to measure in the harness.

**Ecosystem:** [ner-gold-generator](https://github.com/marfago-labs/ner-gold-generator) → [ner-dataset](https://github.com/marfago-labs/ner-dataset) → **ner-detector** (this repo).

**Links:** [GitHub — ner-detector](https://github.com/marfago-labs/ner-detector) · [Live benchmark report](https://marfago-labs.github.io/ner-detector/) · [Dataset stats](https://marfago-labs.github.io/ner-dataset/)

## Agent-legible engineering

Secondary portfolio narrative (not a replacement for measured eval):

- **Contracts in docs:** [gold schema](https://github.com/marfago-labs/ner-gold-generator/blob/master/docs/gold-schema.md), [benchmark YAML](../benchmark/config/compare_backends.yaml), [ADR 001](adr/001-doc-f1-primary-metric.md)
- **Machine-readable outputs:** `metrics.json`, [ner-dataset stats.json](https://github.com/marfago-labs/ner-dataset/blob/master/docs/stats.json)
- **Agent entrypoint:** [for-agents.md](for-agents.md), repo-root `llms.txt`, `uv run python scripts/agent_smoke.py` (pattern-only, no API keys)

Prefer "agent-legible" or "documentation-first for human and AI contributors" over vague "AI-ready."

## Short post hook (optional)

> In enterprise AI, code is becoming a commodity, but rigorous evaluation is not. I shipped an open-source NER benchmark harness to measure the real trade-offs between regex, BERT, GLiNER, and LLMs. Spoiler: LLMs win on quality, BERT on speed, and GLiNER on custom labels without training. Code + methodology in the repo.

## Publish checklist

Before sharing widely with recruiters:

1. **Push** local commits to `marfago-labs/ner-detector` on GitHub.
2. **Visibility:** set repo to **public** if you want open portfolio links.
3. **GitHub Pages:** Settings → Pages → Source **GitHub Actions**; add `OPENROUTER_API_KEY` secret if the workflow should run the LLM backend; re-run **Benchmark report (Pages)**. See [ci.md](ci.md).
4. **Verify** [https://marfago-labs.github.io/ner-detector/](https://marfago-labs.github.io/ner-detector/) loads after deploy.
5. **Secret Protection** + **Push protection** on all three NER repos.
6. **LinkedIn Featured:** paste title + body above; link repo, live report, and dataset stats. Do **not** lead with legacy product names—this stack stands alone as open benchmark tooling.

## License

[MIT](../LICENSE)

# Continuous integration and benchmark reports

## Overview

Workflows use **Node 24–native action versions** (`actions/checkout@v5`, `astral-sh/setup-uv@v8.1.0`, `actions/cache@v5`, `actions/upload-artifact@v6`, `actions/deploy-pages@v5`, etc.) so CI does not rely on the deprecated Node 20 runtime or `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24`.

| Workflow | File | When it runs | Purpose |
|----------|------|----------------|---------|
| **CI** | `.github/workflows/ci.yml` | Every push/PR to `master`/`main` | Gitleaks secret scan + `pytest` with ≥95% coverage (no ML download); checks out sibling `ner-dataset` |
| **Benchmark → Pages** | `.github/workflows/benchmark-pages.yml` | Push to `master`, manual | Full benchmark + publish `report.html` |

## Secret scanning (Gitleaks)

Hardcoded API keys and tokens are blocked by **[Gitleaks](https://github.com/gitleaks/gitleaks)**:

| Where | When |
|-------|------|
| **CI** job `secrets` | Every push/PR (OSS `gitleaks detect` CLI, full git history) |
| **pre-commit** hook `gitleaks` | Staged files before each commit (after `pre-commit install`) |

Config: [`.gitleaks.toml`](../.gitleaks.toml) (extends default rules; allowlists `tests/`, `.env.example`, docs).

```bash
gitleaks detect --source . --config .gitleaks.toml --verbose
```

Keep real keys only in local `.env` (gitignored). CI uses the free Gitleaks binary (not `gitleaks-action`, which can require a paid org license).

---

## Step 1 — Enable GitHub Pages (required for deploy; optional if you use artifacts only)

The **benchmark** job always succeeds independently. The **deploy** job needs Pages turned on in the repo.

1. Open [marfago-labs/ner-detector](https://github.com/marfago-labs/ner-detector) → **Settings** → **Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions** (not “Deploy from branch”).
3. Save, then re-run **Benchmark report (Pages)**.

After a successful deploy, the URL appears under **Settings → Pages** and in the **github-pages** environment summary.

### Private repository

| Situation | What to do |
|-----------|------------|
| Deploy fails with **404** / *Ensure GitHub Pages has been enabled* | Pages is not enabled yet — complete steps 1–3 above. |
| Pages enabled but still 404 | Your org/account may need **GitHub Team**, **Enterprise**, or **Pro** for Pages on **private** repos. Check org billing/features. |
| No Pages on your plan | Set variable `BENCHMARK_DEPLOY_PAGES` = `false` (see below). Download **`ner-benchmark-report`** from the workflow run **Artifacts** (repo members only). |
| Public report for everyone | Make the repo **public**, enable Pages, or publish the artifact elsewhere. |

**Who can view a private Pages site:** usually org members / collaborators with repo access, not the whole internet (unless you set Pages visibility to public where your plan allows it).

---

## Step 2 — Benchmark workflow (already in repo)

See `.github/workflows/benchmark-pages.yml`. Configure it via **repository variables** and **secrets** below (Settings → Secrets and variables → Actions).

---

## Repository variables (`vars`)

Set under **Settings → Secrets and variables → Actions → Variables** (repository scope).

| Variable | Default if unset | Description |
|----------|------------------|-------------|
| `BENCHMARK_REPEATS` | `1` | Trials per backend×dataset (`--repeats`) |
| `BENCHMARK_MAX_EXAMPLES` | *(empty)* | Cap examples per dataset (`--max-examples`) |
| `BENCHMARK_PATTERN_ONLY` | `false` | `true` → only run `pattern` backend |
| `BENCHMARK_DATASETS` | *(empty)* | Comma-separated dataset names (`--datasets`) |
| `BENCHMARK_RUNS` | *(empty)* | Comma-separated run names (`--runs`) |
| `TRANSFORMERS_VERBOSITY` | `error` | Passed to the benchmark job environment |
| `BENCHMARK_OUTPUT_DIR` | `benchmark/results/latest` | Fixed output directory for the published report |
| `BENCHMARK_DEPLOY_PAGES` | `true` | Set to `false` to skip GitHub Pages deploy (artifact upload still runs) |

**CI tip:** use `BENCHMARK_REPEATS=1` on GitHub-hosted runners. For latency variance locally, pass `--repeats 5`.

Gold JSONL files are loaded from the checked-out **[ner-dataset](https://github.com/marfago-labs/ner-dataset)** repo (`NER_DATASET_DIR=ner-dataset/datasets` in workflows). Clone both repos side by side locally, or set `NER_DATASET_DIR` to your gold folder.

### Download report without Pages

**Actions** → latest **Benchmark report (Pages)** run → **Artifacts** → `ner-benchmark-report` → open `report.html` locally.

---

## Repository secrets

Set under **Settings → Secrets and variables → Actions → Secrets** (repository scope).

| Secret | Required | Description |
|--------|----------|-------------|
| `OPENROUTER_API_KEY` | No* | OpenRouter API key for LLM benchmark runs (`llm-gpt-oss`, etc.). Omit or use `BENCHMARK_PATTERN_ONLY=true` for pattern-only reports. |
| `HF_TOKEN` | No | [Hugging Face token](https://huggingface.co/settings/tokens) for gated models or higher download rate limits |

\*Required only when the benchmark config includes live OpenRouter backends and `BENCHMARK_PATTERN_ONLY` is not `true`. The workflow passes these to the job environment (see `.github/workflows/benchmark-pages.yml`).

Local parity uses `.env` instead — see [configuration.md](configuration.md#environment) and `.env.example`.

---

## Step 3 — Workflow permissions

The benchmark deploy job requests:

- `pages: write`
- `id-token: write`

If deploy fails with permissions errors: **Settings → Actions → General → Workflow permissions** → allow read/write for workflows (or use the default GITHUB_TOKEN scope your org allows).

---

## Step 4 — Push and verify

1. Push to `master` (CI runs automatically).
2. **Actions** → confirm **CI** is green.
3. Enable Pages (step 1), then trigger **Benchmark report (Pages)** (push to `master` or **Run workflow**).
4. Open the Pages URL from the deployment summary.

---

## Step 5 — Who can see the report

| Repo visibility | Typical audience |
|-----------------|------------------|
| Private | Org members / collaborators with repo access |
| Public | Anyone with the Pages URL |

---

## Step 6 — Optional README link

After Pages works, add to `README.md`:

```markdown
**[Latest benchmark report](https://marfago-labs.github.io/ner-detector/)** *(update URL from Settings → Pages)*
```

---

## Manual re-run

**Actions** → **Benchmark report (Pages)** → **Run workflow** (uses current variables/secrets).

---

## Local parity

```bash
uv sync --extra dev --extra ml --extra gliner
uv run python benchmark/run_benchmark.py --repeats 1 --output benchmark/results/latest
```

See [benchmarks.md](benchmarks.md) for metrics and interpretation.

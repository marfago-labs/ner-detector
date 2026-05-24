"""Shared HTML: metrics/methodology content and tab layout for NER benchmark reports."""

from __future__ import annotations

import html
from pathlib import Path

from ner_detector.eval.runner import BenchmarkResult, load_benchmark_config

REPORT_TAB_CSS = """
    .report-tabs {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0 0 1.25rem;
      margin-bottom: 1.25rem;
    }
    .report-tabs > input.tab-input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }
    .tab-bar {
      display: flex;
      gap: 0;
      border-bottom: 1px solid var(--border);
      padding: 0 0.75rem;
    }
    .tab-bar label {
      display: inline-block;
      padding: 0.75rem 1rem;
      margin-bottom: -1px;
      cursor: pointer;
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--muted);
      border-bottom: 2px solid transparent;
    }
    .tab-bar label:hover { color: var(--text); }
    .report-tabs .tab-panel {
      display: none;
      padding: 1rem 1.25rem 0;
    }
    .method-body { font-size: 0.84rem; color: var(--muted); max-width: none; }
    .method-lead { margin: 0 0 0.65rem; color: var(--text); }
    .method-caveat {
      margin: 0 0 0.65rem; padding: 0.5rem 0.65rem; background: #fffbeb;
      border: 1px solid #fde68a; border-radius: 6px; font-size: 0.82rem;
    }
    .method-body h3 {
      margin: 0.85rem 0 0.35rem; font-size: 0.78rem; font-weight: 600; color: var(--text);
      text-transform: uppercase; letter-spacing: 0.04em;
    }
    .method-body dl { margin: 0; }
    .method-body dt { font-weight: 600; color: var(--text); margin-top: 0.45rem; font-size: 0.84rem; }
    .method-body dd { margin: 0.1rem 0 0; line-height: 1.4; }
    .method-body p { margin: 0.35rem 0 0; line-height: 1.45; }
    .method-body a { color: var(--accent); text-decoration: none; }
    .method-body a:hover { text-decoration: underline; }
    .method-body ul, .method-body ol { margin: 0.35rem 0 0; padding-left: 1.25rem; line-height: 1.45; }
    .method-body li { margin: 0.2rem 0; }
    .metric-block {
      margin: 1rem 0 0;
      padding: 0.85rem 1rem;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    .metric-block h4 {
      margin: 0 0 0.5rem;
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--text);
    }
    .metric-block p { margin: 0.4rem 0 0; }
    .metric-block p strong { color: var(--text); }
    .metric-block ul {
      margin: 0.35rem 0 0;
      padding-left: 1.25rem;
      line-height: 1.45;
    }
    .metric-block li { margin: 0.2rem 0; }
    .method-refs {
      margin-top: 1.25rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
      font-size: 0.8rem;
    }
    .method-refs h3 { margin: 0 0 0.5rem; font-size: 0.78rem; }
    .method-refs ul { margin: 0; padding-left: 1.25rem; line-height: 1.5; }
    .run-spec {
      margin: 0.5rem 0 0;
      padding: 0.5rem 0.65rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.82rem;
    }
    .run-spec code { font-size: 0.8rem; }
"""


def _methodology_tab_css(tab_prefix: str) -> str:
    results_id = f"{tab_prefix}-results"
    method_id = f"{tab_prefix}-methodology"
    return f"""
    #{results_id}:checked ~ .tab-bar label[for="{results_id}"],
    #{method_id}:checked ~ .tab-bar label[for="{method_id}"] {{
      color: var(--accent);
      border-bottom-color: var(--accent);
    }}
    #{results_id}:checked ~ .tab-panel-results {{ display: block; }}
    #{method_id}:checked ~ .tab-panel-methodology {{ display: block; }}
    """


def render_report_tabs(
    *,
    tab_prefix: str,
    results_html: str,
    methodology_html: str,
    results_label: str = "Results",
    methodology_label: str = "Metrics &amp; methodology",
) -> str:
    """Two-tab layout: results (default) and metrics/methodology."""
    results_id = f"{tab_prefix}-results"
    method_id = f"{tab_prefix}-methodology"
    return f"""
    <div class="report-tabs">
      <input type="radio" name="{tab_prefix}-tab" id="{results_id}" class="tab-input" checked>
      <input type="radio" name="{tab_prefix}-tab" id="{method_id}" class="tab-input">
      <nav class="tab-bar" aria-label="Report sections">
        <label for="{results_id}">{results_label}</label>
        <label for="{method_id}">{methodology_label}</label>
      </nav>
      <div class="tab-panel tab-panel-results">{results_html}</div>
      <div class="tab-panel tab-panel-methodology">{methodology_html}</div>
    </div>
    """


def report_tab_styles(tab_prefix: str) -> str:
    """CSS for tabs; append inside a <style> block after REPORT_TAB_CSS."""
    return REPORT_TAB_CSS + _methodology_tab_css(tab_prefix)


def _format_run_spec(name: str, backend: str, model_id: str | None, labels: list[str] | None, threshold: float) -> str:
    parts = [f"<code>{html.escape(name)}</code> — backend <code>{html.escape(backend)}</code>"]
    if model_id:
        parts.append(f"model <code>{html.escape(model_id)}</code>")
    if labels:
        label_s = ", ".join(html.escape(str(x)) for x in labels)
        parts.append(f"labels [{label_s}]")
    parts.append(f"threshold {threshold:g}")
    return " · ".join(parts)


def render_ner_methodology_content(benchmark: BenchmarkResult) -> str:
    """Inner HTML for the metrics & methodology tab."""
    config_path = html.escape(str(benchmark.config_path))
    output_dir = html.escape(str(benchmark.output_dir))

    try:
        cfg = load_benchmark_config(benchmark.config_path)
        runs_html = "".join(
            f'<div class="run-spec">{_format_run_spec(r.name, r.backend, r.model_id, r.labels, r.threshold)}</div>'
            for r in cfg.runs
        )
        datasets_list = ", ".join(f"<code>{html.escape(d)}</code>" for d in cfg.datasets)
        label_map = html.escape(cfg.label_map)
    except (OSError, ValueError):
        runs_html = "<p>Could not load benchmark config.</p>"
        datasets_list = "—"
        label_map = "unified"

    backends_run = {r.backend for r in benchmark.results}
    pattern_only = backends_run <= {"pattern"} and bool(backends_run)
    has_errors = any(r.error for r in benchmark.results)

    caveats: list[str] = []
    if pattern_only:
        caveats.append(
            "This run included only <code>pattern</code> "
            "(<code>--pattern-only</code> or filtered runs). ML scores are absent."
        )
    if has_errors:
        caveats.append(
            "One or more backend runs failed; see the leaderboard ERROR rows and stderr logs."
        )
    caveat_html = ""
    if caveats:
        caveat_html = '<p class="method-caveat">' + " ".join(caveats) + "</p>"

    n_results = len([r for r in benchmark.results if not r.error])
    n_total = len(benchmark.results)
    repeats = benchmark.repeats
    unstable = [r for r in benchmark.results if not r.error and not r.scores_reproducible]

    repeats_html = ""
    if repeats > 1:
        unstable_note = ""
        if unstable:
            names = ", ".join(f"<code>{html.escape(r.run_name)}</code>" for r in unstable[:5])
            unstable_note = (
                f'<p class="method-caveat">Non-reproducible scores detected for: {names}. '
                "Investigate non-deterministic backends or floating thresholds.</p>"
            )
        repeats_html = f"""
          <h3>Repeated trials ({repeats}×)</h3>
          <p>Each backend×dataset cell is executed <strong>{repeats}</strong> times. Before every trial,
          <code>clear_backend_cache()</code> runs so timing includes model load and inference (mitigates
          one-off CPU allocation spikes).</p>
          <ul>
            <li><strong>Latency</strong> — mean ± std of ms/example across trials; min–max in the stability table.</li>
            <li><strong>Scores</strong> — strict/relaxed F1 and TP/FP/FN must match every trial; otherwise flagged unstable.</li>
            <li><strong>Radar Speed</strong> — uses mean latency vs the slowest mean in that dataset.</li>
          </ul>
          {unstable_note}
          <p>CLI: <code>--repeats N</code> or <code>repeats: N</code> in benchmark YAML (default 5).</p>
        """

    return f"""
        <div class="method-body">
          <p class="method-lead">Each gold example is loaded from JSONL, passed through a NER backend
          (<code>detect_entities</code>), and scored against annotated spans after label normalization.
          This report compares <strong>backend implementations</strong> on fixed gold data — not end-to-end
          application quality or downstream retrieval.</p>
          {caveat_html}

          <h3>This run</h3>
          <dl>
            <dt>Config</dt>
            <dd><code>{config_path}</code></dd>
            <dt>Output</dt>
            <dd><code>{output_dir}</code></dd>
            <dt>Repeats per cell</dt>
            <dd>{repeats}</dd>
            <dt>Completed cells</dt>
            <dd>{n_results} successful backend×dataset runs of {n_total} attempted.</dd>
            <dt>Label map</dt>
            <dd><code>{label_map}</code> (see <code>benchmark/config/label_maps.yaml</code>)</dd>
            <dt>Datasets in config</dt>
            <dd>{datasets_list}</dd>
          </dl>

          {repeats_html}
          <h3>Benchmark process</h3>
          <ol>
            <li>Read <code>benchmark/config/compare_backends.yaml</code> for run definitions and dataset names.</li>
            <li>Load each dataset from <code>benchmark/datasets/&lt;name&gt;.jsonl</code> (or path in config).</li>
            <li>For each (run, dataset) pair: repeat <code>{repeats}</code> time(s) — clear backend cache, run all examples,
            score spans, record wall-clock latency (model load + inference each trial).</li>
            <li>Aggregate trials: mean ± std latency; verify score reproducibility across repeats.</li>
            <li>Write <code>metrics.json</code>, <code>report.md</code>, and this <code>report.html</code>.</li>
          </ol>
          <p>CLI: <code>uv run python benchmark/run_benchmark.py</code> —
          flags <code>--pattern-only</code>, <code>--datasets</code>, <code>--runs</code>,
          <code>--max-examples</code>, <code>--output</code>.</p>

          <h3>Configured backends</h3>
          {runs_html}

          <h3>Gold data format</h3>
          <p>JSONL lines with <code>id</code>, <code>text</code>, and <code>entities</code> (each entity:
          <code>text</code>, <code>label</code>, <code>start</code>, <code>end</code> character offsets).
          Predictions without resolvable character offsets in the source text are skipped and counted in
          <code>skipped_predictions</code>.</p>
          <dl>
            <dt><code>marfago_gold</code></dt>
            <dd>Domain snippets (person, organization, location, arxiv_id, year, …). Intended for
            <code>pattern</code> and unified-label ML backends.</dd>
            <dt><code>conll_dev_sample</code></dt>
            <dd>Short English news sentences with CoNLL-style PER/ORG/LOC/MISC gold. BERT-NER is a strong
            fit; <code>pattern</code> is a weak baseline here (different entity types).</dd>
          </dl>

          <h3>Scoring metrics</h3>

          <div class="metric-block">
            <h4>Strict span F1</h4>
            <p><strong>Objective.</strong> Exact entity detection: same unified label and identical
            character span <code>(start, end)</code> as gold.</p>
            <p><strong>How it is computed.</strong> Greedy one-to-one matching per example: each prediction
            matches at most one gold span with the same label and exact offsets; unmatched predictions are FP,
            unmatched gold are FN. Micro-averaged precision, recall, and F1 across all examples in the run.</p>
            <p><strong>Leaderboard.</strong> Global table is sorted by strict F1 (higher is better). Rank #1 row
            is highlighted in green.</p>
          </div>

          <div class="metric-block">
            <h4>Relaxed span F1</h4>
            <p><strong>Objective.</strong> Partial credit when boundaries differ slightly but the entity is
            largely correct.</p>
            <p><strong>How it is computed.</strong> Same label required; spans match if
            intersection ÷ union (IoU) ≥ <strong>0.5</strong>. Also micro-averaged across the run.</p>
            <p><strong>Use.</strong> Compare backends when tokenization or whitespace causes small boundary
            shifts; do not treat relaxed F1 as “good enough” for production without manual review.</p>
          </div>

          <div class="metric-block">
            <h4>Latency (ms / example)</h4>
            <p><strong>Objective.</strong> Wall-clock time for the entire backend pass over the dataset,
            divided by number of scored examples.</p>
            <p><strong>Includes.</strong> Model load (first call in that run), inference, and Python overhead.
            <strong>Excludes.</strong> Other backends’ runs; no warm-up run is performed separately.</p>
            <p><strong>Caveats.</strong> Small datasets exaggerate load time; CPU vs GPU and batching differ by
            backend. Use for rough operational comparison on the same machine, not absolute SLA numbers.</p>
          </div>

          <div class="metric-block">
            <h4>Radar chart — Speed axis</h4>
            <p><strong>Per dataset only.</strong> Speed = <code>1 − (latency ÷ max latency among backends
            on that dataset)</code>. The slowest backend in that chart scores <strong>0</strong>; the fastest
            scores <strong>1</strong>. A zero on Speed does <em>not</em> mean “no measurement” — it means
            slowest in that comparison.</p>
            <p>Other radar axes plot strict F1, relaxed F1, precision, and recall on a 0–1 scale (higher is
            better). Polygon <strong>area</strong> is a visual composite index only; it weights all axes equally.</p>
          </div>

          <h3>Label normalization</h3>
          <p>Backend labels (e.g. CoNLL <code>PER</code>, GLiNER <code>person</code>) are mapped through
          <code>label_maps.yaml</code> before scoring. Comparing F1 across datasets with incompatible label
          schemes is misleading — e.g. do not rank <code>pattern</code> on CoNLL using strict F1 alone.</p>

          <h3>How to use this scorecard</h3>
          <ul>
            <li>Pick the dataset that matches your deployment labels and language.</li>
            <li>Prefer <strong>strict F1</strong> for span-extraction contracts; use <strong>relaxed F1</strong>
            to diagnose boundary issues.</li>
            <li>Read <strong>latency</strong> from the table (absolute ms); use radar Speed only for relative
            ranking within one dataset.</li>
            <li>Treat <code>pattern</code> as a fast regex baseline, not a ceiling for ML quality.</li>
          </ul>

          <div class="method-refs">
            <h3>References &amp; data sources</h3>
            <ul>
              <li>CoNLL-2003 export: <a href="https://huggingface.co/datasets/eriktks/conll2003">eriktks/conll2003</a></li>
              <li>Default BERT-NER: <a href="https://huggingface.co/dslim/bert-base-NER">dslim/bert-base-NER</a></li>
              <li>GLiNER: <a href="https://huggingface.co/urchade/gliner_medium-v2.1">urchade/gliner_medium-v2.1</a></li>
              <li>Project docs: <code>docs/benchmarks.md</code></li>
            </ul>
          </div>
        </div>
        """

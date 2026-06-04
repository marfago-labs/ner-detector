"""HTML rendering for label confusion matrices."""

from __future__ import annotations

import html

from ner_detector.eval.confusion import MISSED_COL, SPURIOUS_ROW, LabelConfusionMatrix

CONFUSION_MATRIX_CSS = """
    .confusion-section { margin-top: 1.25rem; }
    .confusion-section h3 { font-size: 1rem; margin: 0 0 0.5rem; }
    .confusion-section h4 {
      font-size: 0.88rem; font-weight: 600; margin: 1rem 0 0.35rem;
      color: var(--muted);
    }
    .confusion-note {
      font-size: 0.82rem; color: var(--muted); margin: 0 0 0.65rem;
    }
    .confusion-wrap {
      overflow-x: auto; margin-bottom: 0.75rem;
    }
    table.confusion-matrix {
      width: auto; min-width: 280px; border-collapse: collapse;
      font-size: 0.78rem;
    }
    table.confusion-matrix th, table.confusion-matrix td {
      padding: 0.35rem 0.5rem; text-align: center; border: 1px solid var(--border);
      min-width: 2.5rem;
    }
    table.confusion-matrix th.corner {
      text-align: left; background: #f8fafc; font-weight: 600;
    }
    table.confusion-matrix th.col-head {
      writing-mode: horizontal-tb; max-width: 6rem;
      word-break: break-word; background: #f8fafc;
    }
    table.confusion-matrix th.row-head {
      text-align: left; background: #f8fafc; font-weight: 500;
      white-space: nowrap;
    }
    table.confusion-matrix td.cell-zero { color: #cbd5e1; }
    table.confusion-matrix td.cell-hit {
      background: #dcfce7; font-weight: 600;
    }
    table.confusion-matrix td.cell-off {
      background: #fff7ed;
    }
    table.confusion-matrix td.cell-miss, table.confusion-matrix td.cell-spurious {
      background: #fef2f2;
    }
"""


def _cell_class(gold_label: str, pred_label: str, count: int) -> str:
    if count == 0:
        return "cell-zero"
    if gold_label == SPURIOUS_ROW:
        return "cell-spurious"
    if pred_label == MISSED_COL:
        return "cell-miss"
    if gold_label == pred_label:
        return "cell-hit"
    return "cell-off"


def _cell_background(count: int, max_count: int) -> str:
    if count <= 0 or max_count <= 0:
        return ""
    intensity = min(1.0, count / max_count)
    alpha = 0.08 + 0.35 * intensity
    return f"background-color: rgba(29, 78, 216, {alpha:.3f});"


def render_confusion_matrix_html(
    matrix: LabelConfusionMatrix,
    *,
    caption: str,
) -> str:
    """Render one gold×pred count table."""
    if matrix.is_empty():
        return ""

    rows = matrix.row_labels()
    cols = matrix.col_labels()
    if not rows or not cols:
        return ""

    max_count = max(matrix.counts.values()) if matrix.counts else 0
    header = "".join(
        f'<th class="col-head" scope="col">{html.escape(col)}</th>' for col in cols
    )
    body_rows: list[str] = []
    for gold in rows:
        cells: list[str] = []
        for pred in cols:
            count = matrix.get(gold, pred)
            css = _cell_class(gold, pred, count)
            extra = ""
            if count > 0 and gold not in {SPURIOUS_ROW} and pred not in {MISSED_COL}:
                if gold != pred:
                    extra = f' style="{_cell_background(count, max_count)}"'
            cells.append(
                f'<td class="{css}"{extra}>{count if count else "·"}</td>'
            )
        body_rows.append(
            "<tr>"
            f'<th class="row-head" scope="row">{html.escape(gold)}</th>'
            f'{"".join(cells)}'
            "</tr>"
        )

    return f"""
    <div class="confusion-wrap">
      <p class="confusion-note">{html.escape(caption)} · total paired counts: {matrix.total()}</p>
      <table class="confusion-matrix" aria-label="{html.escape(caption)}">
        <thead>
          <tr>
            <th class="corner" scope="col">Gold ↓ / Pred →</th>
            {header}
          </tr>
        </thead>
        <tbody>
          {"".join(body_rows)}
        </tbody>
      </table>
    </div>
    """


def render_run_confusion_html(
    run_name: str,
    matrix_relaxed: LabelConfusionMatrix,
    matrix_strict: LabelConfusionMatrix,
) -> str:
    """Render relaxed + strict matrices for one backend run."""
    if matrix_relaxed.is_empty() and matrix_strict.is_empty():
        return ""
    relaxed_html = render_confusion_matrix_html(
        matrix_relaxed,
        caption="Relaxed span pairing (IoU ≥ 0.5); labels may differ",
    )
    strict_html = render_confusion_matrix_html(
        matrix_strict,
        caption="Strict span pairing (exact start/end); labels may differ",
    )
    return f"""
    <div class="confusion-run">
      <h4><code>{html.escape(run_name)}</code></h4>
      {relaxed_html}
      {strict_html}
    </div>
    """

#!/usr/bin/env python3
"""Apply marfago-labs theme + site chrome to an existing benchmark HTML report."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ner_detector.eval.curve_svg import curve_chart_css
from ner_detector.eval.html_report import REPORT_PAGE_EXTRA_CSS
from ner_detector.eval.lab_chrome import load_lab_theme_css, site_footer, site_header
from ner_detector.eval.radar_svg import RADAR_CHART_CSS
from ner_detector.eval.report_methodology import report_tab_styles


def retheme_report_html(path: Path, *, footer_note: str | None = None) -> None:
    html = path.read_text(encoding="utf-8")
    wrap_match = re.search(r'<div class="wrap">\s*(.*?)\s*</div>\s*</body>', html, re.DOTALL)
    if not wrap_match:
        raise ValueError(f'Could not find <div class="wrap"> in {path}')
    inner = wrap_match.group(1)
    inner = re.sub(r"\s*<footer>.*?</footer>\s*$", "", inner, flags=re.DOTALL)

    title_match = re.search(r"<title>(.*?)</title>", html)
    title = title_match.group(1) if title_match else "NER backend benchmark — ner-detector"
    note = footer_note or "ner-detector · benchmark/run_benchmark.py"
    has_curves = "ner-bench-curves" in inner or "curve-chart" in inner

    themed = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
{load_lab_theme_css()}
    {REPORT_PAGE_EXTRA_CSS}
    {report_tab_styles("ner-bench", has_curves=has_curves)}
    {RADAR_CHART_CSS}
    {curve_chart_css() if has_curves else ""}
  </style>
</head>
<body>
  {site_header(active="projects")}
  <main class="main">
    {inner}
  </main>
  {site_footer(note=note)}
</body>
</html>
"""
    path.write_text(themed, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="HTML report files to retheme")
    args = parser.parse_args(argv)
    for path in args.paths:
        retheme_report_html(path)
        print(f"Rethemed {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

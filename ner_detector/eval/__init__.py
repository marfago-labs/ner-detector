"""NER evaluation against gold datasets."""

from ner_detector.eval.html_report import render_html_report
from ner_detector.eval.metrics import EntityScores, ScoreSummary, score_example
from ner_detector.eval.report import write_report
from ner_detector.eval.runner import BenchmarkResult, run_benchmark
from ner_detector.eval.types import GoldEntity, GoldExample

__all__ = [
    "BenchmarkResult",
    "EntityScores",
    "GoldEntity",
    "GoldExample",
    "ScoreSummary",
    "render_html_report",
    "run_benchmark",
    "score_example",
    "write_report",
]

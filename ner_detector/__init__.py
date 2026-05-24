"""Named entity recognition with pluggable model backends."""

from ner_detector.detect import detect_entities
from ner_detector.types import DetectedEntity, NerBackend

__all__ = [
    "DetectedEntity",
    "NerBackend",
    "detect_entities",
]

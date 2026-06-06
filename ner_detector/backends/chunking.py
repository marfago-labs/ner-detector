"""Text chunking and cross-chunk entity resolution for document-level NER."""

from __future__ import annotations

from collections.abc import Callable

from ner_detector.types import DetectedEntity

_DEFAULT_MAX_CHUNK_CHARS = 4000
_DEFAULT_CHUNK_OVERLAP = 200


def chunk_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """Split long text into overlapping chunks."""
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= max_chars:
        return [stripped]
    chunks: list[str] = []
    start = 0
    while start < len(stripped):
        end = min(len(stripped), start + max_chars)
        chunks.append(stripped[start:end])
        if end >= len(stripped):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def chunk_offset(source: str, chunk: str) -> int:
    """Return character offset of *chunk* in *source*, or 0 if not found."""
    idx = source.find(chunk)
    return idx if idx >= 0 else 0


def shift_entities(
    entities: list[DetectedEntity],
    *,
    offset: int,
    source: str,
) -> list[DetectedEntity]:
    """Translate chunk-local spans to absolute positions in *source*."""
    shifted: list[DetectedEntity] = []
    for ent in entities:
        if ent.start is None or ent.end is None:
            shifted.append(ent)
            continue
        start = ent.start + offset
        end = ent.end + offset
        if end > len(source):
            continue
        shifted.append(
            DetectedEntity(
                text=source[start:end],
                label=ent.label,
                score=ent.score,
                start=start,
                end=end,
            ),
        )
    return shifted


def merge_overlapping_entities(
    entities: list[DetectedEntity],
    *,
    source: str,
) -> list[DetectedEntity]:
    """Merge same-label spans that overlap or touch (document-level smoothing)."""
    positioned = [
        ent
        for ent in entities
        if ent.start is not None and ent.end is not None and ent.end > ent.start
    ]
    unpositioned = [
        ent
        for ent in entities
        if ent.start is None or ent.end is None or ent.end <= ent.start
    ]
    positioned.sort(key=lambda e: (e.label, e.start or 0, e.end or 0))
    merged: list[DetectedEntity] = []
    for ent in positioned:
        if not merged:
            merged.append(ent)
            continue
        prev = merged[-1]
        if prev.label != ent.label:
            merged.append(ent)
            continue
        prev_start = prev.start or 0
        prev_end = prev.end or 0
        ent_start = ent.start or 0
        ent_end = ent.end or 0
        if ent_start <= prev_end + 1:
            start = min(prev_start, ent_start)
            end = max(prev_end, ent_end)
            score = max(prev.score or 0.0, ent.score or 0.0) or None
            merged[-1] = DetectedEntity(
                text=source[start:end],
                label=prev.label,
                score=round(score, 4) if score is not None else None,
                start=start,
                end=end,
            )
            continue
        merged.append(ent)
    combined = merged + unpositioned
    combined.sort(key=lambda e: (e.start or 0, e.end or 0))
    return combined


def collect_chunked_entities(
    source: str,
    chunks: list[str],
    *,
    detect_chunk: Callable[[str], list[DetectedEntity]],
) -> list[DetectedEntity]:
    """Run *detect_chunk* per chunk and merge with cross-chunk resolution."""
    found: list[DetectedEntity] = []
    seen: set[tuple[int, int, str]] = set()
    for chunk in chunks:
        offset = chunk_offset(source, chunk)
        for ent in detect_chunk(chunk):
            shifted = shift_entities([ent], offset=offset, source=source)
            for item in shifted:
                key = (item.start or 0, item.end or 0, item.label)
                if key in seen:
                    continue
                seen.add(key)
                found.append(item)
    return merge_overlapping_entities(found, source=source)

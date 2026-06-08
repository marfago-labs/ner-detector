"""Validate gold JSONL datasets before benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field

from ner_detector.eval.types import GoldEntity, GoldExample

ARXIV_GOLD_LABELS = frozenset(
    {
        "model",
        "dataset",
        "benchmark",
        "metric",
        "method",
        "number",
        "organization",
    }
)


@dataclass
class GoldValidationIssue:
    example_id: str
    message: str
    entity_text: str | None = None


@dataclass
class GoldValidationReport:
    dataset_name: str
    n_examples: int
    n_entities: int
    issues: list[GoldValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def raise_if_invalid(self) -> None:
        if self.ok:
            return
        sample = "; ".join(f"{issue.example_id}: {issue.message}" for issue in self.issues[:5])
        extra = f" (+{len(self.issues) - 5} more)" if len(self.issues) > 5 else ""
        raise ValueError(f"Invalid gold dataset {self.dataset_name!r}: {sample}{extra}")


def _check_entity_bounds(example: GoldExample, entity: GoldEntity) -> GoldValidationIssue | None:
    if entity.start >= entity.end:
        return GoldValidationIssue(
            example.id,
            f"invalid span offsets ({entity.start}, {entity.end})",
            entity.text,
        )
    if entity.end > len(example.text):
        return GoldValidationIssue(
            example.id,
            f"span end {entity.end} exceeds text length {len(example.text)}",
            entity.text,
        )
    return None


def _check_entity_surface(example: GoldExample, entity: GoldEntity) -> GoldValidationIssue | None:
    surface = example.text[entity.start : entity.end]
    if surface != entity.text:
        return GoldValidationIssue(
            example.id,
            f"span text mismatch: gold={entity.text!r} slice={surface!r}",
            entity.text,
        )
    return None


def validate_gold_examples(
    examples: list[GoldExample],
    *,
    dataset_name: str = "gold",
    allowed_labels: frozenset[str] | None = None,
) -> GoldValidationReport:
    """Check span integrity and optional label vocabulary."""
    issues: list[GoldValidationIssue] = []
    n_entities = 0
    for example in examples:
        if not example.text.strip():
            issues.append(GoldValidationIssue(example.id, "empty example text"))
        seen_spans: set[tuple[int, int, str]] = set()
        for entity in example.entities:
            n_entities += 1
            if not entity.label.strip():
                issues.append(GoldValidationIssue(example.id, "empty entity label", entity.text))
            if allowed_labels is not None and entity.label not in allowed_labels:
                issues.append(
                    GoldValidationIssue(
                        example.id,
                        f"unexpected label {entity.label!r}",
                        entity.text,
                    )
                )
            for check in (_check_entity_bounds, _check_entity_surface):
                issue = check(example, entity)
                if issue is not None:
                    issues.append(issue)
            key = (entity.start, entity.end, entity.label)
            if key in seen_spans:
                issues.append(
                    GoldValidationIssue(
                        example.id,
                        f"duplicate gold span {key}",
                        entity.text,
                    )
                )
            seen_spans.add(key)
    return GoldValidationReport(
        dataset_name=dataset_name,
        n_examples=len(examples),
        n_entities=n_entities,
        issues=issues,
    )


def validate_arxiv_gold(examples: list[GoldExample]) -> GoldValidationReport:
    """Validate the curated arxiv_gold schema and span integrity."""
    return validate_gold_examples(
        examples,
        dataset_name="arxiv_gold",
        allowed_labels=ARXIV_GOLD_LABELS,
    )

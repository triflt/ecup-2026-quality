from __future__ import annotations

"""Label-blind, provenance-preserving description packing."""

import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR_PATH = (
    ROOT / "experiments/490_evidence_grounded_explanations/src/evidence_grounding/extractor.py"
)
DESCRIPTION_BUDGET = 1800
HEAD_SHARE = 0.70
BLOCKED_REASONS = {"conflicting_evidence", "ambiguous_negation", "ambiguous_scope"}
SENTENCE_END = re.compile(r"[.!?;]+(?=\s|$)")


def _load_extractor():
    package_root = EXTRACTOR_PATH.parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    spec = importlib.util.spec_from_file_location(
        "_exp490_evidence_extractor_for_context_pack", EXTRACTOR_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError("frozen evidence extractor is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evidence = _load_extractor()


@dataclass(frozen=True)
class EvidenceSpan:
    start: int
    end: int
    sentence_start: int
    sentence_end: int


@dataclass(frozen=True)
class PackedDescription:
    text: str
    source_positions: tuple[int, ...]
    raw_intervals: tuple[tuple[int, int], ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    blocked_reason: str | None
    packable: bool

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    @property
    def provenance_sha256(self) -> str:
        payload = "\n".join(
            f"{position}:{raw_start}:{raw_end}"
            for position, (raw_start, raw_end) in zip(self.source_positions, self.raw_intervals)
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def sentence_intervals(text: str) -> tuple[tuple[int, int], ...]:
    intervals: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        end = match.end()
        if start < end:
            intervals.append((start, end))
        start = end
    if start < len(text):
        intervals.append((start, len(text)))
    return tuple(intervals)


def containing_sentence(
    intervals: tuple[tuple[int, int], ...], *, start: int, end: int
) -> tuple[int, int]:
    for sentence_start, sentence_end in intervals:
        if sentence_start <= start and end <= sentence_end:
            return sentence_start, sentence_end
    raise ValueError("safe evidence span is not contained in a sentence")


def safe_description_spans(
    *, row_id: str, category: str, name: str, description: str
) -> tuple[tuple[EvidenceSpan, ...], str | None]:
    surface = evidence.surface_text(description)
    results = [
        evidence.extract_evidence(
            row_id=str(row_id),
            category=str(category),
            name=str(name),
            description=str(description),
            frozen_prediction=frozen_prediction,
        )
        for frozen_prediction in (0, 1)
    ]
    blocked = sorted(
        {
            str(result.fallback_reason)
            for result in results
            if result.fallback_reason in BLOCKED_REASONS
        }
    )
    if blocked:
        return (), "+".join(blocked)
    intervals = sentence_intervals(surface.text)
    spans: set[tuple[int, int, int, int]] = set()
    for result in results:
        if result.status != "SAFE" or result.source != "description":
            continue
        if (
            result.polarity == "ambiguous"
            or result.scope == "ambiguous"
            or not result.checks.exact_span
            or not result.checks.no_conflict
            or result.surface_start is None
            or result.surface_end is None
            or result.exact_surface_span is None
        ):
            continue
        start, end = int(result.surface_start), int(result.surface_end)
        if surface.text[start:end] != result.exact_surface_span:
            raise ValueError("frozen extractor returned a non-exact description span")
        sentence_start, sentence_end = containing_sentence(intervals, start=start, end=end)
        spans.add((start, end, sentence_start, sentence_end))
    return tuple(EvidenceSpan(*values) for values in sorted(spans)), None


def _positions_for_pack(
    *, length: int, evidence_spans: tuple[EvidenceSpan, ...], budget: int
) -> tuple[tuple[int, ...], bool]:
    evidence_positions = {
        position
        for span in evidence_spans
        for position in range(span.sentence_start, span.sentence_end)
    }
    if len(evidence_positions) > max(0, budget - 2):
        return (), False
    remaining = budget - len(evidence_positions)
    head_quota = max(1, int(remaining * HEAD_SHARE))
    tail_quota = max(1, remaining - head_quota)
    selected = set(evidence_positions)
    for position in range(length):
        if position not in selected:
            selected.add(position)
            head_quota -= 1
            if head_quota == 0:
                break
    for position in range(length - 1, -1, -1):
        if position not in selected:
            selected.add(position)
            tail_quota -= 1
            if tail_quota == 0:
                break
    if len(selected) > budget:
        raise RuntimeError("context pack exceeded its immutable budget")
    return tuple(sorted(selected)), True


def pack_description(
    *,
    row_id: str,
    category: str,
    name: str,
    description: str,
    budget: int = DESCRIPTION_BUDGET,
) -> PackedDescription:
    if budget != DESCRIPTION_BUDGET:
        raise ValueError(f"context pack requires exact budget {DESCRIPTION_BUDGET}")
    surface = evidence.surface_text(description)
    spans, blocked_reason = safe_description_spans(
        row_id=row_id,
        category=category,
        name=name,
        description=description,
    )
    if len(surface.text) <= budget:
        positions = tuple(range(len(surface.text)))
        return PackedDescription(
            text=surface.text,
            source_positions=positions,
            raw_intervals=surface.raw_intervals,
            evidence_spans=spans,
            blocked_reason=blocked_reason,
            packable=True,
        )
    positions, packable = _positions_for_pack(
        length=len(surface.text), evidence_spans=spans, budget=budget
    )
    if not packable:
        return PackedDescription(
            text="",
            source_positions=(),
            raw_intervals=(),
            evidence_spans=spans,
            blocked_reason="evidence_sentences_exceed_budget",
            packable=False,
        )
    text = "".join(surface.text[position] for position in positions)
    if len(text) > budget or len(set(positions)) != len(positions):
        raise RuntimeError("context pack budget/provenance invariant failed")
    return PackedDescription(
        text=text,
        source_positions=positions,
        raw_intervals=tuple(surface.raw_intervals[position] for position in positions),
        evidence_spans=spans,
        blocked_reason=blocked_reason,
        packable=True,
    )


def baseline_positions(description: str, *, budget: int = DESCRIPTION_BUDGET) -> tuple[int, ...]:
    surface = evidence.surface_text(description).text
    if len(surface) <= budget:
        return tuple(range(len(surface)))
    head = int(budget * HEAD_SHARE)
    head_end = len(surface[:head].rstrip())
    tail_start = len(surface) - len(surface[-(budget - head) :].lstrip())
    return (*range(head_end), *range(tail_start, len(surface)))


def retained_evidence(positions: tuple[int, ...], spans: tuple[EvidenceSpan, ...]) -> int:
    selected = set(positions)
    return sum(
        all(position in selected for position in range(span.start, span.end)) for span in spans
    )

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP490_SRC = ROOT / "experiments" / "490_evidence_grounded_explanations" / "src"
if str(EXP490_SRC) not in sys.path:
    sys.path.insert(0, str(EXP490_SRC))

from evidence_grounding import CONCEPT_VOCABULARY, extract_evidence, surface_text

FORMAT_VERSION = "grounded_auxiliary_target_v1"
NO_SAFE_EVIDENCE = "NO_SAFE_EVIDENCE"
CLOSED_CONCEPTS = frozenset({*CONCEPT_VOCABULARY, NO_SAFE_EVIDENCE})
_FIRST_ATOMIC_VERDICT = re.compile(r"^([01])(?=\s|$)")
_FIELD_ORDER = ("CONCEPT", "SOURCE", "START", "END", "EVIDENCE")


@dataclass(frozen=True)
class ParsedTarget:
    verdict: int
    concept: str
    source: str
    start: int
    end: int
    evidence: str


def parse_first_atomic_verdict(output: str) -> int:
    match = _FIRST_ATOMIC_VERDICT.match(str(output))
    if match is None:
        raise ValueError("output must begin with one atomic 0/1 verdict token")
    return int(match.group(1))


@lru_cache(maxsize=32_768)
def build_structured_target(
    *,
    row_id: str,
    category: str,
    name: str,
    description: str,
    gold_verdict: int,
) -> str:
    result = extract_evidence(
        row_id=str(row_id),
        category=str(category),
        name=str(name or ""),
        description=str(description or ""),
        frozen_prediction=int(gold_verdict),
    )
    if result.status == "SAFE":
        assert result.concept in CONCEPT_VOCABULARY
        assert result.source in {"name", "description"}
        assert result.surface_start is not None and result.surface_end is not None
        assert result.exact_surface_span is not None
        concept = result.concept
        source = result.source
        start = result.surface_start
        end = result.surface_end
        evidence = result.exact_surface_span
    else:
        concept = NO_SAFE_EVIDENCE
        source = "none"
        start = -1
        end = -1
        evidence = ""
    return (
        f"{int(gold_verdict)}\n"
        f"CONCEPT={concept}\n"
        f"SOURCE={source}\n"
        f"START={start}\n"
        f"END={end}\n"
        f"EVIDENCE={json.dumps(evidence, ensure_ascii=False)}"
    )


def parse_structured_target(output: str) -> ParsedTarget:
    lines = str(output).splitlines()
    if len(lines) != 6:
        raise ValueError("structured output must contain exactly six lines")
    verdict = parse_first_atomic_verdict(lines[0])
    values: dict[str, str] = {}
    for expected, line in zip(_FIELD_ORDER, lines[1:], strict=True):
        prefix = f"{expected}="
        if not line.startswith(prefix):
            raise ValueError(f"expected {expected} field")
        values[expected] = line[len(prefix) :]
    concept = values["CONCEPT"]
    if concept not in CLOSED_CONCEPTS:
        raise ValueError("concept is outside the frozen closed vocabulary")
    source = values["SOURCE"]
    if source not in {"name", "description", "none"}:
        raise ValueError("unsupported evidence source")
    try:
        start = int(values["START"])
        end = int(values["END"])
        evidence = json.loads(values["EVIDENCE"])
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid offset or JSON evidence field") from error
    if not isinstance(evidence, str):
        raise TypeError("evidence must be a JSON string")
    if concept == NO_SAFE_EVIDENCE:
        if (source, start, end, evidence) != ("none", -1, -1, ""):
            raise ValueError("NO_SAFE_EVIDENCE must use the frozen empty fallback")
    elif source == "none" or not (0 <= start < end) or not evidence:
        raise ValueError("SAFE concept requires a source, offsets and evidence")
    return ParsedTarget(verdict, concept, source, start, end, evidence)


def validate_exact_evidence(parsed: ParsedTarget, *, name: str, description: str) -> bool:
    if parsed.concept == NO_SAFE_EVIDENCE:
        return True
    raw = name if parsed.source == "name" else description
    surface = surface_text(raw).text
    return (
        0 <= parsed.start < parsed.end <= len(surface)
        and surface[parsed.start : parsed.end] == parsed.evidence
    )


def render_explanation(parsed: ParsedTarget) -> str:
    verdict = "не бан" if parsed.verdict == 1 else "бан"
    if parsed.concept == NO_SAFE_EVIDENCE:
        reason = "однозначное текстовое основание не найдено"
    else:
        reason = f"{parsed.concept}: «{parsed.evidence}»"
    return f"{reason}. Вердикт: {verdict}."

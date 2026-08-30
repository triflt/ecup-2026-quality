"""Conservative text-span evidence extraction for experiment 490.

The module receives an already frozen binary prediction.  It never predicts a
label and never receives the gold label.  Every successful explanation is built
from an exact surface substring and a closed policy concept.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

Status = Literal["SAFE", "NO_SAFE_EVIDENCE"]
Source = Literal["name", "description", "name_and_description", "none"]
Polarity = Literal["asserted", "negated", "conditional", "bounded_absence", "ambiguous"]
Scope = Literal[
    "product",
    "included",
    "excluded",
    "compatibility",
    "empty",
    "integrated_component",
    "reference_only",
    "ambiguous",
]

VOCABULARY_VERSION = "policy_concepts_v1"
_VOCAB_PATH = Path(__file__).resolve().parents[2] / "policy_concepts_v1.json"
_VOCAB_BYTES = _VOCAB_PATH.read_bytes()
VOCABULARY_SHA256 = hashlib.sha256(_VOCAB_BYTES).hexdigest()
_VOCAB_DOCUMENT = json.loads(_VOCAB_BYTES)
if _VOCAB_DOCUMENT.get("version") != VOCABULARY_VERSION:
    raise RuntimeError("Policy vocabulary version does not match the extractor")

CONCEPT_VOCABULARY: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        item["id"]: MappingProxyType(
            {
                "category": item["category"],
                "supports_prediction": item["supports_prediction"],
            }
        )
        for item in _VOCAB_DOCUMENT["concepts"]
    }
)

_CATEGORIES = {"БАД", "Легковоспламеняющиеся"}
_FALLBACK_COMMENT = (
    "По доступному названию и описанию не найдено однозначного текстового "
    "фрагмента, подтверждающего вынесенный вердикт."
)
_FORBIDDEN_COMMENT_WORDS = ("изображение", "упаковка", "видно")
_MIN_COMMENT_LENGTH = 50
_MAX_COMMENT_LENGTH = 300
_MIN_QUOTE_LENGTH = 8
_MAX_QUOTE_LENGTH = 160


@dataclass(frozen=True)
class SurfaceText:
    """Display text plus a per-character map to the original raw field."""

    raw: str
    text: str
    raw_intervals: tuple[tuple[int, int], ...]

    def raw_interval(self, start: int, end: int) -> tuple[int, int]:
        if not (0 <= start < end <= len(self.text)):
            raise ValueError("Surface offsets are outside the display text")
        return self.raw_intervals[start][0], self.raw_intervals[end - 1][1]


@dataclass(frozen=True)
class EvidenceChecks:
    exact_span: bool
    verdict_locked: bool
    length_ok: bool
    no_forbidden_tag: bool
    no_conflict: bool


@dataclass(frozen=True)
class EvidenceResult:
    status: Status
    row_id: str
    category: str
    frozen_prediction: int
    verdict: str
    source: Source
    raw_start: int | None
    raw_end: int | None
    surface_start: int | None
    surface_end: int | None
    exact_surface_span: str | None
    concept: str | None
    polarity: Polarity
    scope: Scope
    comment: str
    fallback_reason: str | None
    checks: EvidenceChecks
    vocabulary_version: str = VOCABULARY_VERSION
    vocabulary_sha256: str = VOCABULARY_SHA256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _MatchText:
    text: str
    surface_indices: tuple[int, ...]


@dataclass(frozen=True)
class _Clause:
    source: Literal["name", "description"]
    surface: SurfaceText
    start: int
    end: int
    text: str
    match: str


@dataclass(frozen=True)
class _Candidate:
    clause: _Clause
    concept: str
    supports_prediction: int
    polarity: Polarity
    scope: Scope
    directness: int
    specificity: int


def _coerce_raw(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def surface_text(value: object) -> SurfaceText:
    """Create safe display text while retaining offsets into the original field."""

    raw = _coerce_raw(value)
    expanded: list[tuple[str, tuple[int, int]]] = []
    cursor = 0
    while cursor < len(raw):
        if raw[cursor] == "<":
            tag_end = raw.find(">", cursor + 1)
            if tag_end != -1:
                expanded.append((" ", (cursor, tag_end + 1)))
                cursor = tag_end + 1
                continue
        if raw[cursor] == "&":
            entity_match = re.match(r"&(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]+);", raw[cursor:])
            if entity_match:
                token = entity_match.group(0)
                decoded = html.unescape(token)
                if decoded != token:
                    interval = (cursor, cursor + len(token))
                    expanded.extend((character, interval) for character in decoded)
                    cursor += len(token)
                    continue
        expanded.append((raw[cursor], (cursor, cursor + 1)))
        cursor += 1

    collapsed: list[str] = []
    intervals: list[tuple[int, int]] = []
    index = 0
    while index < len(expanded):
        character, interval = expanded[index]
        if character.isspace():
            run_start = interval[0]
            run_end = interval[1]
            index += 1
            while index < len(expanded) and expanded[index][0].isspace():
                run_end = expanded[index][1][1]
                index += 1
            if collapsed and index < len(expanded):
                collapsed.append(" ")
                intervals.append((run_start, run_end))
            continue
        collapsed.append(character)
        intervals.append(interval)
        index += 1

    return SurfaceText(raw=raw, text="".join(collapsed), raw_intervals=tuple(intervals))


def _match_text(surface: str) -> _MatchText:
    characters: list[str] = []
    indices: list[int] = []
    for surface_index, character in enumerate(surface):
        normalized = unicodedata.normalize("NFKC", character).lower().replace("ё", "е")
        for normalized_character in normalized:
            characters.append(normalized_character)
            indices.append(surface_index)
    return _MatchText("".join(characters), tuple(indices))


_CLAUSE_BREAK = re.compile(r"[.!?;:\n]+|,\s+(?=(?:а|но|однако)\b)", re.IGNORECASE)


def _iter_clauses(source: Literal["name", "description"], surface: SurfaceText) -> Iterator[_Clause]:
    boundaries = [0]
    boundaries.extend(match.end() for match in _CLAUSE_BREAK.finditer(surface.text))
    boundaries.append(len(surface.text))
    for left, right in pairwise(boundaries):
        while left < right and (surface.text[left].isspace() or surface.text[left] in ".!?;:,\n"):
            left += 1
        while right > left and (surface.text[right - 1].isspace() or surface.text[right - 1] in ".!?;:,\n"):
            right -= 1
        if left >= right:
            continue
        text = surface.text[left:right]
        yield _Clause(source, surface, left, right, text, _match_text(text).text)


def _focused_clause(clause: _Clause, match: re.Match[str]) -> _Clause:
    """Return the smallest exact surface slice that contains the matched construction."""

    normalized = _match_text(clause.text)
    local_start = normalized.surface_indices[match.start()]
    local_end = normalized.surface_indices[match.end() - 1] + 1
    while local_start > 0 and clause.text[local_start - 1].isalnum():
        local_start -= 1
    while local_end < len(clause.text) and clause.text[local_end].isalnum():
        local_end += 1
    while local_end - local_start < _MIN_QUOTE_LENGTH and (
        local_start > 0 or local_end < len(clause.text)
    ):
        if local_end < len(clause.text):
            while local_end < len(clause.text) and clause.text[local_end].isspace():
                local_end += 1
            while local_end < len(clause.text) and not clause.text[local_end].isspace():
                local_end += 1
        elif local_start > 0:
            while local_start > 0 and clause.text[local_start - 1].isspace():
                local_start -= 1
            while local_start > 0 and not clause.text[local_start - 1].isspace():
                local_start -= 1
    start = clause.start + local_start
    end = clause.start + local_end
    text = clause.surface.text[start:end]
    return _Clause(clause.source, clause.surface, start, end, text, _match_text(text).text)


_BAD_POSITIVE_PATTERNS = (
    re.compile(r"\bбад\b"),
    re.compile(r"\bбиологически\s+активн\w*\s+добавк\w*\b"),
    re.compile(r"\bбиодобавк\w*\b"),
)
_BAD_ENGLISH_PATTERN = re.compile(r"\b(?:dietary|food)\s+supplement\b|\bsupplement\s+facts\b")
_BAD_NEGATION_PATTERN = re.compile(
    r"\b(?:не\s+(?:является|относится\s+к)\s+(?:бад|биологически\s+активн\w*\s+добавк\w*)"
    r"|не\s+бад)\b"
)
_BAD_REFERENCE_PATTERN = re.compile(
    r"\b(?:(?:сырье|компонент|упаковка)\s+(?:для\s+)?(?:производств\w*\s+)?(?:бад|биологически\s+активн\w*\s+добавк\w*)"
    r"|для\s+производств\w*\s+(?:бад|биологически\s+активн\w*\s+добавк\w*)"
    r"|при\s+производств\w*\s+(?:бад|биологически\s+активн\w*\s+добавк\w*))\b"
)

_IGNITION_PATTERN = re.compile(r"\b(?:зажигалк\w*|спич(?:ки|ек|ечн\w*)|огнив\w*|факел\w*)\b")
_PYRO_PATTERN = re.compile(
    r"\b(?:фейерверк\w*|салют\w*|петард\w*|бенгальск\w*\s+огн\w*|дым\w*\s+шашк\w*)\b"
)
_COMBUSTIBLE_PATTERN = re.compile(
    r"\b(?:жидкост\w*\s+для\s+розжиг\w*|бензин\w*|керосин\w*|биоэтанол\w*|"
    r"топлив\w*|древесн\w*\s+угол\w*|угол\w*\s+древесн\w*|топливн\w*\s+брикет\w*|"
    r"дров\w*|парафин\w*)\b"
)
_QUALIFYING_OBJECT_PATTERN = re.compile(
    r"\b(?:газов\w*\s+баллон\w*|баллон\w*(?:\s+с\s+(?:бутан\w*|пропан\w*))?|"
    r"газ\w*|топлив\w*|бензин\w*|керосин\w*|биоэтанол\w*|спирт\w*|"
    r"зажигалк\w*|спич(?:ки|ек)|огнив\w*|фейерверк\w*|петард\w*)\b"
)
_EXCLUSION_PATTERN = re.compile(
    rf"(?:\bбез\s+(?:[^,;:.!?]\s*){{0,3}}{_QUALIFYING_OBJECT_PATTERN.pattern}"
    rf"|{_QUALIFYING_OBJECT_PATTERN.pattern}(?:\s+\w+){{0,5}}\s+не\s+(?:входит|включ\w*)\b"
    rf"|{_QUALIFYING_OBJECT_PATTERN.pattern}(?:\s+\w+){{0,5}}\s+приобретается\s+отдельно\b"
    rf"|\bне\s+содержит\s+(?:\w+\s+){{0,2}}{_QUALIFYING_OBJECT_PATTERN.pattern})"
)
_EMPTY_PATTERN = re.compile(
    r"\b(?:пуст\w*|незаправлен\w*|без\s+содержим\w*)\b(?:\s+\w+){0,4}\s+"
    r"(?:баллон\w*|емкост\w*|зажигалк\w*|горелк\w*)\b|"
    r"\b(?:баллон\w*|емкост\w*|зажигалк\w*|горелк\w*)\b(?:\s+\w+){0,4}\s+"
    r"(?:пуст\w*|незаправлен\w*|без\s+содержим\w*)\b"
)
_COMPATIBILITY_PATTERN = re.compile(
    rf"\b(?:подходит\s+для|совместим\w*\s+с|предназначен\w*\s+для|"
    rf"работает\s+(?:на|от)|заправляется)\b(?:\s+\w+){{0,8}}\s+{_QUALIFYING_OBJECT_PATTERN.pattern}"
)
_INTEGRATED_PATTERN = re.compile(
    r"\b(?:встроенн\w*|несъемн\w*|несъёмн\w*|компонент\w*)\b(?:\s+\w+){0,6}\s+"
    r"(?:пьезоподжиг\w*|поджиг\w*|зажигалк\w*)\b"
)
_INCLUSION_PATTERN = re.compile(
    rf"(?:\b(?:в\s+комплект(?:е)?\s+(?:входит|входят)|набор\s+(?:включает|содержит)|"
    rf"поставляется\s+с)\b(?:\s+\w+){{0,10}}\s+{_QUALIFYING_OBJECT_PATTERN.pattern}"
    rf"|{_QUALIFYING_OBJECT_PATTERN.pattern}(?:\s+\w+){{0,5}}\s+в\s+комплекте\b)"
)
_HAZARD_PATTERN = re.compile(r"\b(?:легковоспламеняющ\w*|огнеопасн\w*)\b")
_WRAPPER_PATTERN = re.compile(
    r"\b(?:чехол\w*|футляр\w*|держател\w*|запчаст\w*|защит\w*|макет\w*|наклейк\w*)\b"
)
_AMBIGUOUS_NEGATION_PATTERN = re.compile(r"\bне\s+только\b|\bне\s+не\b")


def _make_candidate(
    clause: _Clause,
    concept: str,
    *,
    polarity: Polarity,
    scope: Scope,
    directness: int = 0,
    specificity: int = 0,
) -> _Candidate:
    metadata = CONCEPT_VOCABULARY[concept]
    return _Candidate(
        clause=clause,
        concept=concept,
        supports_prediction=int(metadata["supports_prediction"]),
        polarity=polarity,
        scope=scope,
        directness=directness,
        specificity=specificity,
    )


def _bad_candidates(clauses: list[_Clause]) -> tuple[list[_Candidate], set[str]]:
    candidates: list[_Candidate] = []
    risks: set[str] = set()
    for clause in clauses:
        text = clause.match
        positive_matches = [
            match
            for pattern in _BAD_POSITIVE_PATTERNS
            if (match := pattern.search(text)) is not None
        ]
        english_match = _BAD_ENGLISH_PATTERN.search(text)
        relevant = bool(positive_matches or english_match)
        if relevant and _AMBIGUOUS_NEGATION_PATTERN.search(text):
            risks.add("ambiguous_negation")
            continue
        if negation_match := _BAD_NEGATION_PATTERN.search(text):
            candidates.append(
                _make_candidate(
                    _focused_clause(clause, negation_match),
                    "BAD_EXPLICIT_NEGATION",
                    polarity="negated",
                    scope="product",
                    specificity=3,
                )
            )
            continue
        has_positive = bool(positive_matches)
        has_english = english_match is not None
        if not (has_positive or has_english):
            continue
        if _BAD_REFERENCE_PATTERN.search(text):
            risks.add("ambiguous_scope")
            continue
        concept = "BAD_EXPLICIT_SUPPLEMENT_MARKING" if has_english else "BAD_EXPLICIT_MARKING"
        evidence_match = english_match if has_english else positive_matches[0]
        assert evidence_match is not None
        candidates.append(
            _make_candidate(
                _focused_clause(clause, evidence_match),
                concept,
                polarity="asserted",
                scope="product",
                specificity=3,
            )
        )
    return candidates, risks


def _flammable_candidates(clauses: list[_Clause]) -> tuple[list[_Candidate], set[str]]:
    candidates: list[_Candidate] = []
    risks: set[str] = set()
    for clause in clauses:
        text = clause.match
        relevant = any(
            pattern.search(text)
            for pattern in (
                _QUALIFYING_OBJECT_PATTERN,
                _IGNITION_PATTERN,
                _PYRO_PATTERN,
                _COMBUSTIBLE_PATTERN,
                _HAZARD_PATTERN,
            )
        )
        if relevant and _AMBIGUOUS_NEGATION_PATTERN.search(text):
            risks.add("ambiguous_negation")
            continue
        if exclusion_match := _EXCLUSION_PATTERN.search(text):
            candidates.append(
                _make_candidate(
                    _focused_clause(clause, exclusion_match),
                    "FL_FUEL_EXCLUDED",
                    polarity="negated",
                    scope="excluded",
                    specificity=3,
                )
            )
            continue
        if empty_match := _EMPTY_PATTERN.search(text):
            candidates.append(
                _make_candidate(
                    _focused_clause(clause, empty_match),
                    "FL_EMPTY_CONTAINER",
                    polarity="negated",
                    scope="empty",
                    specificity=3,
                )
            )
            continue
        if inclusion_match := _INCLUSION_PATTERN.search(text):
            candidates.append(
                _make_candidate(
                    _focused_clause(clause, inclusion_match),
                    "FL_INCLUDED_QUALIFYING_ITEM",
                    polarity="asserted",
                    scope="included",
                    specificity=3,
                )
            )
            continue
        if compatibility_match := _COMPATIBILITY_PATTERN.search(text):
            candidates.append(
                _make_candidate(
                    _focused_clause(clause, compatibility_match),
                    "FL_COMPATIBILITY_ONLY",
                    polarity="conditional",
                    scope="compatibility",
                    specificity=2,
                )
            )
            continue
        if integrated_match := _INTEGRATED_PATTERN.search(text):
            candidates.append(
                _make_candidate(
                    _focused_clause(clause, integrated_match),
                    "FL_INTEGRATED_IGNITION_ONLY",
                    polarity="asserted",
                    scope="integrated_component",
                    specificity=2,
                )
            )
            continue
        if hazard_match := _HAZARD_PATTERN.search(text):
            candidates.append(
                _make_candidate(
                    _focused_clause(clause, hazard_match),
                    "FL_EXPLICIT_HAZARD_MARKING",
                    polarity="asserted",
                    scope="product",
                    specificity=3,
                )
            )
            continue
        if _WRAPPER_PATTERN.search(text):
            if any(pattern.search(text) for pattern in (_IGNITION_PATTERN, _PYRO_PATTERN, _COMBUSTIBLE_PATTERN)):
                risks.add("ambiguous_scope")
            continue
        concept: str | None = None
        evidence_match: re.Match[str] | None = None
        if pyro_match := _PYRO_PATTERN.search(text):
            concept = "FL_PYROTECHNIC_PRODUCT"
            evidence_match = pyro_match
        elif ignition_match := _IGNITION_PATTERN.search(text):
            concept = "FL_STANDALONE_IGNITION_SOURCE"
            evidence_match = ignition_match
        elif combustible_match := _COMBUSTIBLE_PATTERN.search(text):
            concept = "FL_COMBUSTIBLE_PRODUCT"
            evidence_match = combustible_match
        if concept is not None:
            assert evidence_match is not None
            candidates.append(
                _make_candidate(
                    _focused_clause(clause, evidence_match),
                    concept,
                    polarity="asserted",
                    scope="product",
                    specificity=3,
                )
            )
    return candidates, risks


_TEMPLATES: Mapping[str, str] = MappingProxyType(
    {
        "BAD_EXPLICIT_MARKING": (
            "В {source_label} прямо указано «{span}»: это явная маркировка "
            "биологически активной добавки."
        ),
        "BAD_EXPLICIT_SUPPLEMENT_MARKING": (
            "В {source_label} прямо указано «{span}»: это явная маркировка пищевой добавки."
        ),
        "BAD_EXPLICIT_NEGATION": (
            "В {source_label} прямо сказано «{span}»: принадлежность товара к БАД явно отрицается."
        ),
        "BAD_TEXT_MARKING_NOT_FOUND": (
            "В доступных названии и описании нет прямой маркировки БАД; товар назван «{span}»."
        ),
        "FL_STANDALONE_IGNITION_SOURCE": (
            "В {source_label} товар указан как «{span}»: это самостоятельный источник поджига."
        ),
        "FL_PYROTECHNIC_PRODUCT": (
            "В {source_label} товар указан как «{span}»: это явно продаваемое пиротехническое изделие."
        ),
        "FL_COMBUSTIBLE_PRODUCT": (
            "В {source_label} товар указан как «{span}»: горючее вещество является предметом продажи."
        ),
        "FL_INCLUDED_QUALIFYING_ITEM": (
            "В {source_label} сказано «{span}»: квалифицирующий товар явно входит в продаваемый комплект."
        ),
        "FL_EXPLICIT_HAZARD_MARKING": (
            "В {source_label} прямо указано «{span}»: маркировка сообщает об огнеопасности товара."
        ),
        "FL_FUEL_EXCLUDED": (
            "В {source_label} указано «{span}»: баллон, газ или топливо в текущую продажу не включены."
        ),
        "FL_EMPTY_CONTAINER": (
            "В {source_label} указано «{span}»: продаваемая ёмкость или устройство описаны как пустые."
        ),
        "FL_COMPATIBILITY_ONLY": (
            "В {source_label} сказано «{span}»: указана совместимость, а не наличие топлива в продаже."
        ),
        "FL_INTEGRATED_IGNITION_ONLY": (
            "В {source_label} сказано «{span}»: поджиг описан как встроенная часть, а не отдельный товар."
        ),
        "FL_QUALIFYING_ITEM_NOT_FOUND": (
            "В доступных названии и описании нет прямого указания на квалифицирующий товар; товар назван «{span}»."
        ),
    }
)


def _safe_quote(candidate: _Candidate) -> bool:
    span = candidate.clause.text
    if not (_MIN_QUOTE_LENGTH <= len(span) <= _MAX_QUOTE_LENGTH):
        return False
    if "<" in span or ">" in span or "\n" in span or "\r" in span:
        return False
    if any(unicodedata.category(character) == "Cc" for character in span):
        return False
    return span.count("\"") % 2 == 0 and span.count("«") == span.count("»")


def _candidate_sort_key(candidate: _Candidate) -> tuple[int, int, int, int, int, str]:
    source_rank = 0 if candidate.clause.source == "name" else 1
    tie_material = (
        f"{candidate.clause.source}\0{candidate.clause.start}\0"
        f"{candidate.clause.end}\0{candidate.concept}"
    ).encode()
    tie_hash = hashlib.sha256(tie_material).hexdigest()
    return (
        candidate.directness,
        -candidate.specificity,
        source_rank,
        len(candidate.clause.text),
        candidate.clause.start,
        tie_hash,
    )


def _render_comment(concept: str, source: Source, span: str) -> str:
    source_label = "названии" if source == "name" else "описании"
    return _TEMPLATES[concept].format(source_label=source_label, span=span)


def _comment_is_valid(comment: str) -> bool:
    lowered = comment.lower()
    return (
        _MIN_COMMENT_LENGTH <= len(comment) <= _MAX_COMMENT_LENGTH
        and "<" not in comment
        and ">" not in comment
        and "\n" not in comment
        and "\r" not in comment
        and not any(word in lowered for word in _FORBIDDEN_COMMENT_WORDS)
        and not any(unicodedata.category(character) == "Cc" for character in comment)
    )


def _bounded_absence_candidate(
    category: str,
    name_surface: SurfaceText,
) -> _Candidate | None:
    clauses = list(_iter_clauses("name", name_surface))
    if not clauses:
        return None
    clause = min(clauses, key=lambda item: (len(item.text), item.start))
    concept = (
        "BAD_TEXT_MARKING_NOT_FOUND"
        if category == "БАД"
        else "FL_QUALIFYING_ITEM_NOT_FOUND"
    )
    return _make_candidate(
        clause,
        concept,
        polarity="bounded_absence",
        scope="product",
        directness=1,
        specificity=0,
    )


def _no_safe_result(
    *,
    row_id: str,
    category: str,
    frozen_prediction: int,
    reason: str,
) -> EvidenceResult:
    verdict = "не бан" if frozen_prediction == 1 else "бан"
    return EvidenceResult(
        status="NO_SAFE_EVIDENCE",
        row_id=row_id,
        category=category,
        frozen_prediction=frozen_prediction,
        verdict=verdict,
        source="none",
        raw_start=None,
        raw_end=None,
        surface_start=None,
        surface_end=None,
        exact_surface_span=None,
        concept=None,
        polarity="ambiguous",
        scope="ambiguous",
        comment=_FALLBACK_COMMENT,
        fallback_reason=reason,
        checks=EvidenceChecks(
            exact_span=True,
            verdict_locked=True,
            length_ok=_MIN_COMMENT_LENGTH <= len(_FALLBACK_COMMENT) <= _MAX_COMMENT_LENGTH,
            no_forbidden_tag=True,
            no_conflict=reason != "conflicting_evidence",
        ),
    )


def extract_evidence(
    *,
    row_id: str,
    category: str,
    name: str,
    description: str,
    frozen_prediction: int,
    vocabulary_version: str = VOCABULARY_VERSION,
) -> EvidenceResult:
    """Find evidence for an immutable verdict without reading a gold label."""

    if vocabulary_version != VOCABULARY_VERSION:
        raise ValueError(f"Unsupported vocabulary version: {vocabulary_version!r}")
    if category not in _CATEGORIES:
        raise ValueError(f"Unsupported category: {category!r}")
    if isinstance(frozen_prediction, bool) or frozen_prediction not in (0, 1):
        raise ValueError("frozen_prediction must be integer 0 or 1")

    row_id = str(row_id)
    name_surface = surface_text(name)
    description_surface = surface_text(description)
    clauses = [
        *list(_iter_clauses("name", name_surface)),
        *list(_iter_clauses("description", description_surface)),
    ]
    if category == "БАД":
        candidates, risks = _bad_candidates(clauses)
    else:
        candidates, risks = _flammable_candidates(clauses)

    supportive = [item for item in candidates if item.supports_prediction == frozen_prediction]
    opposite = [item for item in candidates if item.supports_prediction != frozen_prediction]
    if supportive and opposite:
        return _no_safe_result(
            row_id=row_id,
            category=category,
            frozen_prediction=frozen_prediction,
            reason="conflicting_evidence",
        )
    if "ambiguous_negation" in risks:
        return _no_safe_result(
            row_id=row_id,
            category=category,
            frozen_prediction=frozen_prediction,
            reason="ambiguous_negation",
        )
    if supportive:
        pool = supportive
    elif opposite:
        return _no_safe_result(
            row_id=row_id,
            category=category,
            frozen_prediction=frozen_prediction,
            reason="opposite_only",
        )
    elif "ambiguous_scope" in risks:
        return _no_safe_result(
            row_id=row_id,
            category=category,
            frozen_prediction=frozen_prediction,
            reason="ambiguous_scope",
        )
    elif frozen_prediction == 0:
        bounded = _bounded_absence_candidate(category, name_surface)
        pool = [] if bounded is None else [bounded]
    else:
        pool = []

    saw_unsafe_quote = False
    saw_bad_length = False
    for candidate in sorted(pool, key=_candidate_sort_key):
        if not _safe_quote(candidate):
            saw_unsafe_quote = True
            continue
        span = candidate.clause.text
        comment = _render_comment(candidate.concept, candidate.clause.source, span)
        if not _comment_is_valid(comment):
            saw_bad_length = True
            continue
        raw_start, raw_end = candidate.clause.surface.raw_interval(
            candidate.clause.start, candidate.clause.end
        )
        verdict = "не бан" if frozen_prediction == 1 else "бан"
        exact_span = candidate.clause.surface.text[
            candidate.clause.start : candidate.clause.end
        ]
        return EvidenceResult(
            status="SAFE",
            row_id=row_id,
            category=category,
            frozen_prediction=frozen_prediction,
            verdict=verdict,
            source=candidate.clause.source,
            raw_start=raw_start,
            raw_end=raw_end,
            surface_start=candidate.clause.start,
            surface_end=candidate.clause.end,
            exact_surface_span=exact_span,
            concept=candidate.concept,
            polarity=candidate.polarity,
            scope=candidate.scope,
            comment=comment,
            fallback_reason=None,
            checks=EvidenceChecks(
                exact_span=exact_span == span,
                verdict_locked=True,
                length_ok=True,
                no_forbidden_tag=True,
                no_conflict=True,
            ),
        )

    if saw_bad_length:
        reason = "no_template_fits_length"
    elif saw_unsafe_quote:
        reason = "unsafe_quote"
    else:
        reason = "no_anchor"
    return _no_safe_result(
        row_id=row_id,
        category=category,
        frozen_prediction=frozen_prediction,
        reason=reason,
    )


def render_submission(result: EvidenceResult) -> str:
    """Render the immutable competition fields from a validated result."""

    expected_verdict = "не бан" if result.frozen_prediction == 1 else "бан"
    if result.verdict != expected_verdict or not result.checks.verdict_locked:
        raise ValueError("Evidence result violates verdict lock")
    if not _comment_is_valid(result.comment):
        raise ValueError("Evidence result contains an invalid comment")
    return f"<комментарий>{result.comment}<вердикт>{expected_verdict}"

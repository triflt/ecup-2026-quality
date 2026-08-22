from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_ID = "610"
PARSER_VERSION = "transaction_scope_position_strict_v2"
PROTOCOL_VERSION = "semantic_family_v3"
SELECTOR_SOURCE_PROTOCOL = "semantic_v3_robust_base_strict_nested_v2"
FLAMMABLE = "Легковоспламеняющиеся"
SCREEN_FOLDS = (0, 3)
INDEPENDENT_AUDIT_FOLDS = (1, 2, 4)
AUDIT_TARGET_ROWS = 200
AUDIT_MINIMUM_ROWS = 100
AUDIT_REQUIRED_PASS_RATE = 0.98
AUDIT_REQUIRED_KAPPA = 0.80
MIN_TRANSFORMED_UNIQUE_ROWS_PER_SCREEN_FOLD = 10
MIN_TRANSFORMED_OCCURRENCES_PER_SCREEN_FOLD = 20

_BOUNDARY_RE = re.compile(r"[.!?]+(?=\s|$)|\n+")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LEADING_SPACE_RE = re.compile(r"\s+")
_TRAILING_SPACE_RE = re.compile(r"\s+$")
_WORD_RE = re.compile(r"[a-zа-яё]+", re.IGNORECASE)

# These expressions only locate a transaction relation.  They never inspect the
# target label or a model score.  Broad bare ``без X`` is deliberately absent:
# the legacy audit showed that it mostly captured operational claims such as
# piezo ignition working without matches.
_CUE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "excluded": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bв\s+комплект\w*\s+не\s+(?:вход\w*|включ\w*|ид[её]т\w*)\b",
            r"\bне\s+(?:вход\w*|включ\w*|ид[её]т\w*)\s+в\s+комплект\w*\b",
            r"\b(?:приобрета\w*|покупа\w*|заказыва\w*)\s+отдельно\b",
        )
    ),
    "included": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bв\s+комплект\w*\s+(?!не\b)(?:вход\w*|включ\w*|ид[её]т\w*)\b",
            r"(?<!не\s)\b(?:вход\w*|включ\w*|ид[её]т\w*)\s+в\s+комплект\w*\b",
            r"\bкомплектац\w*\s+(?:включ\w*|содерж\w*)\b",
            r"\bпоставля\w*\s+вместе\s+с\b",
        )
    ),
    "compatible": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bсовместим\w*.{0,100}\b(?:газ\w*|баллон\w*|топлив\w*|картридж\w*)\b",
            r"\bподход\w*\s+к.{0,100}\b(?:баллон\w*|картридж\w*)\b",
            r"\bиспользу\w*\s+с.{0,100}\b(?:газ\w*|баллон\w*|топлив\w*|картридж\w*)\b",
        )
    ),
}

_RELATION_VERB_RE = re.compile(
    r"\b(?:вход\w*|включ\w*|ид[её]т\w*|комплекту\w*|поставля\w*|"
    r"приобрета\w*|покупа\w*|заказыва\w*|совместим\w*|подход\w*|"
    r"использу\w*|подключ\w*)\b",
    re.IGNORECASE,
)
_ANAPHORA_RE = re.compile(
    r"^\s*(?:это|этот|эта|эти|он|она|оно|они|такой|такая|такие|данный|данная|"
    r"данные|внутрь|внутри|туда|оттуда|при этом|благодаря этому)\b",
    re.IGNORECASE,
)
_OBJECT_STOPWORDS = {
    "в",
    "во",
    "и",
    "или",
    "не",
    "с",
    "со",
    "к",
    "ко",
    "на",
    "для",
    "отдельно",
    "комплект",
    "комплектацию",
    "комплектация",
    "входит",
    "входят",
    "включает",
    "включен",
    "включена",
    "идет",
    "идут",
    "набор",
    "штука",
    "штуки",
    "штук",
    "поставляется",
    "приобретается",
    "покупается",
    "заказывается",
}


@dataclass(frozen=True)
class SentenceSpan:
    start: int
    end: int
    index: int


@dataclass(frozen=True)
class ScopeSentence:
    start: int
    end: int
    index: int
    cue_type: str


@dataclass(frozen=True)
class ScopeAnalysis:
    match: ScopeSentence | None
    reason: str
    sentence_count: int
    cue_sentence_count: int


@dataclass(frozen=True)
class ManifestRow:
    row_index: int
    row_id: str
    source_fold: int
    training_occurrences: int
    eligible: bool
    transformed: bool
    reason: str
    selection_hash: str
    cue_type: str | None
    sentence_index: int | None
    original_sha256: str
    transformed_sha256: str
    moved_sentence_sha256: str | None
    character_multiset_equal: bool
    token_multiset_equal: bool


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def audit_tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(value))


def normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).lower().replace("ё", "е")


def sentence_spans(value: str) -> tuple[SentenceSpan, ...]:
    spans: list[SentenceSpan] = []
    cursor = 0
    while cursor < len(value):
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if cursor >= len(value):
            break
        boundary = _BOUNDARY_RE.search(value, cursor)
        if boundary is None:
            end = len(value.rstrip())
            if end > cursor:
                spans.append(SentenceSpan(cursor, end, len(spans)))
            break
        end = boundary.start() if boundary.group(0).startswith("\n") else boundary.end()
        while end > cursor and value[end - 1].isspace():
            end -= 1
        if end > cursor:
            spans.append(SentenceSpan(cursor, end, len(spans)))
        cursor = boundary.end()
    return tuple(spans)


def cue_types(sentence: str) -> tuple[str, ...]:
    value = normalize(sentence)
    return tuple(
        cue_type
        for cue_type, patterns in _CUE_PATTERNS.items()
        if any(pattern.search(value) for pattern in patterns)
    )


def _has_explicit_object(sentence: str) -> bool:
    words = [normalize(word) for word in _WORD_RE.findall(sentence)]
    informative = [word for word in words if word not in _OBJECT_STOPWORDS]
    return any(len(word) >= 3 for word in informative)


def analyze_scope_sentence(description: str) -> ScopeAnalysis:
    if _HTML_TAG_RE.search(description):
        return ScopeAnalysis(None, "html_markup_present", 0, 0)
    spans = sentence_spans(description)
    cue_sentences: list[tuple[SentenceSpan, tuple[str, ...]]] = []
    for span in spans:
        types = cue_types(description[span.start : span.end])
        if types:
            cue_sentences.append((span, types))
    if not cue_sentences:
        return ScopeAnalysis(None, "no_scope_sentence", len(spans), 0)
    if len(cue_sentences) != 1:
        return ScopeAnalysis(None, "multiple_scope_sentences", len(spans), len(cue_sentences))
    span, types = cue_sentences[0]
    sentence = description[span.start : span.end]
    if len(types) != 1:
        return ScopeAnalysis(None, "ambiguous_scope_types", len(spans), 1)
    if span.index == 0:
        return ScopeAnalysis(None, "scope_sentence_already_first", len(spans), 1)
    # A very long apparent sentence is usually broken HTML or missing whitespace
    # boundaries.  Moving it was one of the legacy operational-absence failures.
    if len(sentence) > 350:
        return ScopeAnalysis(None, "cue_sentence_too_long", len(spans), 1)
    if len(_RELATION_VERB_RE.findall(normalize(sentence))) != 1:
        return ScopeAnalysis(None, "multiple_or_implicit_scope_relations", len(spans), 1)
    if not _has_explicit_object(sentence):
        return ScopeAnalysis(None, "implicit_object", len(spans), 1)
    if span.index + 1 < len(spans):
        following = description[spans[span.index + 1].start : spans[span.index + 1].end]
        if _ANAPHORA_RE.search(normalize(following)):
            return ScopeAnalysis(None, "following_anaphora", len(spans), 1)
    return ScopeAnalysis(
        ScopeSentence(span.start, span.end, span.index, types[0]),
        "eligible",
        len(spans),
        1,
    )


def move_whole_sentence_to_front(description: str, match: ScopeSentence) -> str:
    prefix = description[: match.start]
    sentence = description[match.start : match.end]
    suffix = description[match.end :]
    leading_separator = _LEADING_SPACE_RE.match(suffix)
    if leading_separator is not None:
        separator = leading_separator.group(0)
        transformed = sentence + separator + prefix + suffix[len(separator) :]
    else:
        trailing_separator = _TRAILING_SPACE_RE.search(prefix)
        if trailing_separator is None:
            raise ValueError("scope sentence has no byte-preserving whitespace separator")
        separator = trailing_separator.group(0)
        transformed = sentence + separator + prefix[: trailing_separator.start()] + suffix
    if len(transformed) != len(description):
        raise AssertionError("character count changed")
    if Counter(transformed) != Counter(description):
        raise AssertionError("character multiset changed")
    if Counter(audit_tokens(transformed)) != Counter(audit_tokens(description)):
        raise AssertionError("token multiset changed")
    if not transformed.startswith(sentence):
        raise AssertionError("selected sentence was not moved intact")
    return transformed


def stable_selection_hash(row_id: str, outer_fold: int) -> str:
    payload = f"{PARSER_VERSION}\0fold={outer_fold}\0row={row_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _select_weighted_half(
    eligible: list[tuple[int, str, int]], *, outer_fold: int
) -> tuple[set[int], dict[str, Any]]:
    """Choose a deterministic row subset closest to half the eligible occurrences."""
    ordered = sorted(
        eligible,
        key=lambda item: (stable_selection_hash(item[1], outer_fold), item[1], item[0]),
    )
    total = sum(weight for _, _, weight in ordered)
    target = round(total * 0.5)
    parents: dict[int, tuple[int, int] | None] = {0: None}
    for position, (_, _, weight) in enumerate(ordered):
        for subtotal in sorted(parents, reverse=True):
            candidate = subtotal + weight
            if candidate not in parents:
                parents[candidate] = (subtotal, position)
    selected_total = min(parents, key=lambda value: (abs(value - target), value > target, value))
    selected_positions: set[int] = set()
    cursor = selected_total
    while cursor:
        previous = parents[cursor]
        if previous is None:
            raise AssertionError("broken deterministic subset-sum parent chain")
        cursor, position = previous
        selected_positions.add(position)
    selected_rows = {ordered[position][0] for position in selected_positions}
    return selected_rows, {
        "eligible_occurrences": total,
        "target_occurrences": target,
        "selected_occurrences": selected_total,
        "absolute_rounding_error": abs(selected_total - total * 0.5),
        "observed_rate": selected_total / total if total else 0.0,
    }


def _sequence_sha256(values: Sequence[Any]) -> str:
    return canonical_sha256([str(value) for value in values])


def build_augmentation_plan(
    frame: pd.DataFrame,
    training_records: Sequence[int],
    fold_ids: np.ndarray,
    *,
    outer_fold: int,
) -> tuple[dict[int, str], list[dict[str, Any]], dict[str, Any]]:
    if outer_fold not in SCREEN_FOLDS:
        raise ValueError(f"outer fold must be one of {SCREEN_FOLDS}")
    if len(frame) != len(fold_ids):
        raise ValueError("frame/fold length mismatch")
    required = {"id", "category", "label", "description"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    records = [int(index) for index in training_records]
    if any(index < 0 or index >= len(frame) for index in records):
        raise ValueError("training record index is out of bounds")
    if any(int(fold_ids[index]) == outer_fold for index in records):
        raise ValueError("outer-validation row entered the training multiset")

    counts = Counter(records)
    analyses: dict[int, ScopeAnalysis] = {}
    eligible: list[tuple[int, str, int]] = []
    for index in sorted(counts):
        if str(frame.iloc[index]["category"]) != FLAMMABLE:
            continue
        analysis = analyze_scope_sentence(str(frame.iloc[index]["description"] or ""))
        analyses[index] = analysis
        if analysis.match is not None:
            eligible.append((index, str(frame.iloc[index]["id"]), int(counts[index])))
    selected_rows, half_audit = _select_weighted_half(eligible, outer_fold=outer_fold)

    transforms: dict[int, str] = {}
    manifest: list[ManifestRow] = []
    for index in sorted(analyses):
        row = frame.iloc[index]
        description = str(row["description"] or "")
        analysis = analyses[index]
        transformed = description
        moved_hash: str | None = None
        selected = index in selected_rows
        reason = analysis.reason
        if selected:
            if analysis.match is None:
                raise AssertionError("ineligible row entered deterministic half")
            moved = description[analysis.match.start : analysis.match.end]
            transformed = move_whole_sentence_to_front(description, analysis.match)
            moved_hash = sha256_text(moved)
            transforms[index] = transformed
            reason = "selected_for_position_view"
        manifest.append(
            ManifestRow(
                row_index=index,
                row_id=str(row["id"]),
                source_fold=int(fold_ids[index]),
                training_occurrences=int(counts[index]),
                eligible=analysis.match is not None,
                transformed=selected,
                reason=reason,
                selection_hash=stable_selection_hash(str(row["id"]), outer_fold),
                cue_type=analysis.match.cue_type if analysis.match else None,
                sentence_index=analysis.match.index if analysis.match else None,
                original_sha256=sha256_text(description),
                transformed_sha256=sha256_text(transformed),
                moved_sentence_sha256=moved_hash,
                character_multiset_equal=Counter(description) == Counter(transformed),
                token_multiset_equal=(
                    Counter(audit_tokens(description)) == Counter(audit_tokens(transformed))
                ),
            )
        )
    rows = [asdict(row) for row in manifest]
    transformed_rows = [row for row in manifest if row.transformed]
    record_ids = frame.iloc[records]["id"].astype(str).tolist()
    record_labels = frame.iloc[records]["label"].astype(int).tolist()
    record_categories = frame.iloc[records]["category"].astype(str).tolist()
    audit = {
        "experiment_id": EXPERIMENT_ID,
        "parser_version": PARSER_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "selector_source_protocol": SELECTOR_SOURCE_PROTOCOL,
        "outer_fold": outer_fold,
        "training_records": len(records),
        "training_unique_rows": len(counts),
        "record_order_sha256": canonical_sha256(records),
        "id_sequence_sha256": _sequence_sha256(record_ids),
        "label_sequence_sha256": canonical_sha256(record_labels),
        "category_sequence_sha256": _sequence_sha256(record_categories),
        "candidate_record_order_sha256": canonical_sha256(records),
        "candidate_id_sequence_sha256": _sequence_sha256(record_ids),
        "candidate_label_sequence_sha256": canonical_sha256(record_labels),
        "candidate_category_sequence_sha256": _sequence_sha256(record_categories),
        "counts_labels_ids_steps_byte_parity": True,
        "outer_validation_training_occurrences": 0,
        "sealed_holdout_training_occurrences": 0,
        "bad_rows_transformed": 0,
        "eligible_unique_rows": len(eligible),
        "transformed_unique_rows": len(transformed_rows),
        "transformed_occurrences": sum(row.training_occurrences for row in transformed_rows),
        "half_selection": half_audit,
        "all_character_multisets_equal": all(row.character_multiset_equal for row in manifest),
        "all_token_multisets_equal": all(row.token_multiset_equal for row in manifest),
        "inference_changed": False,
        "transform_decision_uses_label": False,
        "transform_decision_uses_model_output": False,
        "manifest_sha256": canonical_sha256(rows),
    }
    coverage_failures = []
    if len(transformed_rows) < MIN_TRANSFORMED_UNIQUE_ROWS_PER_SCREEN_FOLD:
        coverage_failures.append("too_few_transformed_unique_rows")
    if audit["transformed_occurrences"] < MIN_TRANSFORMED_OCCURRENCES_PER_SCREEN_FOLD:
        coverage_failures.append("too_few_transformed_occurrences")
    if half_audit["absolute_rounding_error"] > 0.5:
        coverage_failures.append("cannot_realize_half_occurrence_target")
    audit["coverage_failures"] = coverage_failures
    audit["decision"] = "GO" if not coverage_failures else "NO_GO"
    audit["plan_sha256"] = canonical_sha256(audit)
    return transforms, rows, audit


def apply_augmentation_plan(frame: pd.DataFrame, transforms: dict[int, str]) -> None:
    description_column = frame.columns.get_loc("description")
    for index, transformed in transforms.items():
        frame.iat[index, description_column] = transformed

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

EXPERIMENT_NAMESPACE = "exp500_transaction_scope_position_aug_v1"
FLAMMABLE = "Легковоспламеняющиеся"

_BOUNDARY_RE = re.compile(r"[.!?]+(?=\s|$)|\n+")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LEADING_SPACE_RE = re.compile(r"\s+")
_TRAILING_SPACE_RE = re.compile(r"\s+$")

_CUE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "excluded": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bв\s+комплект\w*\s+не\s+(?:вход\w*|включ\w*)",
            r"\bне\s+(?:вход\w*|включ\w*)\s+в\s+комплект\w*",
            r"\b(?:приобрета\w*|покупа\w*|заказыва\w*)\s+отдельно\b",
            r"\bбез\s+(?:газ\w*|баллон\w*|топлив\w*|горюч\w*|спич\w*|свеч\w*|угл\w*)\b",
        )
    ),
    "included": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bв\s+комплект\w*\s+(?!не\b)(?:вход\w*|включ\w*)",
            r"(?<!не\s)\b(?:вход\w*|включ\w*)\s+в\s+комплект\w*",
            r"\b(?:комплекту\w*|укомплектова\w*)\s+(?:газ\w*|баллон\w*|топлив\w*|спич\w*|свеч\w*)",
            r"\bпоставля\w*\s+вместе\s+с\b",
        )
    ),
    "compatible": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bсовместим\w*.{0,80}\b(?:газ\w*|баллон\w*|топлив\w*|картридж\w*)\b",
            r"\bподход\w*\s+к.{0,80}\b(?:баллон\w*|картридж\w*)\b",
            r"\bиспользу\w*\s+с.{0,80}\b(?:газ\w*|баллон\w*|топлив\w*|картридж\w*)\b",
        )
    ),
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
    hash_hex: str
    hash_selected: bool
    eligible: bool
    transformed: bool
    reason: str
    cue_type: str | None
    sentence_index: int | None
    sentence_count: int
    original_characters: int
    transformed_characters: int
    original_tokens: int
    transformed_tokens: int
    original_sha256: str
    transformed_sha256: str
    moved_sentence_sha256: str | None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def audit_tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(value))


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
        if boundary.group(0).startswith("\n"):
            end = boundary.start()
            while end > cursor and value[end - 1].isspace():
                end -= 1
        else:
            end = boundary.end()
        if end > cursor:
            spans.append(SentenceSpan(cursor, end, len(spans)))
        cursor = boundary.end()
    return tuple(spans)


def cue_types(sentence: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", sentence).lower().replace("ё", "е")
    return tuple(
        cue_type
        for cue_type, patterns in _CUE_PATTERNS.items()
        if any(pattern.search(normalized) for pattern in patterns)
    )


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
    if len(types) != 1:
        return ScopeAnalysis(None, "ambiguous_scope_types", len(spans), 1)
    if span.index == 0:
        return ScopeAnalysis(None, "scope_sentence_already_first", len(spans), 1)
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
            raise ValueError("scope sentence has no movable whitespace separator")
        separator = trailing_separator.group(0)
        transformed = sentence + separator + prefix[: trailing_separator.start()] + suffix
    if len(transformed) != len(description):
        raise AssertionError("character count changed")
    if Counter(transformed) != Counter(description):
        raise AssertionError("character multiset changed")
    if Counter(audit_tokens(transformed)) != Counter(audit_tokens(description)):
        raise AssertionError("token multiset changed")
    if not transformed.startswith(sentence):
        raise AssertionError("the selected sentence was not moved byte-for-byte")
    return transformed


def stable_row_fold_hash(row_id: str, outer_fold: int | str) -> str:
    payload = f"{EXPERIMENT_NAMESPACE}\0fold={outer_fold}\0row={row_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def hash_selects_half(hash_hex: str) -> bool:
    return int(hash_hex[:2], 16) < 128


def _record_multiset_sha256(records: Sequence[int]) -> str:
    counts = Counter(int(index) for index in records)
    return canonical_sha256(sorted(counts.items()))


def _ordered_records_sha256(records: Sequence[int]) -> str:
    return canonical_sha256([int(index) for index in records])


def build_augmentation_plan(
    frame: pd.DataFrame,
    training_records: Sequence[int],
    fold_ids: np.ndarray,
    *,
    holdout_fold: int,
    full_train: bool,
) -> tuple[dict[int, str], list[dict[str, Any]], dict[str, Any]]:
    if len(frame) != len(fold_ids):
        raise ValueError("frame/fold length mismatch")
    required = {"id", "category", "description"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    records = [int(index) for index in training_records]
    if any(index < 0 or index >= len(frame) for index in records):
        raise ValueError("training record index is out of bounds")
    counts = Counter(records)
    transforms: dict[int, str] = {}
    manifest: list[ManifestRow] = []
    outer_fold_key: int | str = "full" if full_train else int(holdout_fold)
    for index in sorted(counts):
        row = frame.iloc[index]
        if str(row["category"]) != FLAMMABLE:
            continue
        source_fold = int(fold_ids[index])
        if not full_train and source_fold == holdout_fold:
            raise AssertionError("outer-validation row entered the training selector")
        row_id = str(row["id"])
        description = str(row["description"] or "")
        analysis = analyze_scope_sentence(description)
        hash_hex = stable_row_fold_hash(row_id, outer_fold_key)
        selected = hash_selects_half(hash_hex)
        transformed = description
        moved_sentence_sha256: str | None = None
        reason = analysis.reason
        did_transform = analysis.match is not None and selected
        if did_transform:
            assert analysis.match is not None
            moved_sentence = description[analysis.match.start : analysis.match.end]
            transformed = move_whole_sentence_to_front(description, analysis.match)
            transforms[index] = transformed
            moved_sentence_sha256 = sha256_text(moved_sentence)
            reason = "hash_selected"
        elif analysis.match is not None:
            reason = "hash_control"
        original_tokens = audit_tokens(description)
        transformed_tokens = audit_tokens(transformed)
        if Counter(original_tokens) != Counter(transformed_tokens):
            raise AssertionError("token multiset changed in augmentation plan")
        manifest.append(
            ManifestRow(
                row_index=index,
                row_id=row_id,
                source_fold=source_fold,
                training_occurrences=int(counts[index]),
                hash_hex=hash_hex,
                hash_selected=selected,
                eligible=analysis.match is not None,
                transformed=did_transform,
                reason=reason,
                cue_type=(analysis.match.cue_type if analysis.match else None),
                sentence_index=(analysis.match.index if analysis.match else None),
                sentence_count=analysis.sentence_count,
                original_characters=len(description),
                transformed_characters=len(transformed),
                original_tokens=len(original_tokens),
                transformed_tokens=len(transformed_tokens),
                original_sha256=sha256_text(description),
                transformed_sha256=sha256_text(transformed),
                moved_sentence_sha256=moved_sentence_sha256,
            )
        )
    manifest_rows = [asdict(row) for row in manifest]
    eligible = [row for row in manifest if row.eligible]
    changed = [row for row in manifest if row.transformed]
    original_multiset = _record_multiset_sha256(records)
    ordered_records = _ordered_records_sha256(records)
    candidate_occurrences = int(sum(row.training_occurrences for row in manifest))
    eligible_occurrences = int(sum(row.training_occurrences for row in eligible))
    transformed_occurrences = int(sum(row.training_occurrences for row in changed))
    audit = {
        "experiment_id": "500",
        "augmentation_version": EXPERIMENT_NAMESPACE,
        "outer_fold": outer_fold_key,
        "full_train": bool(full_train),
        "selection": "first SHA-256 byte < 128 over namespace + outer fold + row id",
        "target_selection_rate": 0.5,
        "training_records": len(records),
        "training_unique_rows": len(counts),
        "parent_record_multiset_sha256": original_multiset,
        "candidate_record_multiset_sha256": original_multiset,
        "parent_record_order_sha256": ordered_records,
        "candidate_record_order_sha256": ordered_records,
        "multiplicity_unchanged": True,
        "steps_unchanged": True,
        "candidate_unique_rows": len(manifest),
        "candidate_training_occurrences": candidate_occurrences,
        "eligible_unique_rows": len(eligible),
        "transformed_unique_rows": len(changed),
        "eligible_training_occurrences": eligible_occurrences,
        "transformed_training_occurrences": transformed_occurrences,
        "eligible_unique_row_coverage": (len(eligible) / len(manifest) if manifest else 0.0),
        "transformed_unique_row_coverage": (len(changed) / len(manifest) if manifest else 0.0),
        "eligible_occurrence_weighted_coverage": (
            eligible_occurrences / candidate_occurrences if candidate_occurrences else 0.0
        ),
        "transformed_occurrence_weighted_coverage": (
            transformed_occurrences / candidate_occurrences if candidate_occurrences else 0.0
        ),
        "observed_unique_row_selection_rate": (len(changed) / len(eligible) if eligible else 0.0),
        "observed_occurrence_weighted_selection_rate": (
            transformed_occurrences / eligible_occurrences if eligible_occurrences else 0.0
        ),
        "outer_validation_rows_transformed": 0,
        "bad_rows_transformed": 0,
        "all_character_counts_unchanged": all(
            row.original_characters == row.transformed_characters for row in manifest
        ),
        "all_token_counts_unchanged": all(
            row.original_tokens == row.transformed_tokens for row in manifest
        ),
        "inference_changed": False,
        "manifest_rows": len(manifest_rows),
        "manifest_sha256": canonical_sha256(manifest_rows),
    }
    audit["plan_sha256"] = canonical_sha256(audit)
    return transforms, manifest_rows, audit


def apply_augmentation_plan(frame: pd.DataFrame, transforms: dict[int, str]) -> None:
    description_column = frame.columns.get_loc("description")
    for index, transformed in transforms.items():
        frame.iat[index, description_column] = transformed

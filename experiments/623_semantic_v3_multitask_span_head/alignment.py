from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from protocol import CONCEPTS, canonical_text


def _locate_unique_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence:
        raise ValueError("parent prompt token sequence is empty")
    hits = [
        index
        for index in range(len(sequence) - len(subsequence) + 1)
        if sequence[index : index + len(subsequence)] == subsequence
    ]
    if len(hits) != 1:
        raise ValueError(f"parent prompt must occur exactly once in model input; found={len(hits)}")
    return hits[0]


def _character_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    if not 0 <= char_start < char_end:
        raise ValueError("invalid character span")
    touched = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < char_end and end > char_start
    ]
    if not touched:
        raise ValueError("exact character span has no tokenizer tokens")
    if min(offsets[index][0] for index in touched) > char_start:
        raise ValueError("token offsets do not cover evidence start")
    if max(offsets[index][1] for index in touched) < char_end:
        raise ValueError("token offsets do not cover evidence end")
    return touched[0], touched[-1]


@dataclass(frozen=True)
class RowAlignment:
    text_token_mask: list[bool]
    start_target: int
    end_target: int
    concept_target: int
    quality_weight: float
    full_to_prompt_offset: dict[int, tuple[int, int]]


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids, offsets = ids[0], offsets[0]
    result_ids = [int(value) for value in ids]
    result_offsets = [(int(start), int(end)) for start, end in offsets]
    if len(result_ids) != len(result_offsets):
        raise ValueError("tokenizer ids and offsets have different lengths")
    return result_ids, result_offsets


def align_parent_prompt(
    *,
    tokenizer: Any,
    full_input_ids: list[int],
    parent_user_text: str,
    rationale: dict[str, Any] | None,
) -> RowAlignment:
    """Locate the exact parent prompt inside the multimodal token sequence.

    Alignment is deliberately fail-closed. An evidence target is enabled only
    when its exact raw substring occurs once in the unchanged parent user text.
    The validation path uses the same mapping but never receives a rationale.
    """

    prompt_ids, prompt_offsets = _tokenize_with_offsets(tokenizer, parent_user_text)
    shift = _locate_unique_subsequence(full_input_ids, prompt_ids)
    token_mask = [False] * len(full_input_ids)
    full_to_prompt: dict[int, tuple[int, int]] = {}
    for local_index, offset in enumerate(prompt_offsets):
        full_index = shift + local_index
        token_mask[full_index] = True
        full_to_prompt[full_index] = offset
    no_evidence = len(full_input_ids)
    if not rationale or not rationale.get("has_evidence"):
        return RowAlignment(token_mask, no_evidence, no_evidence, -1, 0.0, full_to_prompt)

    exact = str(rationale.get("exact_span", ""))
    hits: list[int] = []
    cursor = 0
    while exact and (found := parent_user_text.find(exact, cursor)) >= 0:
        hits.append(found)
        cursor = found + 1
    concept = rationale.get("concept")
    if len(hits) != 1 or concept not in CONCEPTS:
        return RowAlignment(token_mask, no_evidence, no_evidence, -1, 0.0, full_to_prompt)
    local_start, local_end = _character_span_to_token_span(
        prompt_offsets, hits[0], hits[0] + len(exact)
    )
    return RowAlignment(
        token_mask,
        shift + local_start,
        shift + local_end,
        CONCEPTS.index(str(concept)),
        float(rationale.get("quality_weight", 0.0)),
        full_to_prompt,
    )


def predicted_token_span_to_canonical_offsets(
    *,
    start_token: int,
    end_token: int,
    alignment: RowAlignment,
    parent_user_text: str,
    name: object,
    description: object,
) -> tuple[int | None, int | None]:
    if start_token > end_token:
        return None, None
    covered = [alignment.full_to_prompt_offset.get(index) for index in range(start_token, end_token + 1)]
    if not covered or any(value is None for value in covered):
        return None, None
    offsets = [value for value in covered if value is not None]
    prompt_start = min(value[0] for value in offsets)
    prompt_end = max(value[1] for value in offsets)
    quote = parent_user_text[prompt_start:prompt_end]
    canonical, _ = canonical_text(name, description)
    hits: list[int] = []
    cursor = 0
    while quote and (found := canonical.find(quote, cursor)) >= 0:
        hits.append(found)
        cursor = found + 1
    if len(hits) != 1:
        return None, None
    return hits[0], hits[0] + len(quote)

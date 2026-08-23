from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    verdict: float = 1.0
    span: float = 0.10
    concept: float = 0.05


FROZEN_LOSS_WEIGHTS = LossWeights()


class SpanConceptHead(nn.Module):
    """Small trainable heads over one Qwen hidden-state sequence.

    ``no_evidence`` is appended as the last class of both boundary distributions.
    The closed concept vocabulary is owned by ``protocol.CONCEPTS``.
    """

    def __init__(self, hidden_size: int, concept_count: int) -> None:
        super().__init__()
        self.start = nn.Linear(hidden_size, 1)
        self.end = nn.Linear(hidden_size, 1)
        self.no_evidence = nn.Linear(hidden_size, 2)
        self.concept = nn.Linear(hidden_size, concept_count)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        text_token_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        valid = attention_mask.bool() & text_token_mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError("every row must expose at least one canonical-text token")
        masked_start = self.start(hidden_states).squeeze(-1).masked_fill(~valid, -1e4)
        masked_end = self.end(hidden_states).squeeze(-1).masked_fill(~valid, -1e4)
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None, :]
        last = torch.where(attention_mask.bool(), positions, -1).max(dim=1).values
        pooled = hidden_states[
            torch.arange(hidden_states.shape[0], device=hidden_states.device), last
        ]
        no_evidence = self.no_evidence(pooled)
        return {
            "start_logits": torch.cat([masked_start, no_evidence[:, :1]], dim=1),
            "end_logits": torch.cat([masked_end, no_evidence[:, 1:]], dim=1),
            "concept_logits": self.concept(pooled),
        }


def multitask_loss(
    *,
    verdict_logits: torch.Tensor,
    auxiliary: dict[str, torch.Tensor],
    verdict_targets: torch.Tensor,
    start_targets: torch.Tensor,
    end_targets: torch.Tensor,
    concept_targets: torch.Tensor,
    quality_weights: torch.Tensor,
    weights: LossWeights = FROZEN_LOSS_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Frozen verdict-primary objective.

    Rows without a SAFE exact span have quality weight zero, so all rationale
    terms are physically masked. Verdict CE is always present.
    """

    verdict = F.binary_cross_entropy_with_logits(verdict_logits.float(), verdict_targets.float())
    quality = quality_weights.float()
    denominator = quality.sum().clamp_min(1.0)

    start_per_row = F.cross_entropy(
        auxiliary["start_logits"].float(), start_targets, reduction="none"
    )
    end_per_row = F.cross_entropy(auxiliary["end_logits"].float(), end_targets, reduction="none")
    span = ((start_per_row + end_per_row) * quality).sum() / denominator
    concept_per_row = F.cross_entropy(
        auxiliary["concept_logits"].float(), concept_targets.clamp_min(0), reduction="none"
    )
    concept = (concept_per_row * quality).sum() / denominator
    total = (
        weights.verdict * verdict
        + weights.span * span
        + weights.concept * concept
    )
    return total, {
        "verdict": verdict.detach(),
        "span": span.detach(),
        "concept": concept.detach(),
        "quality_rows": (quality > 0).sum().detach(),
    }


def locate_unique_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence:
        raise ValueError("canonical text token sequence is empty")
    hits = [
        index
        for index in range(len(sequence) - len(subsequence) + 1)
        if sequence[index : index + len(subsequence)] == subsequence
    ]
    if len(hits) != 1:
        raise ValueError(
            f"canonical text must occur exactly once in model input; found={len(hits)}"
        )
    return hits[0]


def character_span_to_token_span(
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
    covered_start = min(offsets[index][0] for index in touched)
    covered_end = max(offsets[index][1] for index in touched)
    if covered_start > char_start or covered_end < char_end:
        raise ValueError("token offsets do not fully cover exact character span")
    return touched[0], touched[-1]


def attach_span_targets(
    *,
    full_input_ids: list[int],
    canonical_input_ids: list[int],
    canonical_offsets: list[tuple[int, int]],
    rationale: dict[str, Any],
) -> tuple[int, int, list[bool]]:
    shift = locate_unique_subsequence(full_input_ids, canonical_input_ids)
    token_mask = [False] * len(full_input_ids)
    for index in range(shift, shift + len(canonical_input_ids)):
        token_mask[index] = True
    no_evidence_index = len(full_input_ids)
    if not rationale.get("has_evidence"):
        return no_evidence_index, no_evidence_index, token_mask
    local_start, local_end = character_span_to_token_span(
        canonical_offsets, int(rationale["char_start"]), int(rationale["char_end"])
    )
    return shift + local_start, shift + local_end, token_mask

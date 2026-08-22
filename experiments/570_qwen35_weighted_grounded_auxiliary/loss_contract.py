from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

EXPERIMENT_ID = "570"
CONTRACT_VERSION = "weighted_grounded_auxiliary_loss_v1"
VERDICT_TOKEN_WEIGHT = 1.0
EVIDENCE_TOKEN_WEIGHT = 0.05
IGNORE_INDEX = -100


def contract_sha256() -> str:
    payload = json.dumps(
        {
            "contract_version": CONTRACT_VERSION,
            "verdict_token_weight": VERDICT_TOKEN_WEIGHT,
            "evidence_token_weight": EVIDENCE_TOKEN_WEIGHT,
            "normalization": "sum_active_token_weights",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_loss_weights(
    labels: torch.Tensor,
    expected_first_token_ids: list[int],
) -> torch.Tensor:
    """Assign 1.0 to the verdict and 0.05 to every later target token."""

    if labels.ndim != 2:
        raise ValueError("labels must be a rank-two tensor")
    if labels.shape[0] != len(expected_first_token_ids):
        raise ValueError("one expected verdict token is required for every row")
    weights = torch.zeros_like(labels, dtype=torch.float32)
    for row_index, expected_token_id in enumerate(expected_first_token_ids):
        supervised = labels[row_index].ne(IGNORE_INDEX).nonzero(as_tuple=False).flatten()
        if len(supervised) < 2:
            raise ValueError("six-line target must contain a verdict and evidence tokens")
        first = int(supervised[0])
        if int(labels[row_index, first]) != int(expected_token_id):
            raise ValueError("first supervised token is not the expected atomic verdict")
        weights[row_index, first] = VERDICT_TOKEN_WEIGHT
        weights[row_index, supervised[1:]] = EVIDENCE_TOKEN_WEIGHT
    return weights


def weighted_causal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    """Causal token CE normalized by the sum of active token weights.

    The all-one branch intentionally calls the ordinary mean CE directly.  It
    is the exact null control for the weighting implementation, not a numeric
    approximation assembled from an unreduced loss.
    """

    if logits.ndim != 3 or labels.ndim != 2 or loss_weights.ndim != 2:
        raise ValueError("expected logits [B,T,V], labels [B,T], weights [B,T]")
    if logits.shape[:2] != labels.shape or labels.shape != loss_weights.shape:
        raise ValueError("logits, labels and loss weights have incompatible shapes")
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = loss_weights[:, 1:].contiguous().float()
    active = shift_labels.ne(IGNORE_INDEX)
    if not bool(active.any()):
        raise ValueError("weighted causal CE received no supervised tokens")
    active_weights = shift_weights[active]
    if bool((active_weights <= 0).any()):
        raise ValueError("every supervised token must have a positive loss weight")
    if bool(torch.equal(active_weights, torch.ones_like(active_weights))):
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shift_labels)
    effective_weights = shift_weights * active
    return (token_loss * effective_weights).sum() / effective_weights.sum()


def exact_unit_weight_null_control(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = labels.ne(IGNORE_INDEX).to(dtype=torch.float32)
    candidate = weighted_causal_cross_entropy(logits, labels, weights)
    reference = F.cross_entropy(
        logits[:, :-1, :].contiguous().float().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
    return candidate, reference


@dataclass
class RuntimeLossAudit:
    batches: int = 0
    training_occurrences: int = 0
    verdict_tokens: int = 0
    evidence_tokens: int = 0
    null_control_checks: int = 0

    def observe(self, labels: torch.Tensor, loss_weights: torch.Tensor) -> None:
        active = labels.ne(IGNORE_INDEX)
        for row_index in range(labels.shape[0]):
            positions = active[row_index].nonzero(as_tuple=False).flatten()
            if len(positions) < 2:
                raise ValueError("runtime target contains fewer than two supervised tokens")
            first_weight = float(loss_weights[row_index, positions[0]])
            later = loss_weights[row_index, positions[1:]]
            if first_weight != VERDICT_TOKEN_WEIGHT:
                raise ValueError("runtime verdict weight differs from the frozen contract")
            if not bool(torch.all(later.eq(EVIDENCE_TOKEN_WEIGHT))):
                raise ValueError("runtime evidence weight differs from the frozen contract")
            if bool(loss_weights[row_index, ~active[row_index]].ne(0).any()):
                raise ValueError("unsupervised tokens received a nonzero loss weight")
            self.verdict_tokens += 1
            self.evidence_tokens += int(len(positions) - 1)
        self.batches += 1
        self.training_occurrences += int(labels.shape[0])

    def record_null_control(self) -> None:
        self.null_control_checks += 1

    def finalize(self, output: Path, *, expected_training_occurrences: int) -> dict[str, Any]:
        failures = []
        if self.training_occurrences != expected_training_occurrences:
            failures.append("training_occurrence_count_mismatch")
        if self.verdict_tokens != expected_training_occurrences:
            failures.append("verdict_token_count_mismatch")
        if self.evidence_tokens <= 0:
            failures.append("no_evidence_tokens_observed")
        if self.null_control_checks != 1:
            failures.append("exact_null_control_not_executed_once")
        report = {
            "experiment_id": EXPERIMENT_ID,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": contract_sha256(),
            "verdict_token_weight": VERDICT_TOKEN_WEIGHT,
            "evidence_token_weight": EVIDENCE_TOKEN_WEIGHT,
            "normalization": "sum_active_token_weights",
            "batches": self.batches,
            "training_occurrences": self.training_occurrences,
            "verdict_tokens": self.verdict_tokens,
            "evidence_tokens": self.evidence_tokens,
            "exact_null_control": {
                "executed": self.null_control_checks == 1,
                "checks": self.null_control_checks,
                "bit_exact": self.null_control_checks == 1,
            },
            "failures": failures,
            "decision": "GO" if not failures else "NO_GO",
        }
        if failures:
            raise ValueError(f"weighted loss runtime audit failed: {failures}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report


def install_weighted_forward(model: Any, audit: RuntimeLossAudit) -> Any:
    """Install the sole training change before PEFT wraps the frozen parent."""

    original_forward: Callable[..., Any] = model.forward

    def weighted_forward(
        self: Any,
        *args: Any,
        labels: torch.Tensor | None = None,
        loss_weights: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if loss_weights is None:
            return original_forward(*args, labels=labels, **kwargs)
        if labels is None:
            raise ValueError("loss_weights require labels")
        outputs = original_forward(*args, labels=None, **kwargs)
        if audit.null_control_checks == 0:
            candidate, reference = exact_unit_weight_null_control(outputs.logits, labels)
            if not bool(torch.equal(candidate.detach(), reference.detach())):
                raise ValueError("unit-weight null control is not bit-exact")
            audit.record_null_control()
        outputs.loss = weighted_causal_cross_entropy(outputs.logits, labels, loss_weights)
        return outputs

    model.forward = MethodType(weighted_forward, model)
    return model

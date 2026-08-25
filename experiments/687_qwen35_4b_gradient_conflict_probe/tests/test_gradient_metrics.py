from __future__ import annotations

import math

import pytest
from gradient_metrics import (
    CHECKPOINT_STEPS,
    DIAGNOSTIC_EFFECTIVE_BATCHES,
    gradient_metrics,
    pcgrad_gate,
    wilson_interval,
)


def test_orthogonal_gradients_preserve_rank_signal() -> None:
    value = gradient_metrics(hard_sq=4.0, rank_sq=9.0, dot=0.0)
    assert value["cosine"] == pytest.approx(0.0)
    assert value["weighted_norm_ratio"] == pytest.approx(0.75)
    assert value["hard_cancellation"] == 0.0
    assert value["projection_retention"] == pytest.approx(1.0)


def test_opposing_gradients_report_cancellation_and_projection() -> None:
    value = gradient_metrics(hard_sq=4.0, rank_sq=4.0, dot=-2.0)
    assert value["conflict"] is True
    assert value["cosine"] == pytest.approx(-0.5)
    assert value["hard_cancellation"] == pytest.approx(0.25)
    assert value["projection_retention"] == pytest.approx(math.sqrt(0.75))


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(20, 80)
    assert low < 0.25 < high


def _row(*, conflict: bool, cancellation: float = 0.1, retention: float = 0.8):
    return {
        "cosine": -0.3 if conflict else 0.3,
        "weighted_norm_ratio": 0.4,
        "hard_cancellation": cancellation if conflict else 0.0,
        "combined_hard_alignment": 0.9,
        "projection_retention": retention,
        "hard_grad_norm": 1.0,
        "rank_grad_norm": 0.8,
        "conflict": conflict,
    }


def test_pcgrad_gate_opens_only_for_stable_conflict() -> None:
    rows = {
        step: [_row(conflict=index < 8) for index in range(DIAGNOSTIC_EFFECTIVE_BATCHES)]
        for step in CHECKPOINT_STEPS
    }
    assert pcgrad_gate(rows)["decision"] == "OPEN_ASYMMETRIC_PCGRAD_SCREEN"


def test_pcgrad_gate_rejects_low_conflict() -> None:
    rows = {
        step: [_row(conflict=False) for _ in range(DIAGNOSTIC_EFFECTIVE_BATCHES)]
        for step in CHECKPOINT_STEPS
    }
    assert pcgrad_gate(rows)["decision"] == "REJECT_PCGRAD_LOW_CONFLICT"


def test_pcgrad_gate_does_not_hide_low_conflict_retention_with_nonconflicts() -> None:
    rows = {
        step: [
            _row(conflict=index < 4, retention=0.0 if index < 4 else 1.0)
            for index in range(DIAGNOSTIC_EFFECTIVE_BATCHES)
        ]
        for step in CHECKPOINT_STEPS
    }
    result = pcgrad_gate(rows)
    assert result["aggregate"]["projection_retention"]["median"] == 1.0
    assert result["aggregate"]["conflicting_projection_retention_median"] == 0.0
    assert result["decision"] == "REJECT_PCGRAD_LOW_RETAINED_RANK_SIGNAL"


def test_checkpoint_coverage_is_fail_closed() -> None:
    rows = {
        step: [_row(conflict=True) for _ in range(DIAGNOSTIC_EFFECTIVE_BATCHES)]
        for step in CHECKPOINT_STEPS[:-1]
    }
    with pytest.raises(ValueError, match="checkpoint coverage"):
        pcgrad_gate(rows)

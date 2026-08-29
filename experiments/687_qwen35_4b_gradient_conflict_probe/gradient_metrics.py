from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any

RANK_WEIGHT = 0.5
CHECKPOINT_STEPS = (0, 170, 340, 510, 680)
DIAGNOSTIC_EFFECTIVE_BATCHES = 16


def gradient_metrics(hard_sq: float, rank_sq: float, dot: float) -> dict[str, float | bool]:
    if not all(math.isfinite(value) for value in (hard_sq, rank_sq, dot)):
        raise ValueError("gradient statistics must be finite")
    if hard_sq <= 0 or rank_sq <= 0:
        raise ValueError("gradient norms must be positive")
    hard_norm = math.sqrt(hard_sq)
    rank_norm = math.sqrt(rank_sq)
    cosine = max(-1.0, min(1.0, dot / (hard_norm * rank_norm)))
    weighted_ratio = RANK_WEIGHT * rank_norm / hard_norm
    conflict = dot < 0
    cancellation = max(0.0, -(RANK_WEIGHT * dot) / hard_sq)
    combined_sq = hard_sq + (RANK_WEIGHT**2) * rank_sq + 2 * RANK_WEIGHT * dot
    combined_sq = max(combined_sq, 0.0)
    combined_alignment = (
        (hard_sq + RANK_WEIGHT * dot) / (hard_norm * math.sqrt(combined_sq))
        if combined_sq > 0
        else -1.0
    )
    if conflict:
        projected_rank_sq = max(0.0, rank_sq - (dot * dot) / hard_sq)
    else:
        projected_rank_sq = rank_sq
    retention = math.sqrt(projected_rank_sq / rank_sq)
    return {
        "hard_grad_norm": hard_norm,
        "rank_grad_norm": rank_norm,
        "cosine": cosine,
        "weighted_norm_ratio": weighted_ratio,
        "conflict": conflict,
        "hard_cancellation": cancellation,
        "combined_hard_alignment": max(-1.0, min(1.0, combined_alignment)),
        "projection_retention": max(0.0, min(1.0, retention)),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    proportion = successes / total
    denominator = 1 + (z * z) / total
    center = (proportion + (z * z) / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt((proportion * (1 - proportion) + (z * z) / (4 * total)) / total)
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        raise ValueError("no gradient measurements")
    conflicts = sum(bool(row["conflict"]) for row in values)
    low, high = wilson_interval(conflicts, len(values))
    numeric = (
        "cosine",
        "weighted_norm_ratio",
        "hard_cancellation",
        "combined_hard_alignment",
        "projection_retention",
    )
    result: dict[str, Any] = {
        "measurements": len(values),
        "conflicts": conflicts,
        "conflict_rate": conflicts / len(values),
        "conflict_wilson_95": [low, high],
    }
    for key in numeric:
        series = [float(row[key]) for row in values]
        result[key] = {
            "median": statistics.median(series),
            "p10": _percentile(series, 0.10),
            "p90": _percentile(series, 0.90),
        }
    conflicting_cancellation = [
        float(row["hard_cancellation"]) for row in values if row["conflict"]
    ]
    conflicting_retention = [
        float(row["projection_retention"]) for row in values if row["conflict"]
    ]
    result["conflicting_hard_cancellation_median"] = (
        statistics.median(conflicting_cancellation)
        if conflicting_cancellation
        else 0.0
    )
    result["conflicting_projection_retention_median"] = (
        statistics.median(conflicting_retention) if conflicting_retention else 0.0
    )
    return result


def pcgrad_gate(checkpoint_rows: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    if tuple(sorted(checkpoint_rows)) != CHECKPOINT_STEPS:
        raise ValueError("checkpoint coverage differs from frozen contract")
    if any(len(rows) != DIAGNOSTIC_EFFECTIVE_BATCHES for rows in checkpoint_rows.values()):
        raise ValueError("diagnostic batch coverage differs from frozen contract")
    pooled = [row for step in CHECKPOINT_STEPS for row in checkpoint_rows[step]]
    aggregate = summarize(pooled)
    per_checkpoint = {str(step): summarize(checkpoint_rows[step]) for step in CHECKPOINT_STEPS}
    stable_checkpoints = sum(
        metrics["conflict_rate"] >= 0.15 for metrics in per_checkpoint.values()
    )
    open_conditions = {
        "complete_finite": all(
            all(
                math.isfinite(float(value))
                for key, value in row.items()
                if key not in {"conflict", "checkpoint_step", "batch_index", "group"}
            )
            for row in pooled
        ),
        "pooled_conflict_rate_ge_0_25": aggregate["conflict_rate"] >= 0.25,
        "wilson_lower_ge_0_15": aggregate["conflict_wilson_95"][0] >= 0.15,
        "three_checkpoints_ge_0_15": stable_checkpoints >= 3,
        "conflicting_cancellation_median_ge_0_05": aggregate[
            "conflicting_hard_cancellation_median"
        ]
        >= 0.05,
        "weighted_norm_ratio_median_ge_0_10": aggregate["weighted_norm_ratio"][
            "median"
        ]
        >= 0.10,
        "conflicting_projection_retention_median_ge_0_50": aggregate[
            "conflicting_projection_retention_median"
        ]
        >= 0.50,
    }
    stop_low_conflict = (
        aggregate["conflict_rate"] < 0.10
        or aggregate["conflict_wilson_95"][1] < 0.20
    )
    if all(open_conditions.values()):
        decision = "OPEN_ASYMMETRIC_PCGRAD_SCREEN"
    elif stop_low_conflict:
        decision = "REJECT_PCGRAD_LOW_CONFLICT"
    elif (
        aggregate["conflict_rate"] < 0.25
        and aggregate["weighted_norm_ratio"]["median"] > 2.0
    ):
        decision = "ROUTE_MAGNITUDE_CONTROL"
    elif (
        aggregate["conflict_rate"] >= 0.25
        and aggregate["conflicting_projection_retention_median"] < 0.50
    ):
        decision = "REJECT_PCGRAD_LOW_RETAINED_RANK_SIGNAL"
    else:
        decision = "DIAGNOSTIC_AMBIGUOUS"
    return {
        "decision": decision,
        "open_conditions": open_conditions,
        "stable_conflict_checkpoints": stable_checkpoints,
        "aggregate": aggregate,
        "per_checkpoint": per_checkpoint,
    }

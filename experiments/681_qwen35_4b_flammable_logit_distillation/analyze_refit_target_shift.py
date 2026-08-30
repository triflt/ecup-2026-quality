from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_oof(paths: list[Path]) -> dict[str, float]:
    result: dict[str, float] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row_id = str(row["id"])
            score = float(row["score"])
            if row_id in result or not math.isfinite(score):
                raise ValueError("duplicate or non-finite OOF score")
            result[row_id] = score
    return result


def correlation(left: list[float], right: list[float]) -> float:
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator


def sigmoid_t2(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, value / 2.0))))


def analyze(runtime: Path, oof: dict[str, float]) -> dict[str, Any]:
    audit = json.loads((runtime / "runtime_audit.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (runtime / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    in_sample = [float(row["teacher_score"]) for row in rows]
    cross_fitted = [oof[str(row["id"])] for row in rows]
    in_sample_soft = [sigmoid_t2(value) for value in in_sample]
    cross_fitted_soft = [sigmoid_t2(value) for value in cross_fitted]
    return {
        "outer_fold": int(audit["outer_fold"]),
        "occurrences": len(rows),
        "unique_ids": len({str(row["id"]) for row in rows}),
        "pearson_logit_correlation": correlation(in_sample, cross_fitted),
        "mean_absolute_logit_difference": statistics.mean(
            abs(a - b) for a, b in zip(in_sample, cross_fitted, strict=True)
        ),
        "sign_disagreement_occurrences": sum(
            (a >= 0.0) != (b >= 0.0)
            for a, b in zip(in_sample, cross_fitted, strict=True)
        ),
        "mean_absolute_temperature_2_soft_target_difference": statistics.mean(
            abs(a - b)
            for a, b in zip(in_sample_soft, cross_fitted_soft, strict=True)
        ),
        "in_sample_ambiguous_soft_targets": sum(0.1 < value < 0.9 for value in in_sample_soft),
        "oof_ambiguous_soft_targets": sum(0.1 < value < 0.9 for value in cross_fitted_soft),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--teacher-oof", type=Path, action="append", required=True)
    args = parser.parse_args()
    oof_scores = load_oof(args.teacher_oof)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "681",
                "diagnostic_only": True,
                "folds": [analyze(runtime, oof_scores) for runtime in args.runtime_dir],
            },
            indent=2,
            sort_keys=True,
        )
    )

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        raise ValueError("average precision requires positive labels")
    ranked = sorted(zip(scores, labels, strict=True), reverse=True)
    true_positives = 0
    total = 0
    area = 0.0
    previous_recall = 0.0
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        end = index
        while end < len(ranked) and ranked[end][0] == score:
            true_positives += ranked[end][1]
            total += 1
            end += 1
        recall = true_positives / positives
        area += (recall - previous_recall) * (true_positives / total)
        previous_recall = recall
        index = end
    return area


def analyze(runtime_dir: Path) -> dict[str, Any]:
    audit = json.loads((runtime_dir / "runtime_audit.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (runtime_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["teacher_score"]) for row in rows]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("non-finite teacher score")
    predictions = [int(score >= 0.0) for score in scores]
    true_positive = sum(label == prediction == 1 for label, prediction in zip(labels, predictions))
    false_positive = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    false_negative = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    soft = [1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, score / 2.0)))) for score in scores]
    return {
        "schema_version": 1,
        "experiment_id": "681",
        "outer_fold": audit["outer_fold"],
        "rows": len(rows),
        "unique_ids": len({str(row["id"]) for row in rows}),
        "positive_occurrences": sum(labels),
        "teacher_train_average_precision": average_precision(labels, scores),
        "teacher_train_f1": 2 * true_positive / max(
            1, 2 * true_positive + false_positive + false_negative
        ),
        "teacher_false_positive_occurrences": false_positive,
        "teacher_false_negative_occurrences": false_negative,
        "hard_disagreement_occurrences": sum(
            label != prediction for label, prediction in zip(labels, predictions)
        ),
        "median_absolute_teacher_logit": statistics.median(abs(score) for score in scores),
        "temperature_2_soft_target_mean_positive": statistics.mean(
            value for value, label in zip(soft, labels) if label == 1
        ),
        "temperature_2_soft_target_mean_negative": statistics.mean(
            value for value, label in zip(soft, labels) if label == 0
        ),
        "temperature_2_ambiguous_soft_targets": sum(0.1 < value < 0.9 for value in soft),
        "diagnostic_only": True,
        "gate_changed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    print(json.dumps(analyze(parser.parse_args().runtime_dir), indent=2, sort_keys=True))

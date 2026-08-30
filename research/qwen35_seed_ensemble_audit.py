from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def category_scores(labels, predictions, categories, mask=None):
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    values = {}
    for category in sorted(np.unique(categories)):
        selected = mask & (categories == category)
        values[category] = f1(labels[selected], predictions[selected])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-key", default="nested_predictions")
    parser.add_argument("--candidate-key", default="probability_nested_predictions")
    parser.add_argument(
        "--candidate-name", default="probability mean of Qwen3.5 seeds 42 and 31415"
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=23031415)
    args = parser.parse_args()

    baseline = np.load(args.baseline, allow_pickle=True)
    candidate = np.load(args.candidate, allow_pickle=True)
    ids = baseline["ids"].astype(str)
    labels = baseline["labels"].astype(np.int8)
    categories = baseline["categories"].astype(str)
    fold_ids = baseline["folds"].astype(np.int8)
    baseline_predictions = baseline[args.baseline_key].astype(np.int8)
    candidate_predictions = candidate[args.candidate_key].astype(np.int8)
    for key, expected in [
        ("ids", ids),
        ("labels", labels),
        ("categories", categories),
        ("folds", fold_ids),
    ]:
        if not np.array_equal(candidate[key].astype(expected.dtype), expected):
            raise ValueError(f"candidate {key} mismatch")

    fold_frame = pd.read_csv(args.folds, dtype={"id": str})
    by_id = fold_frame.set_index("id")
    aligned = by_id.loc[ids]
    if not np.array_equal(aligned["fold"].to_numpy(np.int8), fold_ids):
        raise ValueError("validation fold mismatch")
    group_hashes = aligned["group_hash"].astype(str).to_numpy()

    baseline_category = category_scores(labels, baseline_predictions, categories)
    candidate_category = category_scores(labels, candidate_predictions, categories)
    baseline_macro = float(np.mean(list(baseline_category.values())))
    candidate_macro = float(np.mean(list(candidate_category.values())))

    fold_rows = []
    for fold in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold
        old = category_scores(labels, baseline_predictions, categories, mask)
        new = category_scores(labels, candidate_predictions, categories, mask)
        old_macro = float(np.mean(list(old.values())))
        new_macro = float(np.mean(list(new.values())))
        fold_rows.append({
            "fold": int(fold),
            "baseline_macro_f1": old_macro,
            "candidate_macro_f1": new_macro,
            "delta": new_macro - old_macro,
            "baseline_category_f1": old,
            "candidate_category_f1": new,
        })

    rng = np.random.default_rng(args.seed)
    group_rows = {}
    for category in sorted(np.unique(categories)):
        positions = np.flatnonzero(categories == category)
        groups = {}
        for position in positions:
            groups.setdefault(group_hashes[position], []).append(position)
        group_rows[category] = [np.asarray(row_ids) for row_ids in groups.values()]
    deltas = np.empty(args.bootstrap, dtype=np.float64)
    for iteration in range(args.bootstrap):
        old_values, new_values = [], []
        for category in sorted(group_rows):
            groups = group_rows[category]
            sampled = rng.integers(0, len(groups), size=len(groups))
            positions = np.concatenate([groups[index] for index in sampled])
            old_values.append(f1(labels[positions], baseline_predictions[positions]))
            new_values.append(f1(labels[positions], candidate_predictions[positions]))
        deltas[iteration] = np.mean(new_values) - np.mean(old_values)

    disagreements = {}
    for category in sorted(np.unique(categories)):
        mask = categories == category
        old_correct = baseline_predictions[mask] == labels[mask]
        new_correct = candidate_predictions[mask] == labels[mask]
        disagreements[category] = {
            "changed": int((baseline_predictions[mask] != candidate_predictions[mask]).sum()),
            "corrected": int((~old_correct & new_correct).sum()),
            "regressed": int((old_correct & ~new_correct).sum()),
        }

    category_deltas = {
        category: candidate_category[category] - baseline_category[category]
        for category in baseline_category
    }
    wins = sum(row["delta"] > 0 for row in fold_rows)
    report = {
        "evaluation_version": "nested_grouped_v1",
        "candidate": args.candidate_name,
        "baseline_macro_f1": baseline_macro,
        "candidate_macro_f1": candidate_macro,
        "delta_macro_f1": candidate_macro - baseline_macro,
        "baseline_category_f1": baseline_category,
        "candidate_category_f1": candidate_category,
        "category_delta": category_deltas,
        "folds_won": wins,
        "folds": fold_rows,
        "disagreements": disagreements,
        "group_bootstrap": {
            "iterations": args.bootstrap,
            "seed": args.seed,
            "delta_mean": float(deltas.mean()),
            "delta_ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
            "probability_delta_positive": float((deltas > 0).mean()),
        },
        "acceptance": {
            "delta_at_least_0_001": candidate_macro - baseline_macro >= 0.001,
            "wins_at_least_4_of_5_folds": wins >= 4,
            "no_category_drop_over_0_005": min(category_deltas.values()) >= -0.005,
        },
    }
    report["accepted"] = all(report["acceptance"].values())
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

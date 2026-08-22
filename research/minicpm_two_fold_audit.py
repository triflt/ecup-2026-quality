from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qwen35_locked_190_audit import LOCKED_CONFIG
from qwen35_seed_ensemble_cv import BASE, QWEN3VL, QWEN35_SEED_A, f1, rank01


SCREEN_FOLDS = (0, 3)


def load_candidate(paths: list[str], ids: np.ndarray, folds: np.ndarray, categories: np.ndarray) -> np.ndarray:
    scores = np.full(len(ids), np.nan, dtype=np.float32)
    position = {item_id: index for index, item_id in enumerate(ids)}
    for expected_fold, path in zip(SCREEN_FOLDS, paths):
        frame = pd.read_csv(path, dtype={"id": str})
        if set(frame.fold.astype(int)) != {expected_fold}:
            raise ValueError(f"fold mismatch in {path}")
        for row in frame.itertuples(index=False):
            index = position[row.id]
            if folds[index] != expected_fold:
                raise ValueError(f"registry fold mismatch for id={row.id}")
            scores[index] = float(row.lora_score)
    for fold in SCREEN_FOLDS:
        for category in sorted(np.unique(categories)):
            mask = (folds == fold) & (categories == category)
            if np.isnan(scores[mask]).any():
                raise ValueError(f"missing candidate scores for fold={fold} category={category}")
            scores[mask] = rank01(scores[mask])
    return scores


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 1000)))
    best = (-1.0, 0.0)
    for threshold in candidates:
        value = f1(labels, scores >= threshold)
        if value > best[0]:
            best = (value, float(threshold))
    return best


def cross_fold_predictions(
    matrix: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, dict]:
    predictions = np.zeros(len(labels), dtype=np.int8)
    detail = {}
    for category in sorted(np.unique(categories)):
        weights = np.asarray(LOCKED_CONFIG[category]["weights"], dtype=np.float32)
        scores = np.sum(matrix * weights[None, :], axis=1, dtype=np.float32)
        rows = []
        for validation_fold, calibration_fold in ((0, 3), (3, 0)):
            calibration = (folds == calibration_fold) & (categories == category)
            validation = (folds == validation_fold) & (categories == category)
            calibration_f1, threshold = best_threshold(labels[calibration], scores[calibration])
            predictions[validation] = (scores[validation] >= threshold).astype(np.int8)
            rows.append({
                "validation_fold": validation_fold,
                "calibration_fold": calibration_fold,
                "threshold": threshold,
                "calibration_f1": calibration_f1,
                "validation_f1": f1(labels[validation], predictions[validation]),
            })
        detail[category] = rows
    return predictions, detail


def category_metrics(labels: np.ndarray, predictions: np.ndarray, categories: np.ndarray, mask: np.ndarray) -> dict:
    return {
        category: f1(labels[mask & (categories == category)], predictions[mask & (categories == category)])
        for category in sorted(np.unique(categories))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold0", required=True)
    parser.add_argument("--fold3", required=True)
    parser.add_argument("--folds", default="validation/grouped_text_v1/folds.csv")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=29042)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = np.load(BASE, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    original = np.load(QWEN35_SEED_A, allow_pickle=True)
    ids = base["ids"].astype(str)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    folds = base["fold_ids"].astype(np.int8)
    registry = pd.read_csv(args.folds, dtype={"id": str}).set_index("id").loc[ids]
    if not np.array_equal(registry.fold.to_numpy(np.int8), folds):
        raise ValueError("fold registry mismatch")

    candidate_rank = load_candidate([args.fold0, args.fold3], ids, folds, categories)
    base_rank = original["base_rank"].astype(np.float32)
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)
    original_rank = original["lora_rank"].astype(np.float32)
    baseline, baseline_detail = cross_fold_predictions(
        np.column_stack([base_rank, qwen3vl_rank, original_rank]), labels, categories, folds
    )
    candidate, candidate_detail = cross_fold_predictions(
        np.column_stack([base_rank, qwen3vl_rank, candidate_rank]), labels, categories, folds
    )
    screen = np.isin(folds, SCREEN_FOLDS)
    old_category = category_metrics(labels, baseline, categories, screen)
    new_category = category_metrics(labels, candidate, categories, screen)
    old_macro = float(np.mean(list(old_category.values())))
    new_macro = float(np.mean(list(new_category.values())))
    fold_rows = []
    for fold in SCREEN_FOLDS:
        mask = folds == fold
        old = category_metrics(labels, baseline, categories, mask)
        new = category_metrics(labels, candidate, categories, mask)
        fold_rows.append({
            "fold": fold,
            "baseline_category_f1": old,
            "candidate_category_f1": new,
            "baseline_macro_f1": float(np.mean(list(old.values()))),
            "candidate_macro_f1": float(np.mean(list(new.values()))),
            "delta": float(np.mean(list(new.values())) - np.mean(list(old.values()))),
        })

    group_hashes = registry.group_hash.astype(str).to_numpy()
    groups = {}
    for category in sorted(np.unique(categories)):
        local = np.flatnonzero(screen & (categories == category))
        category_groups = {}
        for index in local:
            category_groups.setdefault(group_hashes[index], []).append(index)
        groups[category] = [np.asarray(value) for value in category_groups.values()]
    rng = np.random.default_rng(args.seed)
    deltas = np.empty(args.bootstrap, dtype=np.float64)
    for iteration in range(args.bootstrap):
        old_values, new_values = [], []
        for category in sorted(groups):
            category_groups = groups[category]
            sampled = rng.integers(0, len(category_groups), len(category_groups))
            positions = np.concatenate([category_groups[index] for index in sampled])
            old_values.append(f1(labels[positions], baseline[positions]))
            new_values.append(f1(labels[positions], candidate[positions]))
        deltas[iteration] = np.mean(new_values) - np.mean(old_values)

    category_delta = {key: new_category[key] - old_category[key] for key in old_category}
    wins = sum(row["delta"] > 0 for row in fold_rows)
    report = {
        "protocol": "minicpm_predeclared_two_fold_screen_v1",
        "screen_folds": list(SCREEN_FOLDS),
        "calibration": "fold 0 threshold from fold 3 and fold 3 threshold from fold 0",
        "baseline_macro_f1": old_macro,
        "candidate_macro_f1": new_macro,
        "delta_macro_f1": new_macro - old_macro,
        "baseline_category_f1": old_category,
        "candidate_category_f1": new_category,
        "category_delta": category_delta,
        "folds_won": wins,
        "folds": fold_rows,
        "changed_predictions": int((baseline[screen] != candidate[screen]).sum()),
        "corrected": int(((baseline[screen] != labels[screen]) & (candidate[screen] == labels[screen])).sum()),
        "regressed": int(((baseline[screen] == labels[screen]) & (candidate[screen] != labels[screen])).sum()),
        "group_bootstrap": {
            "iterations": args.bootstrap,
            "seed": args.seed,
            "delta_ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
            "probability_delta_positive": float((deltas > 0).mean()),
        },
        "recipe_detail": {"baseline": baseline_detail, "candidate": candidate_detail},
        "acceptance": {
            "positive_mean_delta": new_macro > old_macro,
            "wins_both_folds": wins == 2,
            "no_category_drop_over_0_005": min(category_delta.values()) >= -0.005,
        },
    }
    report["accepted"] = all(report["acceptance"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

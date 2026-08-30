from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qwen35_seed_ensemble_cv import (
    BASE,
    QWEN3VL,
    QWEN35_SEED_A,
    best_threshold,
    f1,
    fold_category_ranks,
    load_seed_predictions,
)


LOCKED_CONFIG = {
    "БАД": {
        "weights": [0.50, 0.25, 0.25],
        "production_threshold": 0.27193570137023926,
    },
    "Легковоспламеняющиеся": {
        "weights": [0.15, 0.10, 0.75],
        "production_threshold": 0.953912615776062,
    },
}


def category_scores(labels, predictions, categories, mask=None):
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    return {
        category: f1(
            labels[mask & (categories == category)],
            predictions[mask & (categories == category)],
        )
        for category in sorted(np.unique(categories))
    }


def evaluate_locked(
    matrix: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    nested = np.zeros(len(labels), dtype=np.int8)
    production = np.zeros(len(labels), dtype=np.int8)
    report = {}
    for category in sorted(np.unique(categories)):
        config = LOCKED_CONFIG[category]
        weights = np.asarray(config["weights"], dtype=np.float32)
        scores = np.sum(matrix * weights[None, :], axis=1, dtype=np.float32)
        category_mask = categories == category
        fold_rows = []
        for fold in sorted(np.unique(folds)):
            train = category_mask & (folds != fold)
            valid = category_mask & (folds == fold)
            train_f1, threshold = best_threshold(labels[train], scores[train])
            predictions = (scores[valid] >= threshold).astype(np.int8)
            nested[valid] = predictions
            fold_rows.append(
                {
                    "fold": int(fold),
                    "threshold": threshold,
                    "train_f1": train_f1,
                    "validation_f1": f1(labels[valid], predictions),
                }
            )
        production[category_mask] = (
            scores[category_mask] >= config["production_threshold"]
        ).astype(np.int8)
        report[category] = {
            "weights": {
                "robust_base": float(weights[0]),
                "qwen3vl": float(weights[1]),
                "qwen35": float(weights[2]),
            },
            "nested_f1": f1(labels[category_mask], nested[category_mask]),
            "nested_folds": fold_rows,
            "production_threshold": config["production_threshold"],
            "production_threshold_oof_f1": f1(
                labels[category_mask], production[category_mask]
            ),
        }
    return nested, production, report


def paired_audit(
    *,
    name: str,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    group_hashes: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict:
    baseline_category = category_scores(labels, baseline, categories)
    candidate_category = category_scores(labels, candidate, categories)
    baseline_macro = float(np.mean(list(baseline_category.values())))
    candidate_macro = float(np.mean(list(candidate_category.values())))
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        old = category_scores(labels, baseline, categories, mask)
        new = category_scores(labels, candidate, categories, mask)
        old_macro = float(np.mean(list(old.values())))
        new_macro = float(np.mean(list(new.values())))
        fold_rows.append(
            {
                "fold": int(fold),
                "baseline_macro_f1": old_macro,
                "candidate_macro_f1": new_macro,
                "delta": new_macro - old_macro,
                "baseline_category_f1": old,
                "candidate_category_f1": new,
            }
        )

    group_rows = {}
    for category in sorted(np.unique(categories)):
        positions = np.flatnonzero(categories == category)
        groups = {}
        for position in positions:
            groups.setdefault(group_hashes[position], []).append(position)
        group_rows[category] = [np.asarray(rows) for rows in groups.values()]
    rng = np.random.default_rng(seed)
    deltas = np.empty(bootstrap, dtype=np.float64)
    for iteration in range(bootstrap):
        old_values, new_values = [], []
        for category in sorted(group_rows):
            groups = group_rows[category]
            sampled = rng.integers(0, len(groups), size=len(groups))
            positions = np.concatenate([groups[index] for index in sampled])
            old_values.append(f1(labels[positions], baseline[positions]))
            new_values.append(f1(labels[positions], candidate[positions]))
        deltas[iteration] = np.mean(new_values) - np.mean(old_values)

    category_delta = {
        category: candidate_category[category] - baseline_category[category]
        for category in baseline_category
    }
    wins = sum(row["delta"] > 0 for row in fold_rows)
    return {
        "candidate": name,
        "baseline_macro_f1": baseline_macro,
        "candidate_macro_f1": candidate_macro,
        "delta_macro_f1": candidate_macro - baseline_macro,
        "baseline_category_f1": baseline_category,
        "candidate_category_f1": candidate_category,
        "category_delta": category_delta,
        "folds_won": wins,
        "folds": fold_rows,
        "changed_predictions": int((baseline != candidate).sum()),
        "corrected": int(((baseline != labels) & (candidate == labels)).sum()),
        "regressed": int(((baseline == labels) & (candidate != labels)).sum()),
        "group_bootstrap": {
            "iterations": bootstrap,
            "seed": seed,
            "delta_mean": float(deltas.mean()),
            "delta_ci95": [
                float(np.quantile(deltas, 0.025)),
                float(np.quantile(deltas, 0.975)),
            ],
            "probability_delta_positive": float((deltas > 0).mean()),
        },
        "acceptance": {
            "delta_at_least_0_001": candidate_macro - baseline_macro >= 0.001,
            "wins_at_least_4_of_5_folds": wins >= 4,
            "no_category_drop_over_0_005": min(category_delta.values()) >= -0.005,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=6,
        default=[],
        metavar=("NAME", "FOLD0", "FOLD1", "FOLD2", "FOLD3", "FOLD4"),
    )
    parser.add_argument(
        "--candidate-rank-npz",
        action="append",
        nargs=3,
        default=[],
        metavar=("NAME", "NPZ", "KEY"),
        help="Evaluate an already aligned fold/category rank vector, e.g. Gemma lora_rank",
    )
    parser.add_argument("--folds", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=19042)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = np.load(BASE, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    original = np.load(QWEN35_SEED_A, allow_pickle=True)
    ids = base["ids"].astype(str)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    folds = base["fold_ids"].astype(np.int8)
    for source_name, source in (("qwen3vl", qwen3vl), ("qwen35", original)):
        if not np.array_equal(ids, source["ids"].astype(str)):
            raise ValueError(f"{source_name} id mismatch")
        if not np.array_equal(folds, source["folds"].astype(np.int8)):
            raise ValueError(f"{source_name} fold mismatch")

    fold_frame = pd.read_csv(args.folds, dtype={"id": str}).set_index("id").loc[ids]
    if not np.array_equal(fold_frame["fold"].to_numpy(np.int8), folds):
        raise ValueError("fold registry mismatch")
    group_hashes = fold_frame["group_hash"].astype(str).to_numpy()
    base_rank = original["base_rank"].astype(np.float32)
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)
    original_rank = original["lora_rank"].astype(np.float32)
    baseline_nested, baseline_production, baseline_detail = evaluate_locked(
        np.column_stack([base_rank, qwen3vl_rank, original_rank]),
        labels,
        categories,
        folds,
    )

    candidates = {}
    arrays = {
        "ids": ids,
        "folds": folds,
        "labels": labels,
        "categories": categories,
        "baseline_nested_predictions": baseline_nested,
        "baseline_production_predictions": baseline_production,
    }
    rank_candidates = []
    for specification in args.candidate:
        name, *paths = specification
        logits = load_seed_predictions(paths, base)
        rank_candidates.append((name, fold_category_ranks(logits, folds, categories)))
    for name, path, key in args.candidate_rank_npz:
        source = np.load(path, allow_pickle=True)
        for field, expected in (("ids", ids), ("folds", folds), ("labels", labels), ("categories", categories)):
            if not np.array_equal(source[field].astype(expected.dtype), expected):
                raise ValueError(f"{name} {field} mismatch")
        rank_candidates.append((name, source[key].astype(np.float32)))
    if not rank_candidates:
        raise ValueError("at least one --candidate or --candidate-rank-npz is required")
    if len({name for name, _ in rank_candidates}) != len(rank_candidates):
        raise ValueError("candidate names must be unique")

    for offset, (name, ranks) in enumerate(rank_candidates):
        nested, production, detail = evaluate_locked(
            np.column_stack([base_rank, qwen3vl_rank, ranks]),
            labels,
            categories,
            folds,
        )
        audit = paired_audit(
            name=name,
            labels=labels,
            categories=categories,
            folds=folds,
            group_hashes=group_hashes,
            baseline=baseline_nested,
            candidate=nested,
            bootstrap=args.bootstrap,
            seed=args.seed + offset,
        )
        audit["accepted"] = all(audit["acceptance"].values())
        audit["locked_recipe_detail"] = detail
        audit["production_threshold_diagnostic"] = paired_audit(
            name=name,
            labels=labels,
            categories=categories,
            folds=folds,
            group_hashes=group_hashes,
            baseline=baseline_production,
            candidate=production,
            bootstrap=args.bootstrap,
            seed=args.seed + 100 + offset,
        )
        candidates[name] = audit
        arrays[f"{name}_nested_predictions"] = nested
        arrays[f"{name}_production_predictions"] = production

    result = {
        "evaluation_version": "locked_190_nested_v1",
        "purpose": "Replace only the Qwen3.5 seed42 adapter inside the fixed Public-190 architecture",
        "weights_locked_before_candidate_training": True,
        "nested_thresholds_fit_on_other_four_outer_folds": True,
        "production_threshold_diagnostic_is_optimistic_for_baseline": True,
        "baseline": baseline_detail,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("research")
BASE = Path(os.environ.get("ECUP_BASE_OOF", ROOT / "oof-cache-extracted/oof_scores.npz"))
QWEN3VL = Path(os.environ.get("ECUP_QWEN3VL_OOF", ROOT / "lora-hard-5fold-robust-fusion-report.npz"))
QWEN35_SEED_A = Path(os.environ.get("ECUP_QWEN35_SEED_A_OOF", ROOT / "qwen35-hard-5fold-robust-fusion-report.npz"))
REPORT = Path(os.environ.get("ECUP_REPORT", ROOT / "qwen35-seed31415-ensemble-report.json"))


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def best_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered = labels[order]
    tp = np.cumsum(ordered == 1)
    fp = np.cumsum(ordered == 0)
    fn = int((labels == 1).sum()) - tp
    values = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    best = int(np.argmax(values))
    if best + 1 < len(scores):
        threshold = float((scores[order[best]] + scores[order[best + 1]]) / 2)
    else:
        threshold = float(scores[order[best]] - 1e-7)
    return float(values[best]), threshold


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def fold_category_ranks(
    values: np.ndarray, folds: np.ndarray, categories: np.ndarray
) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float32)
    for fold in sorted(np.unique(folds)):
        for category in sorted(np.unique(categories)):
            positions = np.flatnonzero(
                (folds == fold) & (categories == category)
            )
            result[positions] = rank01(values[positions])
    return result


def load_seed_predictions(
    paths: list[str], base: np.lib.npyio.NpzFile
) -> np.ndarray:
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["id"] = frame.id.astype(str)
    if frame.id.duplicated().any():
        raise ValueError("duplicate seed-B ids")
    ids = base["ids"].astype(str)
    by_id = frame.set_index("id")
    missing = sorted(set(ids) - set(by_id.index))
    if missing:
        raise ValueError(f"missing {len(missing)} ids, examples={missing[:5]}")
    aligned = by_id.loc[ids]
    folds = base["fold_ids"].astype(np.int8)
    if not np.array_equal(folds, aligned.fold.to_numpy(dtype=np.int8)):
        raise ValueError("fold mismatch")
    if not np.array_equal(
        base["labels"].astype(np.int8), aligned.label.to_numpy(dtype=np.int8)
    ):
        raise ValueError("label mismatch")
    return aligned.lora_score.to_numpy(dtype=np.float32)


def simplex_weights(model_count: int, units: int):
    for separators in itertools.combinations(
        range(units + model_count - 1), model_count - 1
    ):
        boundaries = (-1, *separators, units + model_count - 1)
        counts = [
            boundaries[index + 1] - boundaries[index] - 1
            for index in range(model_count)
        ]
        yield np.asarray(counts, dtype=np.float32) / units


def weighted_scores(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(matrix * weights[None, :], axis=1, dtype=np.float32)


def evaluate_matrix(
    names: list[str],
    score_matrix: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    *,
    step: float = 0.05,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    score_matrix = np.asarray(score_matrix, dtype=np.float32)
    units = int(round(1.0 / step))
    weights = list(simplex_weights(score_matrix.shape[1], units))
    nested_predictions = np.zeros(len(labels), dtype=np.int8)
    full_predictions = np.zeros(len(labels), dtype=np.int8)
    report: dict[str, object] = {
        "models": names,
        "step": step,
        "weight_candidates": len(weights),
        "categories": {},
    }
    category_values = []
    for category in sorted(np.unique(categories)):
        category_mask = categories == category
        fold_rows = []
        for fold in sorted(np.unique(folds)):
            train = category_mask & (folds != fold)
            valid = category_mask & (folds == fold)
            best = None
            for values in weights:
                train_scores = weighted_scores(score_matrix[train], values)
                train_f1, threshold = best_threshold(labels[train], train_scores)
                seed_balance = (
                    -abs(float(values[-2] - values[-1]))
                    if len(names) >= 4
                    else 0.0
                )
                preference = (
                    train_f1,
                    float(values[0]),
                    seed_balance,
                    float(np.sum(values == 0)),
                )
                if best is None or preference > best[0]:
                    best = (preference, threshold, values)
            threshold, selected = best[1], best[2]
            valid_scores = weighted_scores(score_matrix[valid], selected)
            predictions = (valid_scores >= threshold).astype(np.int8)
            nested_predictions[valid] = predictions
            fold_rows.append(
                {
                    "fold": int(fold),
                    "weights": {
                        name: float(value)
                        for name, value in zip(names, selected)
                    },
                    "threshold": float(threshold),
                    "validation_f1": f1(labels[valid], predictions),
                }
            )
        nested_f1 = f1(
            labels[category_mask], nested_predictions[category_mask]
        )
        full_best = None
        for values in weights:
            scores = weighted_scores(score_matrix[category_mask], values)
            value, threshold = best_threshold(labels[category_mask], scores)
            fold_f1 = {
                str(int(fold)): f1(
                    labels[category_mask & (folds == fold)],
                    (
                        weighted_scores(
                            score_matrix[category_mask & (folds == fold)], values
                        )
                        >= threshold
                    ).astype(np.int8),
                )
                for fold in sorted(np.unique(folds))
            }
            seed_balance = (
                -abs(float(values[-2] - values[-1]))
                if len(names) >= 4
                else 0.0
            )
            preference = (
                value,
                -float(np.std(list(fold_f1.values()))),
                float(values[0]),
                seed_balance,
            )
            if full_best is None or preference > full_best[0]:
                full_best = (preference, threshold, values, fold_f1)
        threshold, selected, fold_f1 = full_best[1:]
        scores = weighted_scores(score_matrix[category_mask], selected)
        full_predictions[category_mask] = (scores >= threshold).astype(np.int8)
        report["categories"][category] = {
            "nested_f1": nested_f1,
            "folds": fold_rows,
            "full_oof": {
                "f1": float(full_best[0][0]),
                "weights": {
                    name: float(value)
                    for name, value in zip(names, selected)
                },
                "threshold": float(threshold),
                "fold_f1": fold_f1,
                "fold_f1_std": float(np.std(list(fold_f1.values()))),
            },
        }
        category_values.append(nested_f1)
    report["nested_macro_f1"] = float(np.mean(category_values))
    return report, nested_predictions, full_predictions


def single_model_report(
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    ranks: dict[str, np.ndarray],
) -> dict[str, object]:
    report: dict[str, object] = {}
    for category in sorted(np.unique(categories)):
        positions = np.flatnonzero(categories == category)
        report[category] = {}
        for name, scores in ranks.items():
            value, threshold = best_threshold(labels[positions], scores[positions])
            fold_f1 = {
                str(int(fold)): f1(
                    labels[positions][folds[positions] == fold],
                    (scores[positions][folds[positions] == fold] >= threshold).astype(
                        np.int8
                    ),
                )
                for fold in sorted(np.unique(folds))
            }
            report[category][name] = {
                "f1": value,
                "threshold": threshold,
                "fold_f1": fold_f1,
                "fold_f1_std": float(np.std(list(fold_f1.values()))),
            }
        report[category]["seed_rank_correlation"] = float(
            np.corrcoef(ranks["seed42"][positions], ranks["seed31415"][positions])[0, 1]
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-b-predictions", nargs="+", required=True)
    parser.add_argument("--output", default=str(REPORT))
    args = parser.parse_args()

    base = np.load(BASE, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    seed_a = np.load(QWEN35_SEED_A, allow_pickle=True)
    ids = base["ids"].astype(str)
    folds = base["fold_ids"].astype(np.int8)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    for name, source in [("qwen3vl", qwen3vl), ("seed42", seed_a)]:
        if not np.array_equal(ids, source["ids"].astype(str)):
            raise ValueError(f"{name} id mismatch")
        if not np.array_equal(folds, source["folds"].astype(np.int8)):
            raise ValueError(f"{name} fold mismatch")
    seed_b_logits = load_seed_predictions(args.seed_b_predictions, base)
    seed_b_rank = fold_category_ranks(seed_b_logits, folds, categories)
    seed_a_rank = seed_a["lora_rank"].astype(np.float32)
    average_rank = (seed_a_rank + seed_b_rank) / 2.0
    base_rank = seed_a["base_rank"].astype(np.float32)
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)

    single = single_model_report(
        labels,
        categories,
        folds,
        {
            "seed42": seed_a_rank,
            "seed31415": seed_b_rank,
            "mean_rank": average_rank,
        },
    )
    mean_report, mean_nested, mean_full = evaluate_matrix(
        ["robust_base", "qwen3vl", "qwen35_mean_seed"],
        np.column_stack([base_rank, qwen3vl_rank, average_rank]),
        labels,
        categories,
        folds,
    )
    four_report, four_nested, four_full = evaluate_matrix(
        ["robust_base", "qwen3vl", "qwen35_seed42", "qwen35_seed31415"],
        np.column_stack([base_rank, qwen3vl_rank, seed_a_rank, seed_b_rank]),
        labels,
        categories,
        folds,
    )
    report = {
        "rows": int(len(ids)),
        "seed_a": 42,
        "seed_b": 31415,
        "single_models": single,
        "mean_seed_three_head": mean_report,
        "separate_seed_four_head": four_report,
    }
    output = Path(args.output)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output.with_suffix(".npz"),
        ids=ids,
        folds=folds,
        labels=labels,
        categories=categories,
        qwen35_seed42_rank=seed_a_rank,
        qwen35_seed31415_rank=seed_b_rank,
        qwen35_mean_rank=average_rank,
        nested_predictions=mean_nested,
        full_oof_predictions=mean_full,
        mean_nested_predictions=mean_nested,
        mean_full_predictions=mean_full,
        four_nested_predictions=four_nested,
        four_full_predictions=four_full,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

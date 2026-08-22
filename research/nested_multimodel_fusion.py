from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered_labels = labels[order]
    tp = np.cumsum(ordered_labels == 1)
    fp = np.cumsum(ordered_labels == 0)
    fn = int((labels == 1).sum()) - tp
    values = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    best = int(np.argmax(values))
    if best + 1 < len(scores):
        threshold = float((scores[order[best]] + scores[order[best + 1]]) / 2)
    else:
        threshold = float(scores[order[best]] - 1e-7)
    return float(values[best]), threshold


def simplex_weights(model_count: int, units: int):
    for separators in itertools.combinations(range(units + model_count - 1), model_count - 1):
        boundaries = (-1, *separators, units + model_count - 1)
        counts = [boundaries[index + 1] - boundaries[index] - 1 for index in range(model_count)]
        yield np.asarray(counts, dtype=np.float32) / units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="aggregate_lora_oof NPZ files")
    parser.add_argument("--names", nargs="+", required=True, help="LoRA model names, one per input")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument(
        "--max-weights",
        nargs="+",
        type=float,
        help="Optional caps for robust_base followed by each named model",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.inputs) != len(args.names):
        raise ValueError("--inputs and --names must have equal lengths")
    units = int(round(1.0 / args.step))
    if units <= 0 or not np.isclose(units * args.step, 1.0):
        raise ValueError("--step must evenly divide 1.0")

    sources = [np.load(path, allow_pickle=True) for path in args.inputs]
    first = sources[0]
    ids = first["ids"].astype(str)
    folds = first["folds"].astype(np.int8)
    labels = first["labels"].astype(np.int8)
    categories = first["categories"].astype(str)
    for path, source in zip(args.inputs[1:], sources[1:]):
        for key, expected in (("ids", ids), ("folds", folds), ("labels", labels), ("categories", categories)):
            actual = source[key].astype(expected.dtype)
            if not np.array_equal(actual, expected):
                raise ValueError(f"alignment mismatch for {key} in {path}")

    names = ["robust_base", *args.names]
    score_matrix = np.column_stack([
        first["base_rank"].astype(np.float32),
        *[source["lora_rank"].astype(np.float32) for source in sources],
    ])
    if args.max_weights is not None and len(args.max_weights) != score_matrix.shape[1]:
        raise ValueError("--max-weights must have one value per model including robust_base")
    weight_grid = [
        weights
        for weights in simplex_weights(score_matrix.shape[1], units)
        if args.max_weights is None
        or np.all(weights <= np.asarray(args.max_weights, dtype=np.float32) + 1e-7)
    ]
    if not weight_grid:
        raise ValueError("weight caps exclude every simplex candidate")
    nested_predictions = np.zeros(len(labels), dtype=np.int8)
    full_oof_predictions = np.zeros(len(labels), dtype=np.int8)
    report = {
        "rows": int(len(labels)),
        "models": names,
        "weight_step": args.step,
        "weight_candidates": len(weight_grid),
        "max_weights": args.max_weights,
        "categories": {},
    }
    macro = []
    for category in sorted(np.unique(categories)):
        category_mask = categories == category
        fold_rows = []
        for fold in sorted(np.unique(folds)):
            train = category_mask & (folds != fold)
            valid = category_mask & (folds == fold)
            best = None
            for weights in weight_grid:
                train_scores = score_matrix[train] @ weights
                train_f1, threshold = best_threshold(labels[train], train_scores)
                # On ties prefer the Public-proven base, then a sparse mixture.
                tie_break = (float(weights[0]), float(np.sum(weights == 0)))
                candidate = (train_f1, tie_break, threshold, weights)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            train_f1, _, threshold, weights = best
            valid_scores = score_matrix[valid] @ weights
            valid_predictions = (valid_scores >= threshold).astype(np.int8)
            nested_predictions[valid] = valid_predictions
            fold_rows.append({
                "fold": int(fold),
                "rows": int(valid.sum()),
                "positive": int(labels[valid].sum()),
                "weights": {name: float(weight) for name, weight in zip(names, weights)},
                "threshold": float(threshold),
                "train_f1": float(train_f1),
                "validation_f1": f1(labels[valid], valid_predictions),
            })
        category_f1 = f1(labels[category_mask], nested_predictions[category_mask])
        full_best = None
        for weights in weight_grid:
            full_scores = score_matrix[category_mask] @ weights
            full_f1, full_threshold = best_threshold(labels[category_mask], full_scores)
            fold_f1 = {
                str(int(fold)): f1(
                    labels[category_mask & (folds == fold)],
                    (score_matrix[category_mask & (folds == fold)] @ weights >= full_threshold).astype(np.int8),
                )
                for fold in sorted(np.unique(folds))
            }
            candidate = (
                full_f1,
                -float(np.std(list(fold_f1.values()))),
                float(weights[0]),
                weights,
                full_threshold,
                fold_f1,
            )
            if full_best is None or candidate[:3] > full_best[:3]:
                full_best = candidate
        report["categories"][category] = {
            "nested_f1": category_f1,
            "folds": fold_rows,
            "mean_fold_f1": float(np.mean([row["validation_f1"] for row in fold_rows])),
            "std_fold_f1": float(np.std([row["validation_f1"] for row in fold_rows])),
            "full_oof": {
                "f1": float(full_best[0]),
                "weights": {name: float(weight) for name, weight in zip(names, full_best[3])},
                "threshold": float(full_best[4]),
                "fold_f1": full_best[5],
                "fold_f1_std": float(np.std(list(full_best[5].values()))),
            },
        }
        category_scores = score_matrix[category_mask] @ full_best[3]
        full_oof_predictions[category_mask] = (category_scores >= full_best[4]).astype(np.int8)
        macro.append(category_f1)
    report["nested_macro_f1"] = float(np.mean(macro))
    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        output.with_suffix(".npz"),
        ids=ids,
        folds=folds,
        labels=labels,
        categories=categories,
        nested_predictions=nested_predictions,
        full_oof_predictions=full_oof_predictions,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

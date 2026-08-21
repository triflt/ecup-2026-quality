from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SOURCE = Path("research/lora-hard-5fold-robust-fusion-report.npz")
OUTPUT = Path("research/lora-hard-5fold-nested-calibration.json")


def f1(labels, predictions):
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def best_threshold(labels, scores):
    order = np.argsort(scores)[::-1]
    ordered_labels = labels[order]
    tp = np.cumsum(ordered_labels == 1)
    fp = np.cumsum(ordered_labels == 0)
    total_positive = int((labels == 1).sum())
    fn = total_positive - tp
    values = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    best = int(np.argmax(values))
    if best + 1 < len(scores):
        threshold = float((scores[order[best]] + scores[order[best + 1]]) / 2)
    else:
        threshold = float(scores[order[best]] - 1e-7)
    return float(values[best]), threshold


def main():
    data = np.load(SOURCE, allow_pickle=True)
    labels = data["labels"].astype(np.int8)
    categories = data["categories"].astype(str)
    folds = data["folds"].astype(np.int8)
    base = data["base_rank"].astype(np.float32)
    lora = data["lora_rank"].astype(np.float32)
    nested_predictions = np.zeros(len(labels), dtype=np.int8)
    report = {"categories": {}}
    macro = []
    for category in sorted(np.unique(categories)):
        category_mask = categories == category
        fold_rows = []
        for fold in sorted(np.unique(folds)):
            train = category_mask & (folds != fold)
            valid = category_mask & (folds == fold)
            best = None
            for weight_base in np.linspace(0.0, 1.0, 21):
                train_scores = weight_base * base[train] + (1 - weight_base) * lora[train]
                train_f1, threshold = best_threshold(labels[train], train_scores)
                # Prefer more base weight on exact ties because it is already Public-proven.
                candidate = (train_f1, weight_base, threshold)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            train_f1, weight_base, threshold = best
            valid_scores = weight_base * base[valid] + (1 - weight_base) * lora[valid]
            valid_predictions = (valid_scores >= threshold).astype(np.int8)
            nested_predictions[valid] = valid_predictions
            fold_rows.append({
                "fold": int(fold),
                "rows": int(valid.sum()),
                "positive": int(labels[valid].sum()),
                "weight_base": float(weight_base),
                "weight_lora": float(1 - weight_base),
                "threshold": float(threshold),
                "train_f1": float(train_f1),
                "validation_f1": f1(labels[valid], valid_predictions),
            })
        category_f1 = f1(labels[category_mask], nested_predictions[category_mask])
        report["categories"][category] = {
            "nested_f1": category_f1,
            "folds": fold_rows,
            "mean_fold_f1": float(np.mean([row["validation_f1"] for row in fold_rows])),
            "std_fold_f1": float(np.std([row["validation_f1"] for row in fold_rows])),
        }
        macro.append(category_f1)
    report["nested_macro_f1"] = float(np.mean(macro))
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

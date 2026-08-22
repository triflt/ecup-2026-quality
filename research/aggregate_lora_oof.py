from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def f1_score(labels, predictions):
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=bool)
    true_positive = int(np.sum((labels == 1) & predictions))
    false_positive = int(np.sum((labels == 0) & predictions))
    false_negative = int(np.sum((labels == 1) & ~predictions))
    return 2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)


def rank01(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def best_threshold(labels, scores):
    best = (-1.0, 0.0)
    candidates = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 1000)))
    for threshold in candidates:
        value = f1_score(labels, scores >= threshold)
        if value > best[0]:
            best = (float(value), float(threshold))
    return best


def fold_category_ranks(values, folds, categories):
    result = np.empty(len(values), dtype=np.float32)
    for fold in sorted(np.unique(folds)):
        for category in sorted(np.unique(categories)):
            positions = np.flatnonzero((folds == fold) & (categories == category))
            result[positions] = rank01(values[positions])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--base-oof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.predictions]
    predictions = pd.concat(frames, ignore_index=True)
    if predictions["id"].astype(str).duplicated().any():
        raise ValueError("duplicate holdout ids")
    base = np.load(args.base_oof, allow_pickle=True)
    base_ids = base["ids"].astype(str)
    by_id = predictions.set_index(predictions["id"].astype(str))
    missing = sorted(set(base_ids) - set(by_id.index))
    if missing:
        raise ValueError(f"missing {len(missing)} OOF ids, examples={missing[:5]}")
    predictions = by_id.loc[base_ids].reset_index(drop=True)
    folds = base["fold_ids"].astype(np.int8)
    if not np.array_equal(folds, predictions["fold"].to_numpy(dtype=np.int8)):
        raise ValueError("fold mismatch")
    labels = predictions["label"].to_numpy(dtype=np.int8)
    categories = predictions["category"].astype(str).to_numpy()
    lora = fold_category_ranks(
        predictions["lora_score"].to_numpy(dtype=np.float32), folds, categories
    )
    score_key = "fused_scores" if "fused_scores" in base.files else "fused"
    base_rank = fold_category_ranks(
        base[score_key].astype(np.float32), folds, categories
    )

    report = {"rows": len(predictions), "base_score_key": score_key, "categories": {}}
    macro_lora, macro_fused = [], []
    for category in sorted(np.unique(categories)):
        positions = np.flatnonzero(categories == category)
        local_labels = labels[positions]
        lora_f1, lora_threshold = best_threshold(local_labels, lora[positions])
        best = None
        for base_weight in np.linspace(0.0, 1.0, 21):
            fused = base_weight * base_rank[positions] + (1 - base_weight) * lora[positions]
            value, threshold = best_threshold(local_labels, fused)
            fold_f1 = {
                str(fold): float(f1_score(
                    local_labels[folds[positions] == fold],
                    fused[folds[positions] == fold] >= threshold,
                ))
                for fold in sorted(np.unique(folds))
            }
            item = {
                "f1": value,
                "threshold": threshold,
                "weight_base": float(base_weight),
                "weight_lora": float(1 - base_weight),
                "fold_f1": fold_f1,
                "fold_f1_std": float(np.std(list(fold_f1.values()))),
            }
            if best is None or (item["f1"], -item["fold_f1_std"]) > (best["f1"], -best["fold_f1_std"]):
                best = item
        report["categories"][category] = {
            "rows": len(positions),
            "positive": int(local_labels.sum()),
            "lora_f1": lora_f1,
            "lora_threshold": lora_threshold,
            "best_fusion": best,
        }
        macro_lora.append(lora_f1)
        macro_fused.append(best["f1"])
    report["macro_lora"] = float(np.mean(macro_lora))
    report["macro_fused"] = float(np.mean(macro_fused))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        output.with_suffix(".npz"),
        ids=base_ids,
        folds=folds,
        categories=categories,
        labels=labels,
        lora_rank=lora,
        base_rank=base_rank,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

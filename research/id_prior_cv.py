from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path("research/data.csv")
OOF = Path("research/oof-cache-extracted/oof_scores.npz")
OUTPUT = Path("research/id-prior-cv.json")


def f1(labels, predictions):
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def best_threshold(labels, scores):
    best = (-1.0, 0.0)
    for threshold in np.unique(np.quantile(scores, np.linspace(0.002, 0.998, 700))):
        score = f1(labels, scores >= threshold)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best


def rank01(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def nearest_prior(train_ids, train_labels, valid_ids, k):
    order = np.argsort(train_ids)
    train_ids = train_ids[order]
    train_labels = train_labels[order]
    result = np.empty(len(valid_ids), dtype=np.float32)
    half_window = max(k * 2, 32)
    for index, value in enumerate(valid_ids):
        position = np.searchsorted(train_ids, value)
        left = max(0, position - half_window)
        right = min(len(train_ids), position + half_window)
        candidate_ids = train_ids[left:right]
        candidate_labels = train_labels[left:right]
        nearest = np.argsort(np.abs(candidate_ids - value), kind="mergesort")[:k]
        result[index] = candidate_labels[nearest].mean()
    return result


def main():
    frame = pd.read_csv(DATA, usecols=["id", "category", "label"])
    archive = np.load(OOF, allow_pickle=True)
    ids = frame["id"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame["id"].astype(str).to_numpy(), archive["ids"].astype(str)):
        raise ValueError("OOF order mismatch")
    categories = frame["category"].astype(str).to_numpy()
    labels_all = frame["label"].to_numpy(dtype=np.int8)
    folds_all = archive["fold_ids"].astype(np.int8)
    baseline_all = archive["fused_scores"].astype(np.float32)
    report = {"validation": "same grouped folds as multimodal late fusion", "categories": {}}
    macro = {}
    for category in sorted(frame["category"].unique()):
        positions = np.flatnonzero(categories == category)
        local_ids = ids[positions]
        labels = labels_all[positions]
        folds = folds_all[positions]
        baseline = baseline_all[positions]
        base_f1, base_threshold = best_threshold(labels, baseline)
        entry = {"baseline": {"f1": base_f1, "threshold": base_threshold}, "priors": {}}
        for k in (5, 10, 20, 40, 80, 160):
            prior = np.empty(len(positions), dtype=np.float32)
            for fold in range(5):
                train = folds != fold
                valid = folds == fold
                prior[valid] = nearest_prior(
                    local_ids[train], labels[train], local_ids[valid], k
                )
            prior_f1, prior_threshold = best_threshold(labels, prior)
            base_rank = rank01(baseline)
            prior_rank = rank01(prior)
            best_fusion = None
            for weight in np.linspace(0.0, 0.30, 16):
                scores = (1 - weight) * base_rank + weight * prior_rank
                score, threshold = best_threshold(labels, scores)
                if best_fusion is None or score > best_fusion["f1"]:
                    best_fusion = {
                        "f1": score,
                        "threshold": threshold,
                        "id_weight": float(weight),
                    }
            entry["priors"][f"knn_{k}"] = {
                "f1": prior_f1,
                "threshold": prior_threshold,
                "fusion": best_fusion,
            }
        report["categories"][category] = entry
        macro.setdefault("baseline", []).append(base_f1)
        for name, value in entry["priors"].items():
            macro.setdefault(name, []).append(value["f1"])
            macro.setdefault(name + "+baseline", []).append(value["fusion"]["f1"])
    report["macro_f1"] = {name: float(np.mean(values)) for name, values in macro.items()}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


OOF = Path(os.environ.get("ECUP_OOF", "/work/input/oof_scores.npz"))
ALL_IMAGES = Path(os.environ.get("ECUP_ALL_IMAGE_EMBEDDINGS", "/work/input/all/train_embeddings_fp16.npz"))
FIRST_IMAGE = Path(os.environ.get("ECUP_FIRST_IMAGE_EMBEDDINGS", "/work/input/first/train_embeddings_fp16.npz"))
OUTPUT = Path(os.environ.get("ECUP_REPORT", "/work/output/four_head_fusion_report.json"))


def best_threshold(labels, scores):
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    fn = int(sorted_labels.sum()) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    # A >= threshold classifier includes every tied score, so only evaluate
    # the final position of each tied block.
    boundary = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    candidates = np.flatnonzero(boundary)
    best_index = int(candidates[np.argmax(f1[candidates])])
    return float(f1[best_index]), float(sorted_scores[best_index])


def rank01(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def oof_first(features, labels, fold_ids):
    scores = np.empty(len(labels), dtype=np.float32)
    for fold in range(5):
        train = fold_ids != fold
        valid = fold_ids == fold
        model = LinearSVC(
            C=10.0, class_weight="balanced", dual="auto",
            max_iter=10000, random_state=42 + fold,
        )
        model.fit(features[train], labels[train])
        scores[valid] = model.decision_function(features[valid])
    return scores


def oof_trees(features, labels, fold_ids):
    scores = np.empty(len(labels), dtype=np.float32)
    for fold in range(5):
        train = fold_ids != fold
        valid = fold_ids == fold
        model = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=8,
            random_state=42 + fold,
        )
        model.fit(features[train], labels[train])
        scores[valid] = model.predict_proba(features[valid])[:, 1]
    return scores


def search(labels, heads):
    best = None
    # 1,771 simplex points at 0.05 resolution. Exact threshold search is
    # vectorized, avoiding millions of sklearn metric calls.
    for text_steps in range(21):
        for all_steps in range(21 - text_steps):
            for first_steps in range(21 - text_steps - all_steps):
                trees_steps = 20 - text_steps - all_steps - first_steps
                weights = np.array(
                    [text_steps, all_steps, first_steps, trees_steps], dtype=float
                ) / 20
                scores = sum(weight * head for weight, head in zip(weights, heads))
                score, threshold = best_threshold(labels, scores)
                if best is None or score > best["f1"]:
                    best = {
                        "f1": score,
                        "threshold": threshold,
                        "weight_text": float(weights[0]),
                        "weight_all_images": float(weights[1]),
                        "weight_first_image": float(weights[2]),
                        "weight_extra_trees": float(weights[3]),
                    }
    return best


def main():
    started = time.monotonic()
    oof = np.load(OOF, allow_pickle=True)
    all_images = np.load(ALL_IMAGES, allow_pickle=True)
    first_image = np.load(FIRST_IMAGE, allow_pickle=True)
    ids = oof["ids"].astype(str)
    for name, archive in (("all", all_images), ("first", first_image)):
        if not np.array_equal(ids, archive["ids"].astype(str)):
            raise ValueError(f"{name} id mismatch")
    categories = oof["categories"].astype(str)
    labels_all = oof["labels"].astype(np.int8)
    folds_all = oof["fold_ids"].astype(np.int8)
    all_features = all_images["embeddings"].astype(np.float32)
    first_features = first_image["embeddings"].astype(np.float32)
    report = {"validation": "same 5 grouped folds; four-head rank simplex", "categories": {}}
    values = []
    score_cache = {
        name: np.full(len(ids), np.nan, dtype=np.float32)
        for name in ("text", "all_images", "first_image", "extra_trees", "fused")
    }
    predictions = np.zeros(len(ids), dtype=np.int8)
    for category in sorted(np.unique(categories)):
        positions = np.flatnonzero(categories == category)
        labels = labels_all[positions]
        folds = folds_all[positions]
        first_scores = oof_first(first_features[positions], labels, folds)
        tree_scores = oof_trees(all_features[positions], labels, folds)
        heads = [
            rank01(oof["text_scores"][positions].astype(np.float32)),
            rank01(oof["mm_scores"][positions].astype(np.float32)),
            rank01(first_scores),
            rank01(tree_scores),
        ]
        result = search(labels, heads)
        fused = (
            result["weight_text"] * heads[0]
            + result["weight_all_images"] * heads[1]
            + result["weight_first_image"] * heads[2]
            + result["weight_extra_trees"] * heads[3]
        )
        score_cache["text"][positions] = heads[0]
        score_cache["all_images"][positions] = heads[1]
        score_cache["first_image"][positions] = heads[2]
        score_cache["extra_trees"][positions] = heads[3]
        score_cache["fused"][positions] = fused
        predictions[positions] = (fused >= result["threshold"]).astype(np.int8)
        report["categories"][category] = result
        values.append(result["f1"])
        print(category, json.dumps(result, ensure_ascii=False), flush=True)
    report["macro_f1"] = float(np.mean(values))
    report["runtime_minutes"] = (time.monotonic() - started) / 60
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUTPUT.parent / "four_head_oof.npz",
        ids=ids,
        labels=labels_all,
        categories=categories,
        fold_ids=folds_all,
        predictions=predictions,
        **score_cache,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

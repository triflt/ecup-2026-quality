from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


OOF = Path(os.environ.get("ECUP_OOF", "/work/input/oof_scores.npz"))
ALL_IMAGES = Path(os.environ.get("ECUP_ALL_IMAGE_EMBEDDINGS", "/work/input/all/train_embeddings_fp16.npz"))
FIRST_IMAGE = Path(os.environ.get("ECUP_FIRST_IMAGE_EMBEDDINGS", "/work/input/first/train_embeddings_fp16.npz"))
OUTPUT = Path(os.environ.get("ECUP_REPORT", "/work/output/tri_fusion_report.json"))
C_VALUES = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def best_threshold(labels, scores):
    best = (-1.0, 0.0)
    for threshold in np.unique(np.quantile(scores, np.linspace(0.002, 0.998, 700))):
        score = f1_score(labels, scores >= threshold)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best


def rank01(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def oof_svc(features, labels, fold_ids):
    best = None
    for c_value in C_VALUES:
        scores = np.empty(len(labels), dtype=np.float32)
        for fold in range(5):
            train = fold_ids != fold
            valid = fold_ids == fold
            model = LinearSVC(
                C=c_value,
                class_weight="balanced",
                dual="auto",
                max_iter=8000,
                random_state=42 + fold,
            )
            model.fit(features[train], labels[train])
            scores[valid] = model.decision_function(features[valid])
        score, threshold = best_threshold(labels, scores)
        if best is None or score > best["f1"]:
            best = {
                "f1": score,
                "threshold": threshold,
                "C": c_value,
                "scores": scores.copy(),
            }
    return best


def best_pair(labels, left_rank, right_rank):
    best = None
    for left_weight in np.linspace(0.0, 1.0, 21):
        scores = left_weight * left_rank + (1 - left_weight) * right_rank
        score, threshold = best_threshold(labels, scores)
        if best is None or score > best["f1"]:
            best = {
                "f1": score,
                "threshold": threshold,
                "left_weight": float(left_weight),
            }
    return best


def best_triple(labels, text_rank, all_rank, first_rank):
    best = None
    for text_steps in range(21):
        for all_steps in range(21 - text_steps):
            first_steps = 20 - text_steps - all_steps
            weights = np.array([text_steps, all_steps, first_steps], dtype=float) / 20
            scores = weights[0] * text_rank + weights[1] * all_rank + weights[2] * first_rank
            score, threshold = best_threshold(labels, scores)
            if best is None or score > best["f1"]:
                best = {
                    "f1": score,
                    "threshold": threshold,
                    "weight_text": float(weights[0]),
                    "weight_all_images": float(weights[1]),
                    "weight_first_image": float(weights[2]),
                }
    return best


def main():
    started = time.monotonic()
    oof = np.load(OOF, allow_pickle=True)
    all_images = np.load(ALL_IMAGES, allow_pickle=True)
    first_image = np.load(FIRST_IMAGE, allow_pickle=True)
    ids = oof["ids"].astype(str)
    if not np.array_equal(ids, all_images["ids"].astype(str)):
        raise ValueError("all-images id mismatch")
    if not np.array_equal(ids, first_image["ids"].astype(str)):
        raise ValueError("first-image id mismatch")
    categories = oof["categories"].astype(str)
    labels_all = oof["labels"].astype(np.int8)
    fold_ids_all = oof["fold_ids"].astype(np.int8)
    first_features_all = first_image["embeddings"].astype(np.float32)
    report = {"validation": "same grouped folds for all modalities", "categories": {}}
    macro = {"first_image": [], "text_first": [], "all_first": [], "tri_fusion": []}
    for category in sorted(np.unique(categories)):
        positions = np.flatnonzero(categories == category)
        labels = labels_all[positions]
        first = oof_svc(first_features_all[positions], labels, fold_ids_all[positions])
        text_rank = rank01(oof["text_scores"][positions].astype(np.float32))
        all_rank = rank01(oof["mm_scores"][positions].astype(np.float32))
        first_rank = rank01(first["scores"])
        text_first = best_pair(labels, text_rank, first_rank)
        all_first = best_pair(labels, all_rank, first_rank)
        triple = best_triple(labels, text_rank, all_rank, first_rank)
        entry = {
            "first_image": {key: value for key, value in first.items() if key != "scores"},
            "text_first": text_first,
            "all_first": all_first,
            "tri_fusion": triple,
        }
        report["categories"][category] = entry
        for key in macro:
            macro[key].append(entry[key]["f1"])
        print(category, json.dumps(entry, ensure_ascii=False), flush=True)
    report["macro_f1"] = {key: float(np.mean(values)) for key, values in macro.items()}
    report["runtime_minutes"] = (time.monotonic() - started) / 60
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["macro_f1"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

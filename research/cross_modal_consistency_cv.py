from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


OOF = Path(os.environ.get("ECUP_OOF", "/work/input/oof_scores.npz"))
ALL_IMAGES = Path(os.environ.get("ECUP_ALL_IMAGE_EMBEDDINGS", "/work/input/all/train_embeddings_fp16.npz"))
FIRST_IMAGE = Path(os.environ.get("ECUP_FIRST_IMAGE_EMBEDDINGS", "/work/input/first/train_embeddings_fp16.npz"))
TEXT_ONLY = Path(os.environ.get("ECUP_TEXT_EMBEDDINGS", "/work/input/text/train_embeddings_fp16.npz"))
OUTPUT = Path(os.environ.get("ECUP_REPORT", "/work/output/cross_modal_consistency_report.json"))
C_VALUES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best = (-1.0, 0.0)
    candidates = np.unique(np.quantile(scores, np.linspace(0.002, 0.998, 700)))
    for threshold in candidates:
        score = f1_score(labels, scores >= threshold)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def normalized(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32, copy=False)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)


def oof_linear(features, labels, fold_ids, c_values=C_VALUES):
    best = None
    for c_value in c_values:
        scores = np.empty(len(labels), dtype=np.float32)
        for fold in range(5):
            train = fold_ids != fold
            valid = fold_ids == fold
            model = LinearSVC(
                C=c_value,
                class_weight="balanced",
                dual="auto",
                max_iter=10000,
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


def oof_scalar(features, labels, fold_ids):
    scores = np.empty(len(labels), dtype=np.float32)
    for fold in range(5):
        train = fold_ids != fold
        valid = fold_ids == fold
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=42 + fold,
        )
        model.fit(features[train], labels[train])
        scores[valid] = model.decision_function(features[valid])
    score, threshold = best_threshold(labels, scores)
    return {"f1": score, "threshold": threshold, "scores": scores}


def best_fusion(labels, named_scores):
    ranks = {name: rank01(scores) for name, scores in named_scores.items()}
    baseline = ranks["text_tfidf"]
    candidates = []
    for other_name in ("text_qwen", "consistency", "first_image"):
        other = ranks[other_name]
        for text_steps in range(11):
            for other_steps in range(11 - text_steps):
                all_steps = 10 - text_steps - other_steps
                weights = np.array([text_steps, all_steps, other_steps], dtype=float) / 10
                fused = (
                    weights[0] * baseline
                    + weights[1] * ranks["all_images"]
                    + weights[2] * other
                )
                f1, threshold = best_threshold(labels, fused)
                candidates.append({
                    "f1": f1,
                    "threshold": threshold,
                    "weight_text_tfidf": float(weights[0]),
                    "weight_all_images": float(weights[1]),
                    f"weight_{other_name}": float(weights[2]),
                    "third_head": other_name,
                })
    return max(candidates, key=lambda item: item["f1"])


def main():
    started = time.monotonic()
    oof = np.load(OOF, allow_pickle=True)
    archives = {
        "all": np.load(ALL_IMAGES, allow_pickle=True),
        "first": np.load(FIRST_IMAGE, allow_pickle=True),
        "text": np.load(TEXT_ONLY, allow_pickle=True),
    }
    ids = oof["ids"].astype(str)
    for name, archive in archives.items():
        if not np.array_equal(ids, archive["ids"].astype(str)):
            raise ValueError(f"{name} id mismatch")

    text_features = normalized(archives["text"]["embeddings"])
    image_features = normalized(archives["first"]["embeddings"])
    categories = oof["categories"].astype(str)
    labels_all = oof["labels"].astype(np.int8)
    folds_all = oof["fold_ids"].astype(np.int8)

    cosine = np.sum(text_features * image_features, axis=1, keepdims=True)
    l1_mean = np.mean(np.abs(text_features - image_features), axis=1, keepdims=True)
    l2 = np.linalg.norm(text_features - image_features, axis=1, keepdims=True)
    dot_abs = np.mean(np.abs(text_features * image_features), axis=1, keepdims=True)
    scalar_features = np.concatenate([cosine, l1_mean, l2, dot_abs], axis=1)

    # This representation lets a linear head model both modalities and their
    # agreement without a large neural network or any label-dependent routing.
    interaction_features = np.concatenate([
        text_features,
        image_features,
        np.abs(text_features - image_features),
        text_features * image_features,
    ], axis=1).astype(np.float32, copy=False)

    report = {"validation": "same 5 grouped folds for every modality", "categories": {}}
    macro = {
        "text_qwen": [],
        "first_image": [],
        "scalar_consistency": [],
        "interaction": [],
        "best_fusion": [],
    }
    for category in sorted(np.unique(categories)):
        positions = np.flatnonzero(categories == category)
        labels = labels_all[positions]
        folds = folds_all[positions]
        text_result = oof_linear(text_features[positions], labels, folds)
        first_result = oof_linear(image_features[positions], labels, folds)
        scalar_result = oof_scalar(scalar_features[positions], labels, folds)
        interaction_result = oof_linear(
            interaction_features[positions], labels, folds, c_values=(0.003, 0.01, 0.03, 0.1, 0.3)
        )
        named_scores = {
            "text_tfidf": oof["text_scores"][positions].astype(np.float32),
            "all_images": oof["mm_scores"][positions].astype(np.float32),
            "first_image": first_result["scores"],
            "text_qwen": text_result["scores"],
            "consistency": interaction_result["scores"],
        }
        fusion = best_fusion(labels, named_scores)
        entry = {
            "text_qwen": {k: v for k, v in text_result.items() if k != "scores"},
            "first_image": {k: v for k, v in first_result.items() if k != "scores"},
            "scalar_consistency": {k: v for k, v in scalar_result.items() if k != "scores"},
            "interaction": {k: v for k, v in interaction_result.items() if k != "scores"},
            "best_fusion": fusion,
            "cosine_summary": {
                "mean": float(cosine[positions].mean()),
                "positive_mean": float(cosine[positions][labels == 1].mean()),
                "negative_mean": float(cosine[positions][labels == 0].mean()),
            },
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

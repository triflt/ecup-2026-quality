from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold


DATA = Path("/work/input/data.csv")
EMBEDDINGS = Path("/work/input/mm/train_embeddings_fp16.npz")
TEXT_OOF = Path("/work/input/oof_scores.npz")
OUTPUT = Path("/work/output/embedding_head_report.json")


def normalize(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", str(value).lower())
    value = re.sub(r"[^a-zа-яё0-9]+", " ", value)
    return " ".join(value.split())


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


def evaluate_estimator(factory, features, labels, folds, probability=False):
    scores = np.empty(len(labels), dtype=np.float32)
    for fold, (train_idx, valid_idx) in enumerate(folds):
        model = factory(fold)
        model.fit(features[train_idx], labels[train_idx])
        if probability:
            scores[valid_idx] = model.predict_proba(features[valid_idx])[:, 1]
        else:
            scores[valid_idx] = model.decision_function(features[valid_idx])
    f1, threshold = best_threshold(labels, scores)
    return {"f1": f1, "threshold": threshold, "scores": scores}


def centroid_scores(features, labels, folds):
    scores = np.empty(len(labels), dtype=np.float32)
    for train_idx, valid_idx in folds:
        positive = features[train_idx][labels[train_idx] == 1].mean(axis=0)
        negative = features[train_idx][labels[train_idx] == 0].mean(axis=0)
        direction = positive - negative
        direction /= np.linalg.norm(direction) + 1e-12
        scores[valid_idx] = features[valid_idx] @ direction
    f1, threshold = best_threshold(labels, scores)
    return {"f1": f1, "threshold": threshold, "scores": scores}


def main():
    started = time.monotonic()
    if not DATA.exists():
        urllib.request.urlretrieve(os.environ["DATA_URL"], DATA)
    frame = pd.read_csv(DATA)
    archive = np.load(EMBEDDINGS)
    text_archive = np.load(TEXT_OOF, allow_pickle=True)
    ids = frame["id"].astype(str).to_numpy()
    if not np.array_equal(ids, archive["ids"].astype(str)):
        raise ValueError("embedding id mismatch")
    if not np.array_equal(ids, text_archive["ids"].astype(str)):
        raise ValueError("text OOF id mismatch")
    features_all = archive["embeddings"].astype(np.float32)
    text_scores_all = text_archive["text_scores"].astype(np.float32)
    groups_all = (
        frame["name"].fillna("").astype(str)
        + " "
        + frame["description"].fillna("").astype(str)
    ).map(normalize).to_numpy()
    categories_all = frame["category"].astype(str).to_numpy()
    labels_all = frame["label"].to_numpy(dtype=np.int8)

    report = {"validation": "5-fold StratifiedGroupKFold", "categories": {}}
    macro = {}
    for category in sorted(frame["category"].unique()):
        positions = np.flatnonzero(categories_all == category)
        features = features_all[positions]
        labels = labels_all[positions]
        groups = groups_all[positions]
        folds = list(StratifiedGroupKFold(
            5, shuffle=True, random_state=42
        ).split(features, labels, groups))
        results = {}
        results["logreg_c1"] = evaluate_estimator(
            lambda fold: LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000,
                solver="liblinear", random_state=42 + fold,
            ),
            features, labels, folds,
        )
        results["logreg_c3"] = evaluate_estimator(
            lambda fold: LogisticRegression(
                C=3.0, class_weight="balanced", max_iter=2000,
                solver="liblinear", random_state=42 + fold,
            ),
            features, labels, folds,
        )
        results["ridge_a1"] = evaluate_estimator(
            lambda fold: RidgeClassifier(alpha=1.0, class_weight="balanced"),
            features, labels, folds,
        )
        results["ridge_a10"] = evaluate_estimator(
            lambda fold: RidgeClassifier(alpha=10.0, class_weight="balanced"),
            features, labels, folds,
        )
        results["extra_trees"] = evaluate_estimator(
            lambda fold: ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=8,
                random_state=42 + fold,
            ),
            features, labels, folds, probability=True,
        )
        results["centroid"] = centroid_scores(features, labels, folds)

        baseline_mm = text_archive["mm_scores"][positions].astype(np.float32)
        baseline_f1, baseline_threshold = best_threshold(labels, baseline_mm)
        results["linear_svc_baseline"] = {
            "f1": baseline_f1,
            "threshold": baseline_threshold,
            "scores": baseline_mm,
        }
        text_rank = rank01(text_scores_all[positions])
        for model_name, result in list(results.items()):
            mm_rank = rank01(result["scores"])
            best_fusion = None
            for alpha in np.linspace(0.5, 0.95, 10):
                fused = alpha * text_rank + (1 - alpha) * mm_rank
                score, threshold = best_threshold(labels, fused)
                if best_fusion is None or score > best_fusion["f1"]:
                    best_fusion = {
                        "f1": score,
                        "threshold": threshold,
                        "alpha_text": float(alpha),
                    }
            result["text_fusion"] = best_fusion

        text_rank = rank01(text_scores_all[positions])
        svc_rank = rank01(results["linear_svc_baseline"]["scores"])
        trees_rank = rank01(results["extra_trees"]["scores"])
        tri_head_fusion = None
        for text_steps in range(21):
            for svc_steps in range(21 - text_steps):
                tree_steps = 20 - text_steps - svc_steps
                weights = np.array([text_steps, svc_steps, tree_steps], dtype=float) / 20
                scores = weights[0] * text_rank + weights[1] * svc_rank + weights[2] * trees_rank
                score, threshold = best_threshold(labels, scores)
                if tri_head_fusion is None or score > tri_head_fusion["f1"]:
                    tri_head_fusion = {
                        "f1": score,
                        "threshold": threshold,
                        "weight_text": float(weights[0]),
                        "weight_svc": float(weights[1]),
                        "weight_extra_trees": float(weights[2]),
                    }

        clean = {
            name: {key: value for key, value in result.items() if key != "scores"}
            for name, result in results.items()
        }
        report["categories"][category] = {**clean, "tri_head_fusion": tri_head_fusion}
        for name, result in clean.items():
            macro.setdefault(name, []).append(result["f1"])
            macro.setdefault(name + "+text", []).append(result["text_fusion"]["f1"])
        macro.setdefault("tri_head_fusion", []).append(tri_head_fusion["f1"])
        print(category, json.dumps(clean, ensure_ascii=False), flush=True)

    report["macro_f1"] = {name: float(np.mean(values)) for name, values in macro.items()}
    report["runtime_minutes"] = (time.monotonic() - started) / 60
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["macro_f1"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

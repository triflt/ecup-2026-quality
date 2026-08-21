from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


DATA = Path(os.environ.get("ECUP_DATA", "/work/input/data.csv"))
EMBEDDINGS = Path(os.environ.get("ECUP_ALL_IMAGE_EMBEDDINGS", "/work/input/artifacts/train_embeddings_fp16.npz"))
OUTPUT = Path(os.environ.get("ECUP_REPORT", "/work/output/group_cv_report.json"))
C_VALUES = (0.03, 0.1, 0.3, 1.0, 3.0)


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


def oof_svc(features, labels, groups, folds):
    best = None
    for c_value in C_VALUES:
        scores = np.empty(len(labels), dtype=np.float32)
        for train_idx, valid_idx in folds:
            model = LinearSVC(
                C=c_value, class_weight="balanced", dual="auto", max_iter=8000, random_state=42
            )
            model.fit(features[train_idx], labels[train_idx])
            scores[valid_idx] = model.decision_function(features[valid_idx])
        score, threshold = best_threshold(labels, scores)
        if best is None or score > best["f1"]:
            best = {"f1": score, "threshold": threshold, "C": c_value, "scores": scores.copy()}
    return best


def rank01(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def main():
    started = time.monotonic()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        data_url = os.environ["DATA_URL"]
        print("downloading data.csv", flush=True)
        urllib.request.urlretrieve(data_url, DATA)
    frame = pd.read_csv(DATA)
    archive = np.load(EMBEDDINGS)
    if not np.array_equal(frame["id"].astype(str).to_numpy(), archive["ids"].astype(str)):
        raise ValueError("embedding row order mismatch")
    embeddings = archive["embeddings"].astype(np.float32)
    names = frame["name"].fillna("").astype(str)
    descriptions = frame["description"].fillna("").astype(str)
    texts = names + "\n" + names + "\n" + descriptions
    frame["group"] = (names + " " + descriptions).map(normalize)

    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.997,
                                 sublinear_tf=True, max_features=160_000, dtype=np.float32)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True, max_features=200_000, dtype=np.float32)),
    ])
    text_matrix = vectorizer.fit_transform(texts)
    print(f"text_matrix={text_matrix.shape} nnz={text_matrix.nnz}", flush=True)

    report = {"validation": "5-fold StratifiedGroupKFold by normalized full text", "categories": {}}
    macro = {"text": [], "multimodal": [], "late_fusion": []}
    for category in sorted(frame["category"].unique()):
        positions = np.flatnonzero(frame["category"].to_numpy() == category)
        labels = frame.iloc[positions]["label"].to_numpy(dtype=np.int8)
        groups = frame.iloc[positions]["group"].to_numpy()
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(positions)), labels, groups
        ))
        text_result = oof_svc(text_matrix[positions], labels, groups, folds)
        mm_result = oof_svc(embeddings[positions], labels, groups, folds)
        text_rank = rank01(text_result["scores"])
        mm_rank = rank01(mm_result["scores"])
        fusion_best = None
        for alpha in np.linspace(0, 1, 21):
            scores = alpha * text_rank + (1 - alpha) * mm_rank
            score, threshold = best_threshold(labels, scores)
            if fusion_best is None or score > fusion_best["f1"]:
                fusion_best = {"f1": score, "threshold": threshold, "alpha_text": float(alpha)}
        entry = {
            "rows": int(len(positions)),
            "positive_rate": float(labels.mean()),
            "groups": int(pd.Series(groups).nunique()),
            "text": {k: v for k, v in text_result.items() if k != "scores"},
            "multimodal": {k: v for k, v in mm_result.items() if k != "scores"},
            "late_fusion": fusion_best,
            "fold_sizes": [int(len(valid)) for _, valid in folds],
            "fold_positives": [int(labels[valid].sum()) for _, valid in folds],
        }
        report["categories"][category] = entry
        for key in macro:
            macro[key].append(entry[key]["f1"])
        print(category, json.dumps(entry, ensure_ascii=False), flush=True)
    report["macro_f1"] = {key: float(np.mean(values)) for key, values in macro.items()}
    report["runtime_minutes"] = (time.monotonic() - started) / 60
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

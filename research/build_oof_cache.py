from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


DATA = Path("/work/input/data.csv")
DATA_PARTS = Path("/work/input/data_parts")
EMBEDDINGS = Path("/work/input/artifacts/train_embeddings_fp16.npz")
OUTPUT = Path("/work/output")
CONFIG = {
    "БАД": {
        "alpha_text": 0.85,
        "fusion_threshold": 0.24864045896205267,
        "text_threshold": -0.12829011704277063,
        "mm_threshold": -0.43812282607790737,
    },
    "Легковоспламеняющиеся": {
        "alpha_text": 0.65,
        "fusion_threshold": 0.9591804083988902,
        "text_threshold": -0.31445564352391603,
        "mm_threshold": 0.10255206533582154,
    },
}


def normalize(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", str(value).lower())
    value = re.sub(r"[^a-zа-яё0-9]+", " ", value)
    return " ".join(value.split())


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def oof_scores(features, labels, folds, c_value: float) -> np.ndarray:
    scores = np.empty(len(labels), dtype=np.float32)
    for fold, (train_idx, valid_idx) in enumerate(folds):
        model = LinearSVC(
            C=c_value,
            class_weight="balanced",
            dual="auto",
            max_iter=8000,
            random_state=42 + fold,
        )
        model.fit(features[train_idx], labels[train_idx])
        scores[valid_idx] = model.decision_function(features[valid_idx])
    return scores


def main():
    started = time.monotonic()
    data_path = DATA
    if not data_path.exists():
        parts = sorted(DATA_PARTS.rglob("data.csv.gz.part-*"))
        if parts:
            data_path = Path("/work/data.csv.gz")
            with data_path.open("wb") as destination:
                for part in parts:
                    destination.write(part.read_bytes())
        elif os.environ.get("DATA_URL"):
            data_path = Path("/work/data.csv")
            urllib.request.urlretrieve(os.environ["DATA_URL"], data_path)
        else:
            raise FileNotFoundError("data.csv, compressed parts, and DATA_URL are all absent")
    frame = pd.read_csv(data_path)
    archive = np.load(EMBEDDINGS)
    ids = frame["id"].astype(str).to_numpy()
    if not np.array_equal(ids, archive["ids"].astype(str)):
        raise ValueError("embedding row order mismatch")
    embeddings = archive["embeddings"].astype(np.float32)
    names = frame["name"].fillna("").astype(str)
    descriptions = frame["description"].fillna("").astype(str)
    texts = names + "\n" + names + "\n" + descriptions
    groups = (names + " " + descriptions).map(normalize).to_numpy()

    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.997,
            sublinear_tf=True, max_features=160_000, dtype=np.float32,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3,
            sublinear_tf=True, max_features=200_000, dtype=np.float32,
        )),
    ])
    text_matrix = vectorizer.fit_transform(texts)
    print(f"text_matrix={text_matrix.shape} nnz={text_matrix.nnz}", flush=True)

    text_scores = np.full(len(frame), np.nan, dtype=np.float32)
    mm_scores = np.full(len(frame), np.nan, dtype=np.float32)
    fused_scores = np.full(len(frame), np.nan, dtype=np.float32)
    predictions = np.zeros(len(frame), dtype=np.int8)
    text_predictions = np.zeros(len(frame), dtype=np.int8)
    mm_predictions = np.zeros(len(frame), dtype=np.int8)
    fold_ids = np.full(len(frame), -1, dtype=np.int8)
    report = {"validation": "5-fold StratifiedGroupKFold by normalized full text", "categories": {}}

    categories = frame["category"].astype(str).to_numpy()
    labels_all = frame["label"].to_numpy(dtype=np.int8)
    for category, config in CONFIG.items():
        positions = np.flatnonzero(categories == category)
        labels = labels_all[positions]
        local_groups = groups[positions]
        folds = list(StratifiedGroupKFold(
            5, shuffle=True, random_state=42
        ).split(np.zeros(len(positions)), labels, local_groups))
        for fold, (_, valid_idx) in enumerate(folds):
            fold_ids[positions[valid_idx]] = fold
        local_text = oof_scores(text_matrix[positions], labels, folds, 1.0)
        local_mm = oof_scores(embeddings[positions], labels, folds, 3.0)
        local_fused = (
            config["alpha_text"] * rank01(local_text)
            + (1.0 - config["alpha_text"]) * rank01(local_mm)
        )
        local_pred = (local_fused >= config["fusion_threshold"]).astype(np.int8)
        local_text_pred = (local_text >= config["text_threshold"]).astype(np.int8)
        local_mm_pred = (local_mm >= config["mm_threshold"]).astype(np.int8)
        text_scores[positions] = local_text
        mm_scores[positions] = local_mm
        fused_scores[positions] = local_fused
        predictions[positions] = local_pred
        text_predictions[positions] = local_text_pred
        mm_predictions[positions] = local_mm_pred
        report["categories"][category] = {
            "rows": int(len(positions)),
            "f1": float(f1_score(labels, local_pred)),
            "predicted_positive": int(local_pred.sum()),
            "true_positive": int(labels.sum()),
            "errors": int(np.sum(local_pred != labels)),
        }

    if np.isnan(text_scores).any() or np.isnan(mm_scores).any() or (fold_ids < 0).any():
        raise ValueError("incomplete OOF arrays")

    hard = frame[["id", "category", "label", "name", "description"]].copy()
    hard["fold"] = fold_ids
    hard["text_score"] = text_scores
    hard["mm_score"] = mm_scores
    hard["fused_score"] = fused_scores
    hard["prediction"] = predictions
    hard["text_prediction"] = text_predictions
    hard["mm_prediction"] = mm_predictions
    hard["disagreement"] = text_predictions != mm_predictions
    hard["error"] = predictions != labels_all
    hard["uncertainty"] = [
        abs(score - CONFIG[category]["fusion_threshold"])
        for score, category in zip(fused_scores, categories)
    ]
    # Selection is label-blind and can be reproduced on public/private test.
    hard["selector_score"] = hard["uncertainty"] - 0.025 * hard["disagreement"].astype(float)
    hard["selector_rank"] = hard.groupby("category")["selector_score"].rank(
        method="first", ascending=True
    ).astype(int)
    hard = hard.sort_values(["category", "selector_rank"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    hard.to_csv(OUTPUT / "hard_cases.csv", index=False)
    np.savez_compressed(
        OUTPUT / "oof_scores.npz",
        ids=ids,
        labels=labels_all,
        categories=categories,
        fold_ids=fold_ids,
        text_scores=text_scores,
        mm_scores=mm_scores,
        fused_scores=fused_scores,
        predictions=predictions,
    )
    report["macro_f1"] = float(np.mean([
        entry["f1"] for entry in report["categories"].values()
    ]))
    report["runtime_minutes"] = (time.monotonic() - started) / 60
    (OUTPUT / "oof_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

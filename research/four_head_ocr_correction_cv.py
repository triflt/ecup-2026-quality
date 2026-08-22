from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


INPUT = Path("/work/input")
OUTPUT = Path("/work/output/four_head_ocr_report.json")
C_VALUES = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def locate(name):
    matches = list(INPUT.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"expected one {name}, found {matches}")
    return matches[0]


def best_threshold(labels, scores):
    best = (-1.0, 0.0)
    for threshold in np.unique(np.quantile(scores, np.linspace(0.002, 0.998, 700))):
        score = f1_score(labels, scores >= threshold)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best


def vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, sublinear_tf=True,
            max_features=20_000, dtype=np.float32,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, max_features=40_000, dtype=np.float32,
        )),
    ])


def main():
    started = time.monotonic()
    oof = np.load(locate("four_head_oof.npz"), allow_pickle=True)
    ocr = pd.read_csv(locate("ocr_hard_cases.csv"))
    categories = oof["categories"].astype(str)
    ids = oof["ids"].astype(str)
    labels_all = oof["labels"].astype(np.int8)
    baseline_predictions = oof["predictions"].astype(np.int8)
    category = "Легковоспламеняющиеся"
    selected = ocr[ocr["category"].astype(str) == category].copy()
    selected["id_key"] = selected["id"].astype(str)
    index_by_id = {item_id: index for index, item_id in enumerate(ids)}
    selected["global_index"] = selected["id_key"].map(index_by_id)
    if len(selected) != 150 or selected["global_index"].isna().any():
        raise ValueError("invalid OCR selection")
    selected_indices = selected["global_index"].to_numpy(dtype=np.int64)
    labels = labels_all[selected_indices]
    folds = oof["fold_ids"][selected_indices].astype(np.int8)
    texts = selected["ocr"].fillna("").astype(str).to_numpy()
    numeric = np.column_stack([
        oof["text"][selected_indices],
        oof["all_images"][selected_indices],
        oof["first_image"][selected_indices],
        oof["extra_trees"][selected_indices],
        oof["fused"][selected_indices],
        selected["prediction"].to_numpy(dtype=np.float32),
        selected["disagreement"].to_numpy(dtype=np.float32),
        selected["selector_rank"].to_numpy(dtype=np.float32) / 150.0,
    ]).astype(np.float32)
    best = None
    for c_value in C_VALUES:
        scores = np.empty(len(selected), dtype=np.float32)
        for fold in range(5):
            train = folds != fold
            valid = folds == fold
            encoder = vectorizer()
            train_text = encoder.fit_transform(texts[train])
            valid_text = encoder.transform(texts[valid])
            train_matrix = sparse.hstack(
                [train_text, sparse.csr_matrix(numeric[train])], format="csr"
            )
            valid_matrix = sparse.hstack(
                [valid_text, sparse.csr_matrix(numeric[valid])], format="csr"
            )
            model = LinearSVC(
                C=c_value, class_weight="balanced", dual="auto",
                max_iter=10000, random_state=42 + fold,
            )
            model.fit(train_matrix, labels[train])
            scores[valid] = model.decision_function(valid_matrix)
        score, threshold = best_threshold(labels, scores)
        if best is None or score > best["selected_f1"]:
            best = {
                "selected_f1": score,
                "threshold": threshold,
                "C": c_value,
                "scores": scores.copy(),
            }

    corrected = baseline_predictions.copy()
    corrected[selected_indices] = (best["scores"] >= best["threshold"]).astype(np.int8)
    category_scores = {}
    for value in sorted(np.unique(categories)):
        mask = categories == value
        category_scores[value] = {
            "baseline_f1": float(f1_score(labels_all[mask], baseline_predictions[mask])),
            "corrected_f1": float(f1_score(labels_all[mask], corrected[mask])),
            "changed": int(np.sum(baseline_predictions[mask] != corrected[mask])),
        }
    report = {
        "selection": "same label-blind 150 hard flammable cases",
        "ocr_nonempty": int(selected["ocr"].fillna("").str.strip().ne("").sum()),
        "selected_baseline_f1": float(f1_score(labels, baseline_predictions[selected_indices])),
        "selected_correction_oof_f1": best["selected_f1"],
        "C": best["C"],
        "threshold": best["threshold"],
        "categories": category_scores,
        "macro_baseline_f1": float(np.mean([v["baseline_f1"] for v in category_scores.values()])),
        "macro_corrected_f1": float(np.mean([v["corrected_f1"] for v in category_scores.values()])),
        "runtime_seconds": time.monotonic() - started,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

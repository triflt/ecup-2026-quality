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
OUTPUT = Path("/work/output/ocr_correction_report.json")
C_VALUES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
NUMERIC_COLUMNS = (
    "text_score",
    "mm_score",
    "fused_score",
    "uncertainty",
    "disagreement",
    "prediction",
    "selector_rank",
)


def locate(name: str) -> Path:
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


def numeric_matrix(frame):
    values = frame[list(NUMERIC_COLUMNS)].copy()
    values["disagreement"] = values["disagreement"].astype(float)
    values["selector_rank"] = values["selector_rank"].astype(float) / 150.0
    return sparse.csr_matrix(values.to_numpy(dtype=np.float32))


def category_oof(selected):
    labels = selected["label"].to_numpy(dtype=np.int8)
    folds = selected["fold"].to_numpy(dtype=np.int8)
    texts = selected["ocr"].fillna("").astype(str).to_numpy()
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
                [train_text, numeric_matrix(selected.iloc[np.flatnonzero(train)])],
                format="csr",
            )
            valid_matrix = sparse.hstack(
                [valid_text, numeric_matrix(selected.iloc[np.flatnonzero(valid)])],
                format="csr",
            )
            model = LinearSVC(
                C=c_value,
                class_weight="balanced",
                dual="auto",
                max_iter=8000,
                random_state=42 + fold,
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
    return best


def main():
    started = time.monotonic()
    hard = pd.read_csv(locate("hard_cases.csv"))
    ocr = pd.read_csv(locate("ocr_hard_cases.csv"))
    hard["id_key"] = hard["id"].astype(str)
    ocr["id_key"] = ocr["id"].astype(str)
    selected = ocr[["id_key", "ocr"]].merge(hard, on="id_key", how="left", validate="one_to_one")
    if len(selected) != 300 or selected["fold"].isna().any():
        raise ValueError(f"invalid selected merge: rows={len(selected)}")
    report = {"selection": "150 label-blind hard cases per category", "categories": {}}
    baseline_macro, corrected_macro = [], []
    for category in sorted(hard["category"].unique()):
        full = hard[hard["category"] == category].copy()
        local = selected[selected["category"] == category].copy()
        if len(local) != 150:
            raise ValueError(f"{category}: expected 150 OCR rows, got {len(local)}")
        result = category_oof(local)
        selected_predictions = (result["scores"] >= result["threshold"]).astype(np.int8)
        baseline_predictions = full["prediction"].to_numpy(dtype=np.int8)
        corrected_predictions = baseline_predictions.copy()
        index_by_id = {value: index for index, value in enumerate(full["id_key"])}
        for item_id, prediction in zip(local["id_key"], selected_predictions):
            corrected_predictions[index_by_id[item_id]] = prediction
        labels = full["label"].to_numpy(dtype=np.int8)
        baseline_f1 = float(f1_score(labels, baseline_predictions))
        corrected_f1 = float(f1_score(labels, corrected_predictions))
        selected_baseline_f1 = float(f1_score(
            local["label"].to_numpy(dtype=np.int8),
            local["prediction"].to_numpy(dtype=np.int8),
        ))
        entry = {
            "rows": int(len(full)),
            "ocr_rows": int(len(local)),
            "ocr_nonempty": int(local["ocr"].fillna("").str.strip().ne("").sum()),
            "selected_baseline_f1": selected_baseline_f1,
            "selected_correction_oof_f1": result["selected_f1"],
            "C": result["C"],
            "threshold": result["threshold"],
            "full_baseline_f1": baseline_f1,
            "full_corrected_f1": corrected_f1,
            "changed": int(np.sum(selected_predictions != local["prediction"].to_numpy(dtype=np.int8))),
        }
        report["categories"][category] = entry
        baseline_macro.append(baseline_f1)
        corrected_macro.append(corrected_f1)
        print(category, json.dumps(entry, ensure_ascii=False), flush=True)
    report["macro_baseline_f1"] = float(np.mean(baseline_macro))
    report["macro_corrected_f1"] = float(np.mean(corrected_macro))
    report["runtime_seconds"] = time.monotonic() - started
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

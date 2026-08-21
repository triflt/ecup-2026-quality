from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


INPUT = Path("/work/input")
OUTPUT = Path("/work/output")
C_VALUES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
NUMERIC = (
    "text_score", "mm_score", "fused_score", "uncertainty", "disagreement",
    "selector_rank", "base_rank", "lora_rank", "lora_fused_score", "lora_prediction",
)


def f1(labels, predictions):
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def best_threshold(labels, scores):
    candidates = np.unique(np.quantile(scores, np.linspace(0.002, 0.998, 500)))
    return max((f1(labels, scores >= threshold), float(threshold)) for threshold in candidates)


def vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=20_000, dtype=np.float32)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=40_000, dtype=np.float32)),
    ])


def numeric_matrix(frame):
    values = frame[list(NUMERIC)].copy()
    values["disagreement"] = values.disagreement.astype(float)
    values["selector_rank"] = values.selector_rank.astype(float) / 150.0
    return sparse.csr_matrix(values.to_numpy(dtype=np.float32))


def fit_model(frame, positions, c_value):
    local = frame.iloc[positions]
    encoder = vectorizer()
    text = encoder.fit_transform(local.ocr.fillna("").astype(str))
    matrix = sparse.hstack([text, numeric_matrix(local)], format="csr")
    model = LinearSVC(C=c_value, class_weight="balanced", dual="auto", max_iter=8000, random_state=42)
    model.fit(matrix, local.label.to_numpy(dtype=np.int8))
    return encoder, model


def score_model(frame, positions, encoder, model):
    local = frame.iloc[positions]
    text = encoder.transform(local.ocr.fillna("").astype(str))
    matrix = sparse.hstack([text, numeric_matrix(local)], format="csr")
    return model.decision_function(matrix)


def nested_category(local):
    labels = local.label.to_numpy(dtype=np.int8)
    folds = local.fold.to_numpy(dtype=np.int8)
    outer_scores = np.zeros(len(local), dtype=np.float32)
    outer_predictions = np.zeros(len(local), dtype=np.int8)
    rows = []
    choices = []
    for outer_fold in sorted(np.unique(folds)):
        train_positions = np.flatnonzero(folds != outer_fold)
        valid_positions = np.flatnonzero(folds == outer_fold)
        best = None
        for c_value in C_VALUES:
            inner_scores = np.zeros(len(train_positions), dtype=np.float32)
            train_folds = folds[train_positions]
            for inner_fold in sorted(np.unique(train_folds)):
                inner_train_local = np.flatnonzero(train_folds != inner_fold)
                inner_valid_local = np.flatnonzero(train_folds == inner_fold)
                encoder, model = fit_model(local, train_positions[inner_train_local], c_value)
                inner_scores[inner_valid_local] = score_model(
                    local, train_positions[inner_valid_local], encoder, model
                )
            value, threshold = best_threshold(labels[train_positions], inner_scores)
            candidate = (value, -c_value, c_value, threshold)
            if best is None or candidate > best:
                best = candidate
        _, _, c_value, threshold = best
        encoder, model = fit_model(local, train_positions, c_value)
        scores = score_model(local, valid_positions, encoder, model)
        predictions = (scores >= threshold).astype(np.int8)
        outer_scores[valid_positions] = scores
        outer_predictions[valid_positions] = predictions
        choices.append(c_value)
        rows.append({
            "fold": int(outer_fold), "rows": len(valid_positions), "C": c_value,
            "threshold": threshold, "f1": f1(labels[valid_positions], predictions),
        })
    chosen_c = Counter(choices).most_common(1)[0][0]
    # Five-fold OOF for the fixed final C, then calibrate a full-data model.
    cv_scores = np.zeros(len(local), dtype=np.float32)
    for fold in sorted(np.unique(folds)):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        encoder, model = fit_model(local, train, chosen_c)
        cv_scores[valid] = score_model(local, valid, encoder, model)
    cv_f1, final_threshold = best_threshold(labels, cv_scores)
    encoder, model = fit_model(local, np.arange(len(local)), chosen_c)
    return {
        "predictions": outer_predictions,
        "nested_selected_f1": f1(labels, outer_predictions),
        "folds": rows,
        "chosen_C": chosen_c,
        "five_fold_fixed_C_f1": cv_f1,
        "threshold": final_threshold,
        "encoder": encoder,
        "model": model,
    }


def main():
    hard = pd.read_csv(INPUT / "hard_cases.csv")
    ocr = pd.read_csv(INPUT / "ocr_hard_cases.csv")
    lora = np.load(INPUT / "lora_fusion.npz", allow_pickle=True)
    hard["id_key"] = hard.id.astype(str)
    ocr["id_key"] = ocr.id.astype(str)
    lora_positions = {item_id: index for index, item_id in enumerate(lora["ids"].astype(str))}
    if set(hard.id_key) != set(lora_positions):
        raise ValueError("LoRA id set mismatch")
    aligned = np.asarray([lora_positions[item_id] for item_id in hard.id_key], dtype=np.int64)
    hard["base_rank"] = lora["base_rank"][aligned]
    hard["lora_rank"] = lora["lora_rank"][aligned]
    hard["lora_fused_score"] = 0.0
    hard["lora_prediction"] = 0
    configs = {
        "БАД": (0.40, 0.60, 0.25814030990600584),
        "Легковоспламеняющиеся": (0.75, 0.25, 0.9654546632766724),
    }
    for category, (weight_base, weight_lora, threshold) in configs.items():
        mask = hard.category.to_numpy() == category
        scores = weight_base * hard.loc[mask, "base_rank"] + weight_lora * hard.loc[mask, "lora_rank"]
        hard.loc[mask, "lora_fused_score"] = scores
        hard.loc[mask, "lora_prediction"] = (scores >= threshold).astype(np.int8)
    selected = ocr[["id_key", "ocr"]].merge(hard, on="id_key", how="left", validate="one_to_one")
    report = {"selection": "existing 150 label-blind hard cases per category", "categories": {}}
    bundle = {}
    baseline_macro, corrected_macro = [], []
    for category in sorted(hard.category.unique()):
        full = hard[hard.category == category].copy()
        local = selected[selected.category == category].reset_index(drop=True)
        result = nested_category(local)
        corrected = full.lora_prediction.to_numpy(dtype=np.int8).copy()
        index_by_id = {value: index for index, value in enumerate(full.id_key)}
        for item_id, prediction in zip(local.id_key, result["predictions"]):
            corrected[index_by_id[item_id]] = prediction
        labels = full.label.to_numpy(dtype=np.int8)
        baseline_f1 = f1(labels, full.lora_prediction)
        corrected_f1 = f1(labels, corrected)
        report["categories"][category] = {
            "rows": len(full), "ocr_rows": len(local),
            "ocr_nonempty": int(local.ocr.fillna("").str.strip().ne("").sum()),
            "baseline_lora_f1": baseline_f1, "nested_corrected_f1": corrected_f1,
            "selected_nested_f1": result["nested_selected_f1"],
            "chosen_C": result["chosen_C"], "final_threshold": result["threshold"],
            "folds": result["folds"],
        }
        bundle[category] = {
            "encoder": result["encoder"], "model": result["model"],
            "threshold": result["threshold"], "numeric_columns": NUMERIC,
        }
        baseline_macro.append(baseline_f1)
        corrected_macro.append(corrected_f1)
    report["macro_baseline"] = float(np.mean(baseline_macro))
    report["macro_nested_corrected"] = float(np.mean(corrected_macro))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ocr_lora_nested_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(bundle, OUTPUT / "ocr_lora_correction.joblib")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

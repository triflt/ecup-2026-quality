from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


CODE = Path("/work/code")
sys.path.insert(0, str(CODE))
from src.model import StrongTextBundle, compose_text, fingerprint, normalize  # noqa: E402


DATA = Path("/work/input/data.csv")
MM_ARTIFACTS = Path("/work/input/artifacts")
OUTPUT = Path("/work/output")
C_BY_CATEGORY = {"БАД": 1.0, "Легковоспламеняющиеся": 1.0}
GROUPED_RAW_THRESHOLDS = {
    "БАД": -0.12829011704277063,
    "Легковоспламеняющиеся": -0.31445564352391603,
}
FUSION = {
    "БАД": {"alpha_text": 0.85, "threshold": 0.24864045896205267},
    "Легковоспламеняющиеся": {"alpha_text": 0.65, "threshold": 0.9591804083988902},
}


def consistent_lookup(frame, key_column, minimum_count):
    result = {}
    for key, labels in frame.groupby(["category", key_column], sort=False)["label"]:
        values = labels.astype(int).tolist()
        if len(values) >= minimum_count and min(values) == max(values):
            result[key] = values[0]
    return result


def main():
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        urllib.request.urlretrieve(os.environ["DATA_URL"], DATA)
    frame = pd.read_csv(DATA)
    frame["name"] = frame["name"].fillna("")
    frame["description"] = frame["description"].fillna("")
    frame["text"] = [compose_text(a, b) for a, b in zip(frame.name, frame.description)]
    frame["text_hash"] = frame["text"].map(fingerprint)
    frame["normalized_name"] = frame["name"].map(normalize)

    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.997, sublinear_tf=True,
            max_features=180_000, dtype=np.float32,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
            max_features=220_000, dtype=np.float32,
        )),
    ])
    matrix = vectorizer.fit_transform(frame["text"])
    categories = frame["category"].astype(str).to_numpy()
    models, text_sorted = {}, {}
    for category in sorted(frame.category.unique()):
        mask = categories == category
        labels = frame.loc[mask, "label"].to_numpy(dtype=np.int8)
        model = LinearSVC(
            C=C_BY_CATEGORY[category], class_weight="balanced", dual="auto",
            max_iter=8000, random_state=42,
        )
        model.fit(matrix[mask], labels)
        models[category] = model
        text_sorted[category] = np.sort(model.decision_function(matrix[mask]).astype(np.float32))
        print(category, "text_train_rows", int(mask.sum()), flush=True)

    text_bundle = StrongTextBundle(
        vectorizer=vectorizer,
        category_models=models,
        thresholds=GROUPED_RAW_THRESHOLDS,
        exact_lookup=consistent_lookup(frame, "text_hash", 2),
        name_lookup=consistent_lookup(frame, "normalized_name", 3),
    )

    mm_bundle = joblib.load(MM_ARTIFACTS / "multimodal_classifier.joblib")
    archive = np.load(MM_ARTIFACTS / "train_embeddings_fp16.npz")
    if not np.array_equal(frame.id.astype(str).to_numpy(), archive["ids"].astype(str)):
        raise ValueError("embedding row order mismatch")
    embeddings = archive["embeddings"].astype(np.float32)
    mm_sorted = {}
    for category, model in mm_bundle["models"].items():
        mask = categories == category
        mm_sorted[category] = np.sort(model.decision_function(embeddings[mask]).astype(np.float32))

    calibration = {
        "version": 1,
        "validation": "5-fold StratifiedGroupKFold by normalized full text",
        "fusion": FUSION,
        "text_score_sorted": text_sorted,
        "multimodal_score_sorted": mm_sorted,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(text_bundle, OUTPUT / "strong_text_full.joblib", compress=3)
    joblib.dump(calibration, OUTPUT / "fusion_calibration.joblib", compress=3)
    (OUTPUT / "training_summary.json").write_text(json.dumps({
        "rows": len(frame),
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "C_by_category": C_BY_CATEGORY,
        "grouped_raw_thresholds": GROUPED_RAW_THRESHOLDS,
        "fusion": FUSION,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", sorted(p.name for p in OUTPUT.iterdir()), flush=True)


if __name__ == "__main__":
    main()

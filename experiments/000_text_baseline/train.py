import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "submission"))
from src.model import StrongTextBundle, compose_text, fingerprint, normalize


def tune_threshold(labels, scores):
    best_threshold, best_score = 0.0, -1.0
    for threshold in np.quantile(scores, np.linspace(0.01, 0.99, 300)):
        score = f1_score(labels, scores >= threshold)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def consistent_lookup(frame, key_column, minimum_count):
    result = {}
    for key, labels in frame.groupby(["category", key_column], sort=False)["label"]:
        values = labels.astype(int).tolist()
        if len(values) >= minimum_count and min(values) == max(values):
            result[key] = values[0]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output", default="experiments/000_text_baseline/submission/strong_text.joblib")
    args = parser.parse_args()

    data = pd.read_csv(args.train_data)
    data["name"] = data["name"].fillna("")
    data["description"] = data["description"].fillna("")
    data["text"] = [compose_text(a, b) for a, b in zip(data["name"], data["description"])]
    data["text_hash"] = data["text"].map(fingerprint)
    data["normalized_name"] = data["name"].map(normalize)

    train_positions, validation_positions = [], []
    for category in sorted(data["category"].unique()):
        positions = np.flatnonzero(data["category"].to_numpy() == category)
        left, right = train_test_split(
            positions,
            test_size=0.22,
            random_state=42,
            stratify=data.iloc[positions]["label"].to_numpy(),
        )
        train_positions.extend(left.tolist())
        validation_positions.extend(right.tolist())
    train_positions = np.asarray(sorted(train_positions))
    validation_positions = np.asarray(sorted(validation_positions))

    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.997,
            sublinear_tf=True, max_features=180_000, dtype=np.float32,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3,
            sublinear_tf=True, max_features=220_000, dtype=np.float32,
        )),
    ])
    train_matrix = vectorizer.fit_transform(data.iloc[train_positions]["text"])
    validation_matrix = vectorizer.transform(data.iloc[validation_positions]["text"])

    models, thresholds = {}, {}
    validation_predictions = np.zeros(len(validation_positions), dtype=np.int8)
    validation_categories = data.iloc[validation_positions]["category"].to_numpy()
    labels_all = data.iloc[validation_positions]["label"].to_numpy()
    for category in sorted(data["category"].unique()):
        train_mask = data.iloc[train_positions]["category"].to_numpy() == category
        validation_mask = validation_categories == category
        y_train = data.iloc[train_positions]["label"].to_numpy()[train_mask]
        y_validation = labels_all[validation_mask]
        best = None
        for c_value in (0.5, 1.0, 2.0, 4.0):
            model = LinearSVC(C=c_value, class_weight="balanced", random_state=42)
            model.fit(train_matrix[train_mask], y_train)
            scores = model.decision_function(validation_matrix[validation_mask])
            threshold, score = tune_threshold(y_validation, scores)
            if best is None or score > best[0]:
                best = (score, model, threshold, c_value)
        models[category] = best[1]
        thresholds[category] = best[2]
        scores = best[1].decision_function(validation_matrix[validation_mask])
        validation_predictions[validation_mask] = scores >= best[2]
        print(f"category={category!r} C={best[3]} threshold={best[2]:.5f} f1={best[0]:.5f}")

    category_f1 = []
    for category in sorted(data["category"].unique()):
        mask = validation_categories == category
        score = f1_score(labels_all[mask], validation_predictions[mask])
        category_f1.append(score)
    print(f"macro_category_f1={np.mean(category_f1):.5f}")

    bundle = StrongTextBundle(
        vectorizer=vectorizer,
        category_models=models,
        thresholds=thresholds,
        exact_lookup=consistent_lookup(data, "text_hash", 2),
        name_lookup=consistent_lookup(data, "normalized_name", 3),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle.save(output)


if __name__ == "__main__":
    main()

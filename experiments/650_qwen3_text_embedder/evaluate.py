from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

CATEGORIES = ("БАД", "Легковоспламеняющиеся")
SCREEN_FOLDS = (0, 3)


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(((labels == 1) & predictions).sum())
    fp = int(((labels == 0) & predictions).sum())
    fn = int(((labels == 1) & ~predictions).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.997,
                    sublinear_tf=True,
                    max_features=160_000,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                    max_features=200_000,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def fit_scores(
    train_feature: np.ndarray | sparse.spmatrix,
    valid_feature: np.ndarray | sparse.spmatrix,
    train_labels: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        dual="auto",
        max_iter=8_000,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", RuntimeWarning)
        model.fit(train_feature, train_labels)
        scores = model.decision_function(valid_feature)
    scores = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(scores).all():
        raise FloatingPointError("classifier produced non-finite scores")
    return scores


def load_runtime(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            item_id = str(row["id"])
            if item_id in rows:
                raise ValueError(f"duplicate runtime id: {item_id}")
            rows[item_id] = row
    return rows


def evaluate(
    *,
    runtime_path: Path,
    embedding_path: Path,
    baseline_path: Path,
    output_path: Path,
    folds_to_evaluate: tuple[int, ...] = SCREEN_FOLDS,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite evaluation report")
    runtime = load_runtime(runtime_path)
    with np.load(baseline_path, allow_pickle=False) as baseline:
        ids = baseline["ids"].astype(str)
        labels = baseline["labels"].astype(np.int8)
        categories = baseline["categories"].astype(str)
        folds = baseline["folds"].astype(np.int8)
    with np.load(embedding_path, allow_pickle=False) as archive:
        embedding_ids = archive["ids"].astype(str)
        embeddings = archive["embeddings"].astype(np.float32)
        prototype_names = archive["prototype_names"].astype(str)
        prototype_embeddings = archive["prototype_embeddings"].astype(np.float32)
    if set(ids) != set(runtime) or set(ids) != set(embedding_ids):
        raise ValueError("runtime, baseline and embedding ID sets differ")
    embedding_position = {item_id: index for index, item_id in enumerate(embedding_ids)}
    order = np.asarray([embedding_position[item_id] for item_id in ids], dtype=np.int64)
    embeddings = embeddings[order]
    if embeddings.shape != (len(ids), 1024) or not np.isfinite(embeddings).all():
        raise ValueError("embedding matrix schema or finiteness mismatch")
    texts = np.asarray(
        [f"{runtime[item_id]['name']}\n{runtime[item_id]['name']}\n{runtime[item_id]['description']}" for item_id in ids],
        dtype=str,
    )
    prototype_map = {name: prototype_embeddings[index] for index, name in enumerate(prototype_names)}
    zero_scores = np.empty(len(ids), dtype=np.float32)
    for category in CATEGORIES:
        local = categories == category
        zero_scores[local] = (
            embeddings[local] @ prototype_map[f"{category}|1"]
            - embeddings[local] @ prototype_map[f"{category}|0"]
        )

    tfidf_scores = np.full(len(ids), np.nan, dtype=np.float32)
    supervised_scores = np.full(len(ids), np.nan, dtype=np.float32)
    for fold in folds_to_evaluate:
        outer_train = folds != fold
        outer_valid = folds == fold
        text_model = vectorizer()
        train_text = text_model.fit_transform(texts[outer_train])
        valid_text = text_model.transform(texts[outer_valid])
        train_global = np.flatnonzero(outer_train)
        valid_global = np.flatnonzero(outer_valid)
        for category in CATEGORIES:
            train_local = categories[train_global] == category
            valid_local = categories[valid_global] == category
            train_indices = train_global[train_local]
            valid_indices = valid_global[valid_local]
            tfidf_scores[valid_indices] = fit_scores(
                train_text[train_local],
                valid_text[valid_local],
                labels[train_indices],
                seed=650 + fold,
            )
            supervised_scores[valid_indices] = fit_scores(
                embeddings[train_indices],
                embeddings[valid_indices],
                labels[train_indices],
                seed=650 + fold,
            )
    scope = np.isin(folds, folds_to_evaluate)
    if np.isnan(tfidf_scores[scope]).any() or np.isnan(supervised_scores[scope]).any():
        raise ValueError("screen scores are incomplete")
    tfidf_pred = tfidf_scores >= 0.0
    supervised_pred = supervised_scores >= 0.0
    zero_pred = zero_scores >= 0.0
    fold_metrics: dict[str, Any] = {}
    for fold in folds_to_evaluate:
        fold_metrics[str(fold)] = {"categories": {}}
        for category in CATEGORIES:
            local = (folds == fold) & (categories == category)
            fold_metrics[str(fold)]["categories"][category] = {
                "tfidf_f1": f1(labels[local], tfidf_pred[local]),
                "zero_shot_f1": f1(labels[local], zero_pred[local]),
                "supervised_f1": f1(labels[local], supervised_pred[local]),
            }
        for value in fold_metrics[str(fold)]["categories"].values():
            value["supervised_delta_vs_tfidf"] = value["supervised_f1"] - value["tfidf_f1"]
        fold_metrics[str(fold)]["tfidf_macro_f1"] = float(
            np.mean([value["tfidf_f1"] for value in fold_metrics[str(fold)]["categories"].values()])
        )
        fold_metrics[str(fold)]["zero_shot_macro_f1"] = float(
            np.mean([value["zero_shot_f1"] for value in fold_metrics[str(fold)]["categories"].values()])
        )
        fold_metrics[str(fold)]["supervised_macro_f1"] = float(
            np.mean([value["supervised_f1"] for value in fold_metrics[str(fold)]["categories"].values()])
        )
        fold_metrics[str(fold)]["delta"] = (
            fold_metrics[str(fold)]["supervised_macro_f1"]
            - fold_metrics[str(fold)]["tfidf_macro_f1"]
        )
    corrected = int((scope & (tfidf_pred != labels) & (supervised_pred == labels)).sum())
    regressed = int((scope & (tfidf_pred == labels) & (supervised_pred != labels)).sum())
    category_metrics = {}
    for category in CATEGORIES:
        local = scope & (categories == category)
        base = f1(labels[local], tfidf_pred[local])
        candidate = f1(labels[local], supervised_pred[local])
        category_metrics[category] = {
            "tfidf_f1": base,
            "supervised_f1": candidate,
            "delta": candidate - base,
        }
    mean_delta = float(np.mean([fold_metrics[str(fold)]["delta"] for fold in folds_to_evaluate]))
    gates = {
        "each_screen_fold_positive": all(fold_metrics[str(fold)]["delta"] > 0 for fold in folds_to_evaluate),
        "mean_delta_at_least_0_0015": mean_delta >= 0.0015,
        "no_category_drop_below_minus_0_002": all(value["delta"] >= -0.002 for value in category_metrics.values()),
        "corrected_to_regressed_at_least_1_5": corrected > 0 if regressed == 0 else corrected / regressed >= 1.5,
    }
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "experiment_id": "650",
        "evaluation_version": "semantic_family_v3",
        "folds_evaluated": list(folds_to_evaluate),
        "decision_threshold": 0.0,
        "threshold_tuned": False,
        "sealed_rows": 0,
        "folds": fold_metrics,
        "categories": category_metrics,
        "mean_macro_delta": mean_delta,
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": None if regressed == 0 else corrected / regressed,
        "gates": gates,
        "passed": passed,
        "decision": "GO_FULL_EVALUATION" if passed else "NO_GO_REJECT_650",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    folds = (0, 1, 2, 3, 4) if args.full else SCREEN_FOLDS
    result = evaluate(
        runtime_path=args.runtime,
        embedding_path=args.embeddings,
        baseline_path=args.baseline,
        output_path=args.output,
        folds_to_evaluate=folds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from semantic_v3_contract import (
    DEFAULT_ALL_EMBEDDINGS,
    DEFAULT_DATA,
    DEFAULT_FIRST_EMBEDDINGS,
    DEFAULT_FOLDS,
    DEVELOPMENT_FOLDS,
    ROBUST_PROTOCOL_VERSION,
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_vectorizer() -> FeatureUnion:
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


def vectorize_inner_split(
    train_texts: np.ndarray, valid_texts: np.ndarray
) -> tuple[FeatureUnion, sparse.spmatrix, sparse.spmatrix]:
    """Fit vocabulary and IDF on one inner-train split only."""

    vectorizer = make_vectorizer()
    train_matrix = vectorizer.fit_transform(train_texts)
    valid_matrix = vectorizer.transform(valid_texts)
    return vectorizer, train_matrix, valid_matrix


def require_finite(name: str, values: np.ndarray | sparse.spmatrix) -> None:
    checked = values.data if sparse.issparse(values) else np.asarray(values)
    if not np.isfinite(checked).all():
        count = int((~np.isfinite(checked)).sum())
        raise FloatingPointError(f"{name} contains {count} non-finite values")


def embedding_stats(values: np.ndarray) -> dict[str, float | int | bool]:
    require_finite("embedding input", values)
    norms = np.linalg.norm(values, axis=1)
    require_finite("embedding norms", norms)
    return {
        "finite": True,
        "rows": len(values),
        "dimension": int(values.shape[1]),
        "min": float(values.min()),
        "max": float(values.max()),
        "norm_min": float(norms.min()),
        "norm_median": float(np.median(norms)),
        "norm_max": float(norms.max()),
        "zero_norm_rows": int((norms == 0).sum()),
        "normalization_applied": False,
    }


def fit_and_score(
    *,
    name: str,
    model: LinearSVC | ExtraTreesClassifier,
    train_feature: np.ndarray | sparse.spmatrix,
    valid_feature: np.ndarray | sparse.spmatrix,
    labels: np.ndarray,
) -> np.ndarray:
    require_finite(f"{name} train features", train_feature)
    require_finite(f"{name} validation features", valid_feature)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train_feature, labels)
        if hasattr(model, "coef_"):
            require_finite(f"{name} coefficients", model.coef_)
        if isinstance(model, ExtraTreesClassifier):
            scores = model.predict_proba(valid_feature)[:, 1]
        else:
            scores = model.decision_function(valid_feature)
    scores = np.asarray(scores, dtype=np.float32)
    require_finite(f"{name} scores", scores)
    return scores


def rank01(values: np.ndarray) -> np.ndarray:
    require_finite("rank input", values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def project_empirical_rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Project outer scores onto the inner-OOF rank scale without outer labels."""

    require_finite("empirical-rank reference", reference)
    require_finite("empirical-rank values", values)
    ordered = np.sort(np.asarray(reference, dtype=np.float64), kind="mergesort")
    if len(ordered) < 2:
        raise ValueError("rank reference needs at least two values")
    positions = np.searchsorted(ordered, values, side="right") - 1
    return np.clip(positions / (len(ordered) - 1), 0.0, 1.0).astype(np.float32)


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    fn = int(sorted_labels.sum()) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    boundary = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    candidates = np.flatnonzero(boundary)
    best = int(candidates[np.argmax(f1[candidates])])
    return float(f1[best]), float(sorted_scores[best])


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(((labels == 1) & predictions).sum())
    fp = int(((labels == 0) & predictions).sum())
    fn = int(((labels == 1) & ~predictions).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def search_fusion(labels: np.ndarray, heads: list[np.ndarray]) -> dict[str, object]:
    best: dict[str, object] | None = None
    for text_steps in range(21):
        for all_steps in range(21 - text_steps):
            for first_steps in range(21 - text_steps - all_steps):
                tree_steps = 20 - text_steps - all_steps - first_steps
                weights = (
                    np.asarray([text_steps, all_steps, first_steps, tree_steps], dtype=np.float32)
                    / 20
                )
                scores = sum(weight * head for weight, head in zip(weights, heads))
                score, threshold = best_threshold(labels, scores)
                candidate = {
                    "inner_f1": score,
                    "threshold": threshold,
                    "weights": weights.tolist(),
                }
                if best is None or score > float(best["inner_f1"]):
                    best = candidate
    if best is None:
        raise RuntimeError("fusion search produced no candidate")
    return best


def svc(c_value: float, seed: int, *, max_iter: int) -> LinearSVC:
    return LinearSVC(
        C=c_value,
        class_weight="balanced",
        dual="auto",
        max_iter=max_iter,
        random_state=seed,
    )


def trees(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=8,
        random_state=seed,
    )


def inner_oof_heads(
    *,
    texts: np.ndarray,
    all_images: np.ndarray,
    first_image: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> list[np.ndarray]:
    outputs = [np.full(len(labels), np.nan, dtype=np.float32) for _ in range(4)]
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = folds == fold
        _, train_text, valid_text = vectorize_inner_split(texts[train], texts[valid])
        models = [
            svc(1.0, 42 + int(fold), max_iter=8_000),
            svc(3.0, 42 + int(fold), max_iter=8_000),
            svc(10.0, 42 + int(fold), max_iter=10_000),
            trees(42 + int(fold)),
        ]
        train_features = [
            train_text,
            all_images[train],
            first_image[train],
            all_images[train],
        ]
        valid_features = [
            valid_text,
            all_images[valid],
            first_image[valid],
            all_images[valid],
        ]
        names = ("text", "all_images", "first_image", "extra_trees")
        for output, name, model, train_feature, valid_feature in zip(
            outputs, names, models, train_features, valid_features
        ):
            output[valid] = fit_and_score(
                name=f"inner fold {fold} {name}",
                model=model,
                train_feature=train_feature,
                valid_feature=valid_feature,
                labels=labels[train],
            )
    if any(np.isnan(output).any() for output in outputs):
        raise ValueError("inner OOF head is incomplete")
    return outputs


def fit_outer_heads(
    *,
    train_text: sparse.spmatrix,
    valid_text: sparse.spmatrix,
    train_all: np.ndarray,
    valid_all: np.ndarray,
    train_first: np.ndarray,
    valid_first: np.ndarray,
    labels: np.ndarray,
    outer_fold: int,
) -> list[np.ndarray]:
    models = [
        svc(1.0, 42 + outer_fold, max_iter=8_000),
        svc(3.0, 42 + outer_fold, max_iter=8_000),
        svc(10.0, 42 + outer_fold, max_iter=10_000),
        trees(42 + outer_fold),
    ]
    train_features = [train_text, train_all, train_first, train_all]
    valid_features = [valid_text, valid_all, valid_first, valid_all]
    names = ("text", "all_images", "first_image", "extra_trees")
    return [
        fit_and_score(
            name=f"outer fold {outer_fold} {name}",
            model=model,
            train_feature=train_feature,
            valid_feature=valid_feature,
            labels=labels,
        )
        for name, model, train_feature, valid_feature in zip(
            names, models, train_features, valid_features
        )
    ]


def train(
    *,
    data_path: Path,
    folds_path: Path,
    all_embeddings_path: Path,
    first_embeddings_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    data = pd.read_csv(data_path, dtype={"id": str})
    registry = pd.read_csv(folds_path, dtype={"id": str})
    if not np.array_equal(data["id"].astype(str).to_numpy(), registry["id"].to_numpy()):
        raise ValueError("data/folds id mismatch")
    development_mask = registry["split"].astype(str).to_numpy() == "development"
    sealed_ids = set(registry.loc[~development_mask, "id"].astype(str))
    development_positions = np.flatnonzero(development_mask)
    frame = data.iloc[development_positions].reset_index(drop=True)
    folds = registry.loc[development_mask, "development_fold"].to_numpy(np.int8)
    ids = np.asarray(frame["id"].astype(str).tolist(), dtype=str)
    if set(ids) & sealed_ids:
        raise ValueError("sealed ids entered robust-base development frame")
    labels = frame["label"].to_numpy(np.int8)
    categories = np.asarray(frame["category"].astype(str).tolist(), dtype=str)
    texts = (
        frame["name"].fillna("").astype(str)
        + "\n"
        + frame["name"].fillna("").astype(str)
        + "\n"
        + frame["description"].fillna("").astype(str)
    ).to_numpy()
    arrays = {}
    feature_audit = {}
    for name, path in (("all", all_embeddings_path), ("first", first_embeddings_path)):
        archive = np.load(path, allow_pickle=False)
        if not np.array_equal(archive["ids"].astype(str), data["id"].astype(str)):
            raise ValueError(f"{name} embedding id mismatch")
        arrays[name] = archive["embeddings"][development_positions].astype(np.float32)
        feature_audit[name] = embedding_stats(arrays[name])

    head_names = ("text", "all_images", "first_image", "extra_trees")
    output_heads = {name: np.full(len(ids), np.nan, np.float32) for name in head_names}
    robust_scores = np.full(len(ids), np.nan, np.float32)
    predictions = np.zeros(len(ids), np.int8)
    fold_reports = []
    for outer_fold in DEVELOPMENT_FOLDS:
        outer_train = folds != outer_fold
        outer_valid = folds == outer_fold
        vectorizer = make_vectorizer()
        train_text_all = vectorizer.fit_transform(texts[outer_train])
        valid_text_all = vectorizer.transform(texts[outer_valid])
        outer_train_global = np.flatnonzero(outer_train)
        outer_valid_global = np.flatnonzero(outer_valid)
        for category in sorted(np.unique(categories)):
            train_local = categories[outer_train_global] == category
            valid_local = categories[outer_valid_global] == category
            train_positions = outer_train_global[train_local]
            valid_positions = outer_valid_global[valid_local]
            train_text = train_text_all[train_local]
            valid_text = valid_text_all[valid_local]
            train_labels = labels[train_positions]
            inner_folds = folds[train_positions]
            inner_raw = inner_oof_heads(
                texts=texts[train_positions],
                all_images=arrays["all"][train_positions],
                first_image=arrays["first"][train_positions],
                labels=train_labels,
                folds=inner_folds,
            )
            inner_ranks = [rank01(values) for values in inner_raw]
            fusion = search_fusion(train_labels, inner_ranks)
            outer_raw = fit_outer_heads(
                train_text=train_text,
                valid_text=valid_text,
                train_all=arrays["all"][train_positions],
                valid_all=arrays["all"][valid_positions],
                train_first=arrays["first"][train_positions],
                valid_first=arrays["first"][valid_positions],
                labels=train_labels,
                outer_fold=outer_fold,
            )
            projected = [
                project_empirical_rank(reference, values)
                for reference, values in zip(inner_raw, outer_raw)
            ]
            weights = np.asarray(fusion["weights"], dtype=np.float32)
            fused = sum(weight * values for weight, values in zip(weights, projected))
            local_predictions = fused >= float(fusion["threshold"])
            robust_scores[valid_positions] = fused
            predictions[valid_positions] = local_predictions.astype(np.int8)
            for name, values in zip(head_names, projected):
                output_heads[name][valid_positions] = values
            fold_reports.append(
                {
                    "outer_fold": outer_fold,
                    "category": category,
                    "train_rows": len(train_positions),
                    "validation_rows": len(valid_positions),
                    "inner_folds": sorted(np.unique(inner_folds).astype(int).tolist()),
                    "weights": fusion["weights"],
                    "threshold": fusion["threshold"],
                    "inner_f1": fusion["inner_f1"],
                    "outer_f1": f1(labels[valid_positions], local_predictions),
                }
            )
    if np.isnan(robust_scores).any() or any(np.isnan(x).any() for x in output_heads.values()):
        raise ValueError("robust-base development OOF is incomplete")
    npz_path = output_dir / "robust_base_semantic_v3.npz"
    np.savez_compressed(
        npz_path,
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        semantic_components=np.asarray(
            registry.loc[development_mask, "semantic_component"].astype(str).tolist(),
            dtype=str,
        ),
        protocol_version=np.asarray(ROBUST_PROTOCOL_VERSION),
        robust_base_score=robust_scores,
        predictions=predictions,
        **output_heads,
    )
    category_f1 = {
        category: f1(labels[categories == category], predictions[categories == category])
        for category in sorted(np.unique(categories))
    }
    report = {
        "version": ROBUST_PROTOCOL_VERSION,
        "status": "completed",
        "development_rows": len(ids),
        "sealed_rows_seen_by_models_or_metrics": 0,
        "outer_folds": list(DEVELOPMENT_FOLDS),
        "projection": "empirical CDF of inner-OOF raw head scores",
        "inner_text_protocol": (
            "for every inner fold, fit vocabulary and IDF only on inner-train; "
            "transform inner-validation"
        ),
        "outer_text_protocol": "fit vocabulary and IDF only on outer-train",
        "numerical_safety": {
            "runtime_and_convergence_warnings": "fatal",
            "nonfinite_features_coefficients_or_scores": "fatal",
            "library_versions": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "input_embedding_stats": feature_audit,
        },
        "category_f1": category_f1,
        "macro_f1": float(np.mean(list(category_f1.values()))),
        "fold_category_reports": fold_reports,
        "runtime_minutes": (time.monotonic() - started) / 60,
        "input_sha256": {
            "data": sha256(data_path),
            "folds": sha256(folds_path),
            "all_embeddings": sha256(all_embeddings_path),
            "first_embeddings": sha256(first_embeddings_path),
        },
        "output_sha256": {"robust_base_semantic_v3.npz": sha256(npz_path)},
    }
    (output_dir / "robust_base_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--all-embeddings", type=Path, default=DEFAULT_ALL_EMBEDDINGS)
    parser.add_argument("--first-embeddings", type=Path, default=DEFAULT_FIRST_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train(
        data_path=args.data,
        folds_path=args.folds,
        all_embeddings_path=args.all_embeddings,
        first_embeddings_path=args.first_embeddings,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

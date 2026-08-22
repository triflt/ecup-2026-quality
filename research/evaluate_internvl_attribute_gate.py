from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/data.csv"
FOLDS = ROOT / "validation/grouped_text_v1/folds.csv"
LOCKED = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
FLAMMABLE = "Легковоспламеняющиеся"
FEATURES = [
    "fuel_or_gas",
    "empty_equipment",
    "ignition_source",
    "fuel_included",
    "combustible_material",
    "absence_or_negation",
    "locked_score",
    "locked_uncertainty",
]


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered = labels[order]
    tp = np.cumsum(ordered == 1)
    fp = np.cumsum(ordered == 0)
    fn = int((labels == 1).sum()) - tp
    values = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    best = int(np.argmax(values))
    if best + 1 == len(scores):
        return float(scores[order[best]] - 1e-7)
    return float((scores[order[best]] + scores[order[best + 1]]) / 2)


def estimator() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.3,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=30042,
        ),
    )


def category_scores(
    labels: np.ndarray,
    predictions: np.ndarray,
    categories: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    return {
        category: f1(
            labels[mask & (categories == category)],
            predictions[mask & (categories == category)],
        )
        for category in sorted(np.unique(categories))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=30042)
    args = parser.parse_args()

    locked = np.load(LOCKED, allow_pickle=False)
    ids = locked["ids"].astype(str)
    labels = locked["labels"].astype(np.int8)
    categories = locked["categories"].astype(str)
    folds = locked["folds"].astype(np.int8)
    baseline = locked["baseline_nested_predictions"].astype(np.int8)
    fold_frame = pd.read_csv(FOLDS, dtype={"id": str}).set_index("id").loc[ids]
    if not np.array_equal(fold_frame.fold.to_numpy(np.int8), folds):
        raise ValueError("fold registry mismatch")
    group_hashes = fold_frame.group_hash.astype(str).to_numpy()

    manifest = pd.read_csv(args.manifest, compression="gzip", dtype={"id": str})
    predicted = pd.read_csv(args.predictions, dtype={"id": str})
    if manifest.id.duplicated().any() or predicted.id.duplicated().any():
        raise ValueError("duplicate attribute ids")
    if set(manifest.id) != set(predicted.id):
        raise ValueError("attribute prediction ids do not match selector manifest")
    selected = manifest.merge(predicted, on=["id", "fold"], validate="one_to_one")
    selected = selected.set_index("id")
    id_to_position = {item_id: position for position, item_id in enumerate(ids)}
    selected["position"] = [id_to_position[item_id] for item_id in selected.index]
    selected = selected.sort_values("position")
    positions = selected.position.to_numpy(np.int64)
    if not np.all(categories[positions] == FLAMMABLE):
        raise ValueError("selector contains non-flammable rows")
    valid = selected.valid.astype(bool).to_numpy() & selected.image_download_ok.astype(bool).to_numpy()
    for feature in FEATURES[:6]:
        values = selected[feature].to_numpy()
        valid &= np.isin(values, [0, 1])
    matrix = selected[FEATURES].to_numpy(np.float64)
    selected_labels = labels[positions]
    selected_folds = folds[positions]

    candidate = baseline.copy()
    fold_detail = []
    for outer in sorted(np.unique(folds)):
        donor = valid & (selected_folds != outer)
        outer_rows = valid & (selected_folds == outer)
        inner_scores = np.full(len(selected), np.nan, dtype=np.float64)
        for inner in sorted(np.unique(selected_folds[donor])):
            inner_train = donor & (selected_folds != inner)
            inner_valid = donor & (selected_folds == inner)
            model = estimator()
            model.fit(matrix[inner_train], selected_labels[inner_train])
            inner_scores[inner_valid] = model.predict_proba(matrix[inner_valid])[:, 1]
        if not np.isfinite(inner_scores[donor]).all():
            raise ValueError(f"incomplete inner predictions for outer fold {outer}")
        threshold = best_threshold(selected_labels[donor], inner_scores[donor])
        model = estimator()
        model.fit(matrix[donor], selected_labels[donor])
        outer_scores = model.predict_proba(matrix[outer_rows])[:, 1]
        outer_predictions = (outer_scores >= threshold).astype(np.int8)
        candidate[positions[outer_rows]] = outer_predictions
        fold_detail.append(
            {
                "outer_fold": int(outer),
                "donor_rows": int(donor.sum()),
                "outer_rows": int(outer_rows.sum()),
                "threshold_from_inner_oof": threshold,
                "selected_outer_baseline_f1": f1(
                    selected_labels[outer_rows], baseline[positions[outer_rows]]
                ),
                "selected_outer_candidate_f1": f1(
                    selected_labels[outer_rows], outer_predictions
                ),
            }
        )

    baseline_category = category_scores(labels, baseline, categories)
    candidate_category = category_scores(labels, candidate, categories)
    baseline_macro = float(np.mean(list(baseline_category.values())))
    candidate_macro = float(np.mean(list(candidate_category.values())))
    folds_report = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        old = category_scores(labels, baseline, categories, mask)
        new = category_scores(labels, candidate, categories, mask)
        old_macro = float(np.mean(list(old.values())))
        new_macro = float(np.mean(list(new.values())))
        folds_report.append(
            {
                "fold": int(fold),
                "baseline_macro_f1": old_macro,
                "candidate_macro_f1": new_macro,
                "delta": new_macro - old_macro,
                "baseline_category_f1": old,
                "candidate_category_f1": new,
            }
        )

    group_rows: dict[str, list[np.ndarray]] = {}
    for category in sorted(np.unique(categories)):
        groups: dict[str, list[int]] = {}
        for position in np.flatnonzero(categories == category):
            groups.setdefault(group_hashes[position], []).append(position)
        group_rows[category] = [np.asarray(rows) for rows in groups.values()]
    rng = np.random.default_rng(args.seed)
    deltas = np.empty(args.bootstrap, dtype=np.float64)
    for iteration in range(args.bootstrap):
        old_values, new_values = [], []
        for category in sorted(group_rows):
            groups = group_rows[category]
            sampled = rng.integers(0, len(groups), size=len(groups))
            sample = np.concatenate([groups[index] for index in sampled])
            old_values.append(f1(labels[sample], baseline[sample]))
            new_values.append(f1(labels[sample], candidate[sample]))
        deltas[iteration] = np.mean(new_values) - np.mean(old_values)

    category_delta = {
        category: candidate_category[category] - baseline_category[category]
        for category in baseline_category
    }
    wins = sum(row["delta"] > 0 for row in folds_report)
    report = {
        "protocol": "internvl_attribute_nested_gate_v1",
        "selector_rows": int(len(selected)),
        "valid_attribute_rows": int(valid.sum()),
        "invalid_rows_keep_baseline": int((~valid).sum()),
        "features": FEATURES,
        "gate": "C=0.3 balanced logistic; threshold from four donor-fold inner OOF predictions",
        "baseline_macro_f1": baseline_macro,
        "candidate_macro_f1": candidate_macro,
        "delta_macro_f1": candidate_macro - baseline_macro,
        "baseline_category_f1": baseline_category,
        "candidate_category_f1": candidate_category,
        "category_delta": category_delta,
        "folds_won": wins,
        "folds": folds_report,
        "gate_fold_detail": fold_detail,
        "changed_predictions": int((baseline != candidate).sum()),
        "corrected": int(((baseline != labels) & (candidate == labels)).sum()),
        "regressed": int(((baseline == labels) & (candidate != labels)).sum()),
        "group_bootstrap": {
            "iterations": args.bootstrap,
            "seed": args.seed,
            "probability_delta_positive": float((deltas > 0).mean()),
            "delta_ci95": [
                float(np.quantile(deltas, 0.025)),
                float(np.quantile(deltas, 0.975)),
            ],
        },
        "downstream_prior_replay": "No flammable production family prior exists in experiment 190; BAD predictions remain byte-identical.",
        "acceptance": {
            "delta_at_least_0_003": candidate_macro - baseline_macro >= 0.003,
            "wins_at_least_4_of_5_folds": wins >= 4,
            "no_category_drop_over_0_005": min(category_delta.values()) >= -0.005,
            "bootstrap_probability_at_least_0_90": float((deltas > 0).mean()) >= 0.90,
        },
    }
    report["accepted"] = all(report["acceptance"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_npz,
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        baseline_predictions=baseline,
        candidate_predictions=candidate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

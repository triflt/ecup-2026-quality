from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "experiments/320_bad_regulatory_evidence/artifacts/bad_regulatory_selector_v1.csv.gz"
LOCKED = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
QWEN3VL = ROOT / "research/lora-hard-5fold-robust-fusion-report.npz"
QWEN35 = ROOT / "research/qwen35-hard-5fold-robust-fusion-report.npz"
CONNECTED = ROOT / "validation/connected_family_repeated_v1/rows.csv"
BAD = "БАД"
PRODUCTION_THRESHOLD = 0.27193570137023926
RIDGE_ALPHAS = (10.0, 100.0, 1000.0)
RESIDUAL_WEIGHTS = (0.10, 0.20, 0.35)
RESIDUAL_CLIP = 0.25
EXPECTED_SHA256 = {
    "selector": "d254eba1c8d14114b982beee3877c3c8e361be8c83e4de2690e44b140ca195c1",
    "locked": "176b76b0b0fd3527e9e15c8f2edd6a43f90fd3a24db1567a32a4b05e215139e8",
    "qwen3vl": "ba432e13624e6c3b1c7304ced8cacf580f4ffcc0a0cde1af8b6afb098bf6dc01",
    "qwen35": "147f2b2b87d8220566526b38ab0e085c0bc44cd82b0442baca6b019516c3f1d8",
    "connected": "4dea3dbb832edc7134120963e031b86562d1948845eb8669ac6e46d800b7a7dc",
}
IDENTITY_COLUMNS = {
    "id",
    "fold",
    "group_hash",
    "connected_component",
    "safe_for_selection",
}


@dataclass(frozen=True)
class Prepared:
    ids: np.ndarray
    labels: np.ndarray
    categories: np.ndarray
    historical_folds: np.ndarray
    baseline_historical: np.ndarray
    locked_scores: np.ndarray
    historical_thresholds: np.ndarray
    safe: np.ndarray
    selector: np.ndarray
    consistent_for_training: np.ndarray
    cohorts: dict[str, np.ndarray]
    components: np.ndarray
    group_hashes: np.ndarray
    repeat_folds: dict[str, np.ndarray]
    feature_positions: np.ndarray
    features: np.ndarray
    feature_names: list[str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def category_scores(
    labels: np.ndarray,
    predictions: np.ndarray,
    categories: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    return {
        category: f1(
            labels[mask & (categories == category)],
            predictions[mask & (categories == category)],
        )
        for category in sorted(np.unique(categories))
    }


def family_weights(labels: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Inverse connected-family size, then equal total mass for both labels."""
    _, inverse, counts = np.unique(components, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    for label in (0, 1):
        mask = labels == label
        mass = float(weights[mask].sum())
        if mass <= 0:
            raise ValueError(f"training split has no label {label}")
        weights[mask] *= 0.5 / mass
    return weights / weights.mean()


def fit_residual(
    features: np.ndarray,
    labels: np.ndarray,
    locked_scores: np.ndarray,
    components: np.ndarray,
    alpha: float,
):
    target = labels.astype(np.float32) - locked_scores.astype(np.float32)
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, solver="lsqr", tol=1e-5, max_iter=2000),
    )
    model.fit(
        features,
        target,
        ridge__sample_weight=family_weights(labels, components),
    )
    return model


def residual_score(
    locked_scores: np.ndarray,
    residuals: np.ndarray,
    weight: float,
) -> np.ndarray:
    clipped = np.clip(residuals, -RESIDUAL_CLIP, RESIDUAL_CLIP)
    return np.clip(locked_scores + weight * clipped, 0.0, 1.0)


def load_prepared() -> Prepared:
    for name, path in (
        ("selector", SELECTOR),
        ("locked", LOCKED),
        ("qwen3vl", QWEN3VL),
        ("qwen35", QWEN35),
        ("connected", CONNECTED),
    ):
        actual = sha256(path)
        if actual != EXPECTED_SHA256[name]:
            raise ValueError(
                f"frozen input {name} hash mismatch: {actual} != {EXPECTED_SHA256[name]}"
            )
    selector = pd.read_csv(SELECTOR, dtype={"id": str})
    connected = pd.read_csv(
        CONNECTED,
        dtype={"id": str, "connected_component": str, "group_hash": str},
    )
    locked = np.load(LOCKED, allow_pickle=False)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    qwen35 = np.load(QWEN35, allow_pickle=True)
    ids = locked["ids"].astype(str)
    labels = locked["labels"].astype(np.int8)
    categories = locked["categories"].astype(str)
    historical_folds = locked["folds"].astype(np.int8)
    for name, source, source_ids_key in (
        ("qwen3vl", qwen3vl, "ids"),
        ("qwen35", qwen35, "ids"),
    ):
        if not np.array_equal(source[source_ids_key].astype(str), ids):
            raise ValueError(f"{name} ids are not aligned")
        if "labels" in source.files and not np.array_equal(
            source["labels"].astype(np.int8), labels
        ):
            raise ValueError(f"{name} labels are not aligned")
        if "categories" in source.files and not np.array_equal(
            source["categories"].astype(str), categories
        ):
            raise ValueError(f"{name} categories are not aligned")
    if connected["id"].duplicated().any() or set(connected["id"]) != set(ids):
        raise ValueError("connected registry ids must be unique and complete")
    connected = connected.set_index("id").loc[ids]
    if not np.array_equal(connected["fold"].to_numpy(np.int8), historical_folds):
        raise ValueError("historical fold mismatch")
    if not np.array_equal(connected["label"].to_numpy(np.int8), labels):
        raise ValueError("connected registry label mismatch")
    if not np.array_equal(connected["category"].astype(str).to_numpy(), categories):
        raise ValueError("connected registry category mismatch")
    for name, source in (("qwen3vl", qwen3vl), ("qwen35", qwen35)):
        if not np.array_equal(source["folds"].astype(np.int8), historical_folds):
            raise ValueError(f"{name} fold mismatch")
    id_to_position = {item: position for position, item in enumerate(ids)}
    if selector["id"].duplicated().any() or not set(selector["id"]).issubset(id_to_position):
        raise ValueError("invalid selector ids")
    feature_positions = selector["id"].map(id_to_position).to_numpy(np.int64)
    safe = connected["safe_for_selection"].to_numpy(bool)
    selector_mask = np.zeros(len(ids), dtype=bool)
    selector_mask[feature_positions] = True
    if not np.array_equal(selector["safe_for_selection"].to_numpy(bool), safe[feature_positions]):
        raise ValueError("selector safe mask mismatch")

    numeric_names = [column for column in selector.columns if column not in IDENTITY_COLUMNS]
    if any("label" in column.lower() or "prediction" in column.lower() for column in numeric_names):
        raise ValueError("selector features contain a forbidden target-derived column")
    numeric = selector[numeric_names].copy()
    count_columns = [column for column in numeric if column.startswith("count_")]
    numeric[count_columns] = np.log1p(numeric[count_columns].astype(np.float32))
    numeric_values = numeric.astype(np.float32).to_numpy()
    features = numeric_values.astype(np.float32)
    feature_names = numeric_names

    locked_scores = (
        0.50 * qwen35["base_rank"].astype(np.float32)
        + 0.25 * qwen3vl["lora_rank"].astype(np.float32)
        + 0.25 * qwen35["lora_rank"].astype(np.float32)
    )
    if not np.allclose(
        selector["locked_score"].to_numpy(np.float32),
        locked_scores[feature_positions],
        rtol=0.0,
        atol=2e-7,
    ):
        raise ValueError("frozen selector locked_score mismatch")
    if not np.allclose(
        selector["locked_uncertainty"].to_numpy(np.float32),
        np.abs(locked_scores[feature_positions] - PRODUCTION_THRESHOLD),
        rtol=0.0,
        atol=2e-7,
    ):
        raise ValueError("frozen selector locked_uncertainty mismatch")
    baseline_historical = locked["baseline_nested_predictions"].astype(np.int8)
    historical_thresholds = np.full(len(ids), PRODUCTION_THRESHOLD, dtype=np.float32)
    for fold in sorted(np.unique(historical_folds)):
        donors = (categories == BAD) & (historical_folds != fold)
        historical_thresholds[historical_folds == fold] = best_threshold(
            labels[donors], locked_scores[donors]
        )
    derived_bad = (locked_scores >= historical_thresholds).astype(np.int8)
    if np.any(
        (categories == BAD) & (derived_bad != baseline_historical)
    ):
        raise ValueError("fixed Public-190 threshold no longer reproduces historical BAD OOF")
    cohorts = {}
    for cohort in (
        "sports_nutrition",
        "medicine_language",
        "food_beverage",
        "pet_veterinary",
    ):
        mask = np.zeros(len(ids), dtype=bool)
        mask[feature_positions] = selector[cohort].to_numpy(bool)
        cohorts[cohort] = mask
    component_label_count = (
        pd.DataFrame({"component": connected["connected_component"].astype(str), "label": labels})
        .groupby("component")["label"]
        .nunique()
    )
    mixed_components = set(component_label_count[component_label_count > 1].index)
    consistent_for_training = ~connected["connected_component"].astype(str).isin(
        mixed_components
    ).to_numpy()
    repeat_folds = {
        f"repeat_{index}": connected[f"repeat_{index}_fold"].to_numpy(np.int8)
        for index in range(3)
    }
    return Prepared(
        ids=ids,
        labels=labels,
        categories=categories,
        historical_folds=historical_folds,
        baseline_historical=baseline_historical,
        locked_scores=locked_scores,
        historical_thresholds=historical_thresholds,
        safe=safe,
        selector=selector_mask,
        consistent_for_training=consistent_for_training,
        cohorts=cohorts,
        components=connected["connected_component"].astype(str).to_numpy(),
        group_hashes=connected["group_hash"].astype(str).to_numpy(),
        repeat_folds=repeat_folds,
        feature_positions=feature_positions,
        features=features,
        feature_names=feature_names,
    )


def feature_rows(prepared: Prepared, full_positions: np.ndarray) -> np.ndarray:
    mapping = np.full(len(prepared.ids), -1, dtype=np.int64)
    mapping[prepared.feature_positions] = np.arange(len(prepared.feature_positions))
    rows = mapping[full_positions]
    if np.any(rows < 0):
        raise ValueError("requested a feature row outside selector")
    return rows


def select_hyperparameters(
    prepared: Prepared,
    folds: np.ndarray,
    outer_fold: int,
    decision_thresholds: np.ndarray,
) -> tuple[float, float, list[dict[str, float]]]:
    donor = prepared.safe & (prepared.categories == BAD) & (folds >= 0) & (folds != outer_fold)
    trainable = donor & prepared.selector
    fit_eligible = trainable & prepared.consistent_for_training
    residual_by_alpha: dict[float, np.ndarray] = {}
    for alpha in RIDGE_ALPHAS:
        predictions = np.full(len(prepared.ids), np.nan, dtype=np.float32)
        for inner_fold in sorted(np.unique(folds[donor])):
            inner_valid = trainable & (folds == inner_fold)
            inner_train = fit_eligible & (folds != inner_fold)
            train_positions = np.flatnonzero(inner_train)
            valid_positions = np.flatnonzero(inner_valid)
            if not len(valid_positions):
                continue
            model = fit_residual(
                prepared.features[feature_rows(prepared, train_positions)],
                prepared.labels[train_positions],
                prepared.locked_scores[train_positions],
                prepared.components[train_positions],
                alpha,
            )
            predictions[valid_positions] = model.predict(
                prepared.features[feature_rows(prepared, valid_positions)]
            ).astype(np.float32)
        if np.isnan(predictions[trainable]).any():
            raise ValueError("inner OOF residual prediction is incomplete")
        residual_by_alpha[alpha] = predictions

    audit = []
    baseline = (prepared.locked_scores >= decision_thresholds).astype(np.int8)
    for alpha in RIDGE_ALPHAS:
        for weight in RESIDUAL_WEIGHTS:
            candidate = baseline.copy()
            candidate[trainable] = (
                residual_score(
                    prepared.locked_scores[trainable],
                    residual_by_alpha[alpha][trainable],
                    weight,
                )
                >= decision_thresholds[trainable]
            ).astype(np.int8)
            fold_deltas = []
            for inner_fold in sorted(np.unique(folds[donor])):
                mask = donor & (folds == inner_fold)
                fold_deltas.append(
                    f1(prepared.labels[mask], candidate[mask])
                    - f1(prepared.labels[mask], baseline[mask])
                )
            audit.append(
                {
                    "ridge_alpha": alpha,
                    "residual_weight": weight,
                    "mean_inner_fold_bad_delta": float(np.mean(fold_deltas)),
                    "inner_folds_won": int(sum(delta > 0 for delta in fold_deltas)),
                }
            )
    # Deterministic conservative tie-break: mean delta, fold wins, stronger
    # regularization, then smaller residual shift.
    chosen = max(
        audit,
        key=lambda row: (
            row["mean_inner_fold_bad_delta"],
            row["inner_folds_won"],
            row["ridge_alpha"],
            -row["residual_weight"],
        ),
    )
    return float(chosen["ridge_alpha"]), float(chosen["residual_weight"]), audit


def crossfit_topology(
    prepared: Prepared,
    folds: np.ndarray,
    topology: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    valid = folds >= 0
    decision_thresholds = (
        prepared.historical_thresholds
        if topology == "historical"
        else np.full(len(prepared.ids), PRODUCTION_THRESHOLD, dtype=np.float32)
    )
    # The expensive 190 component is not retrained for cheap topology repeats.
    # For repeat topologies BAD uses the frozen production threshold; this avoids
    # carrying label-derived historical thresholds across a new outer split.
    baseline = prepared.baseline_historical.copy()
    if topology != "historical":
        repeat_bad = valid & (prepared.categories == BAD)
        baseline[repeat_bad] = (
            prepared.locked_scores[repeat_bad] >= decision_thresholds[repeat_bad]
        ).astype(np.int8)
    candidate = baseline.copy()
    fold_reports: list[dict[str, object]] = []
    for outer_fold in sorted(np.unique(folds[valid])):
        alpha, weight, inner_audit = select_hyperparameters(
            prepared, folds, int(outer_fold), decision_thresholds
        )
        train = (
            prepared.safe
            & prepared.selector
            & prepared.consistent_for_training
            & (prepared.categories == BAD)
            & valid
            & (folds != outer_fold)
        )
        outer = (
            prepared.safe
            & prepared.selector
            & (prepared.categories == BAD)
            & (folds == outer_fold)
        )
        train_positions = np.flatnonzero(train)
        outer_positions = np.flatnonzero(outer)
        model = fit_residual(
            prepared.features[feature_rows(prepared, train_positions)],
            prepared.labels[train_positions],
            prepared.locked_scores[train_positions],
            prepared.components[train_positions],
            alpha,
        )
        residuals = model.predict(
            prepared.features[feature_rows(prepared, outer_positions)]
        ).astype(np.float32)
        candidate[outer_positions] = (
            residual_score(prepared.locked_scores[outer_positions], residuals, weight)
            >= decision_thresholds[outer_positions]
        ).astype(np.int8)
        fold_mask = valid & (folds == outer_fold)
        old = category_scores(
            prepared.labels, baseline, prepared.categories, fold_mask
        )
        new = category_scores(
            prepared.labels, candidate, prepared.categories, fold_mask
        )
        fold_reports.append(
            {
                "fold": int(outer_fold),
                "ridge_alpha": alpha,
                "residual_weight": weight,
                "baseline_category_f1": old,
                "candidate_category_f1": new,
                "macro_delta": float(np.mean(list(new.values())) - np.mean(list(old.values()))),
                "inner_grid": inner_audit,
            }
        )
    if np.any(candidate[~prepared.selector] != baseline[~prepared.selector]):
        raise ValueError("non-selector prediction changed")
    if np.any(candidate[~prepared.safe] != baseline[~prepared.safe]):
        raise ValueError("unsafe connected-component prediction changed")
    if np.any(candidate[prepared.categories != BAD] != baseline[prepared.categories != BAD]):
        raise ValueError("flammable prediction changed")
    return baseline, candidate, fold_reports


def order_permutation_stability(
    prepared: Prepared,
    folds: np.ndarray,
    baseline: np.ndarray,
    reference_candidate: np.ndarray,
    fold_reports: list[dict[str, object]],
    seed: int = 31415,
) -> dict[str, object]:
    """Refit best/worst folds after a fixed donor-row permutation.

    Ridge/LSQR is deterministic; this is a numerical and data-order stability
    check replacing a stochastic-seed repeat that is inapplicable to this head.
    """
    chosen = [
        min(fold_reports, key=lambda row: row["macro_delta"]),
        max(fold_reports, key=lambda row: row["macro_delta"]),
    ]
    rng = np.random.default_rng(seed)
    reports = []
    for fold_report in chosen:
        outer_fold = int(fold_report["fold"])
        train = (
            prepared.safe
            & prepared.selector
            & prepared.consistent_for_training
            & (prepared.categories == BAD)
            & (folds != outer_fold)
        )
        outer = (
            prepared.safe
            & prepared.selector
            & (prepared.categories == BAD)
            & (folds == outer_fold)
        )
        train_positions = np.flatnonzero(train)
        train_positions = train_positions[rng.permutation(len(train_positions))]
        outer_positions = np.flatnonzero(outer)
        model = fit_residual(
            prepared.features[feature_rows(prepared, train_positions)],
            prepared.labels[train_positions],
            prepared.locked_scores[train_positions],
            prepared.components[train_positions],
            float(fold_report["ridge_alpha"]),
        )
        residuals = model.predict(
            prepared.features[feature_rows(prepared, outer_positions)]
        ).astype(np.float32)
        repeated = baseline.copy()
        repeated[outer_positions] = (
            residual_score(
                prepared.locked_scores[outer_positions],
                residuals,
                float(fold_report["residual_weight"]),
            )
            >= prepared.historical_thresholds[outer_positions]
        ).astype(np.int8)
        fold_mask = folds == outer_fold
        safe_fold_mask = fold_mask & prepared.safe
        old_full = category_scores(
            prepared.labels, baseline, prepared.categories, fold_mask
        )
        new_full = category_scores(
            prepared.labels, repeated, prepared.categories, fold_mask
        )
        old_safe = category_scores(
            prepared.labels, baseline, prepared.categories, safe_fold_mask
        )
        new_safe = category_scores(
            prepared.labels, repeated, prepared.categories, safe_fold_mask
        )
        reports.append(
            {
                "fold": outer_fold,
                "reference_prediction_mismatches": int(
                    (repeated[outer_positions] != reference_candidate[outer_positions]).sum()
                ),
                "full_macro_delta": float(
                    np.mean(list(new_full.values())) - np.mean(list(old_full.values()))
                ),
                "safe_macro_delta": float(
                    np.mean(list(new_safe.values())) - np.mean(list(old_safe.values()))
                ),
                "full_category_delta": {
                    key: new_full[key] - old_full[key] for key in old_full
                },
                "safe_category_delta": {
                    key: new_safe[key] - old_safe[key] for key in old_safe
                },
            }
        )
    return {
        "kind": "deterministic_order_permutation_stability",
        "seed": seed,
        "folds": reports,
        "positive_mean_full_delta": float(
            np.mean([row["full_macro_delta"] for row in reports])
        )
        > 0,
        "positive_mean_safe_delta": float(
            np.mean([row["safe_macro_delta"] for row in reports])
        )
        > 0,
        "no_category_drop_over_0_005": all(
            min(row[scope].values()) >= -0.005
            for row in reports
            for scope in ("full_category_delta", "safe_category_delta")
        ),
        "reference_prediction_mismatches_zero": all(
            row["reference_prediction_mismatches"] == 0 for row in reports
        ),
    }


def topology_audit(
    prepared: Prepared,
    folds: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    fold_reports: list[dict[str, object]],
    mask_override: np.ndarray | None = None,
) -> dict[str, object]:
    mask = folds >= 0 if mask_override is None else (folds >= 0) & mask_override
    old = category_scores(prepared.labels, baseline, prepared.categories, mask)
    new = category_scores(prepared.labels, candidate, prepared.categories, mask)
    changed = mask & (baseline != candidate)
    corrected = changed & (baseline != prepared.labels) & (candidate == prepared.labels)
    regressed = changed & (baseline == prepared.labels) & (candidate != prepared.labels)
    category_delta = {category: new[category] - old[category] for category in old}
    macro_delta = float(np.mean(list(new.values())) - np.mean(list(old.values())))
    masked_folds = []
    for fold in sorted(np.unique(folds[mask])):
        fold_mask = mask & (folds == fold)
        fold_old = category_scores(
            prepared.labels, baseline, prepared.categories, fold_mask
        )
        fold_new = category_scores(
            prepared.labels, candidate, prepared.categories, fold_mask
        )
        masked_folds.append(
            {
                "fold": int(fold),
                "baseline_category_f1": fold_old,
                "candidate_category_f1": fold_new,
                "macro_delta": float(
                    np.mean(list(fold_new.values()))
                    - np.mean(list(fold_old.values()))
                ),
            }
        )
    cohort_audits = {}
    for name, cohort in prepared.cohorts.items():
        cohort_mask = mask & cohort & (prepared.categories == BAD)
        old_f1 = f1(prepared.labels[cohort_mask], baseline[cohort_mask])
        new_f1 = f1(prepared.labels[cohort_mask], candidate[cohort_mask])
        cohort_audits[name] = {
            "rows": int(cohort_mask.sum()),
            "baseline_f1": old_f1,
            "candidate_f1": new_f1,
            "delta_f1": new_f1 - old_f1,
            "corrected": int((corrected & cohort_mask).sum()),
            "regressed": int((regressed & cohort_mask).sum()),
        }
    context_mask = mask & (prepared.categories == BAD) & ~prepared.cohorts[
        "sports_nutrition"
    ] & (
        prepared.cohorts["medicine_language"]
        | prepared.cohorts["food_beverage"]
        | prepared.cohorts["pet_veterinary"]
    )
    context_old_f1 = f1(prepared.labels[context_mask], baseline[context_mask])
    context_new_f1 = f1(prepared.labels[context_mask], candidate[context_mask])
    return {
        "rows": int(mask.sum()),
        "baseline_category_f1": old,
        "candidate_category_f1": new,
        "baseline_macro_f1": float(np.mean(list(old.values()))),
        "candidate_macro_f1": float(np.mean(list(new.values()))),
        "delta_macro_f1": macro_delta,
        "category_delta": category_delta,
        "folds_won": int(sum(row["macro_delta"] > 0 for row in masked_folds)),
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "regressed": int(regressed.sum()),
        "corrected_to_regressed": (
            float(corrected.sum() / regressed.sum()) if regressed.any() else None
        ),
        "cohorts": cohort_audits,
        "non_sports_regulatory_context": {
            "rows": int(context_mask.sum()),
            "baseline_f1": context_old_f1,
            "candidate_f1": context_new_f1,
            "delta_f1": context_new_f1 - context_old_f1,
            "corrected": int((corrected & context_mask).sum()),
            "regressed": int((regressed & context_mask).sum()),
        },
        "folds": masked_folds,
        "selection_details": fold_reports,
        "structural_invariants": {
            "changed_outside_selector": int((changed & ~prepared.selector).sum()),
            "changed_unsafe": int((changed & ~prepared.safe).sum()),
            "changed_flammable": int((changed & (prepared.categories != BAD)).sum()),
        },
    }


def paired_component_bootstrap(
    prepared: Prepared,
    baseline: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    units: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    grouped: dict[str, list[np.ndarray]] = {}
    for category in sorted(np.unique(prepared.categories)):
        positions = np.flatnonzero(mask & (prepared.categories == category))
        rows: dict[str, list[int]] = {}
        for position in positions:
            rows.setdefault(str(units[position]), []).append(int(position))
        grouped[category] = [np.asarray(value, dtype=np.int64) for value in rows.values()]
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        old_values, new_values = [], []
        for category in sorted(grouped):
            category_units = grouped[category]
            sampled = rng.integers(0, len(category_units), size=len(category_units))
            positions = np.concatenate([category_units[index] for index in sampled])
            old_values.append(f1(prepared.labels[positions], baseline[positions]))
            new_values.append(f1(prepared.labels[positions], candidate[positions]))
        deltas[iteration] = float(np.mean(new_values) - np.mean(old_values))
    return {
        "iterations": iterations,
        "seed": seed,
        "probability_delta_positive": float((deltas > 0).mean()),
        "delta_mean": float(deltas.mean()),
        "delta_ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["historical"],
        choices=["historical", "repeat_0", "repeat_1", "repeat_2"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if len(set(args.topologies)) != len(args.topologies):
        raise ValueError("topologies must be unique")
    started = time.monotonic()
    prepared = load_prepared()
    report: dict[str, object] = {
        "experiment_id": "320",
        "evaluation_version": "component_transfer_gate_v4",
        "recipe_frozen_before_310_result": True,
        "model": "family-weighted linear ridge residual",
        "ridge_alphas": list(RIDGE_ALPHAS),
        "residual_weights": list(RESIDUAL_WEIGHTS),
        "residual_clip": RESIDUAL_CLIP,
        "production_threshold": PRODUCTION_THRESHOLD,
        "historical_locked_thresholds": sorted(
            {float(value) for value in prepared.historical_thresholds}
        ),
        "candidate_specific_threshold_tuning": False,
        "production_refit_rule": "Use the modal (ridge_alpha, residual_weight) selected independently by the five nested inner-fold searches; ties prefer larger ridge_alpha then smaller residual_weight. Do not use outer scores.",
        "feature_count": len(prepared.feature_names),
        "feature_names": prepared.feature_names,
        "topologies": {},
        "input_sha256": {
            "selector": sha256(SELECTOR),
            "locked": sha256(LOCKED),
            "qwen3vl": sha256(QWEN3VL),
            "qwen35": sha256(QWEN35),
            "connected": sha256(CONNECTED),
        },
    }
    arrays: dict[str, np.ndarray] = {
        "ids": prepared.ids,
        "labels": prepared.labels,
        "categories": prepared.categories,
        "historical_folds": prepared.historical_folds,
    }
    for topology in args.topologies:
        folds = (
            prepared.historical_folds
            if topology == "historical"
            else prepared.repeat_folds[topology]
        )
        baseline, candidate, fold_reports = crossfit_topology(
            prepared, folds, topology
        )
        audit = topology_audit(
            prepared, folds, baseline, candidate, fold_reports
        )
        if topology == "historical":
            audit["group_bootstrap"] = paired_component_bootstrap(
                prepared,
                baseline,
                candidate,
                folds >= 0,
                prepared.group_hashes,
                args.bootstrap,
                32042,
            )
        report["topologies"][topology] = audit
        if topology == "historical":
            safe_audit = topology_audit(
                prepared,
                folds,
                baseline,
                candidate,
                fold_reports,
                mask_override=prepared.safe,
            )
            safe_audit["component_bootstrap"] = paired_component_bootstrap(
                prepared,
                baseline,
                candidate,
                prepared.safe,
                prepared.components,
                args.bootstrap,
                32142,
            )
            report["historical_connected_safe"] = safe_audit
            report["deterministic_order_stability"] = order_permutation_stability(
                prepared,
                folds,
                baseline,
                candidate,
                fold_reports,
                seed=31415,
            )
            counts: dict[tuple[float, float], int] = {}
            for row in fold_reports:
                key = (float(row["ridge_alpha"]), float(row["residual_weight"]))
                counts[key] = counts.get(key, 0) + 1
            production_config = max(
                counts,
                key=lambda key: (counts[key], key[0], -key[1]),
            )
            report["predeclared_production_config"] = {
                "ridge_alpha": production_config[0],
                "residual_weight": production_config[1],
                "outer_inner_selection_votes": {
                    f"alpha={key[0]},weight={key[1]}": value
                    for key, value in sorted(counts.items())
                },
                "outer_scores_used_for_choice": False,
            }
        arrays[f"{topology}_folds"] = folds
        arrays[f"{topology}_baseline_predictions"] = baseline
        arrays[f"{topology}_candidate_predictions"] = candidate
        if topology == "historical":
            arrays["folds"] = folds
            arrays["baseline_nested_predictions"] = baseline
            arrays["exp320_nested_predictions"] = candidate
    if set(args.topologies) == {"historical", "repeat_0", "repeat_1", "repeat_2"}:
        historical = report["topologies"]["historical"]
        connected_safe = report["historical_connected_safe"]
        repeats = [report["topologies"][f"repeat_{index}"] for index in range(3)]
        repeat_mean_delta = float(
            np.mean([item["delta_macro_f1"] for item in repeats])
        )
        repeat_wins = int(sum(item["folds_won"] for item in repeats))
        report["acceptance"] = {
            "historical_delta_at_least_0_003": historical["delta_macro_f1"] >= 0.003,
            "historical_wins_at_least_4_of_5": historical["folds_won"] >= 4,
            "historical_no_category_drop_over_0_005": min(
                historical["category_delta"].values()
            )
            >= -0.005,
            "historical_bootstrap_probability_at_least_0_90": historical[
                "group_bootstrap"
            ]["probability_delta_positive"]
            >= 0.90,
            "connected_delta_at_least_0_003": connected_safe["delta_macro_f1"]
            >= 0.003,
            "connected_wins_at_least_4_of_5": connected_safe["folds_won"] >= 4,
            "connected_no_category_drop_over_0_005": min(
                connected_safe["category_delta"].values()
            )
            >= -0.005,
            "connected_bootstrap_probability_at_least_0_90": connected_safe[
                "component_bootstrap"
            ]["probability_delta_positive"]
            >= 0.90,
            "all_repeat_deltas_positive": all(
                item["delta_macro_f1"] > 0 for item in repeats
            ),
            "repeat_mean_delta_at_least_0_003": repeat_mean_delta >= 0.003,
            "repeat_wins_at_least_11_of_15": repeat_wins >= 11,
            "bad_delta_at_least_0_006": historical["category_delta"][BAD] >= 0.006,
            "sports_delta_at_least_0_01": (
                historical["cohorts"]["sports_nutrition"]["delta_f1"]
            )
            >= 0.01,
            "non_sports_context_drop_at_most_0_005": historical[
                "non_sports_regulatory_context"
            ]["delta_f1"]
            >= -0.005,
            "no_context_cohort_net_regression": all(
                historical["cohorts"][name]["corrected"]
                >= historical["cohorts"][name]["regressed"]
                for name in ("medicine_language", "food_beverage", "pet_veterinary")
            ),
            "corrected_to_regressed_at_least_2": (
                historical["regressed"] == 0
                or historical["corrected"] / historical["regressed"] >= 2.0
            ),
            "structural_invariants": all(
                value == 0
                for value in historical["structural_invariants"].values()
            ),
            "order_stability_positive_full_and_safe": report[
                "deterministic_order_stability"
            ]["positive_mean_full_delta"]
            and report["deterministic_order_stability"]["positive_mean_safe_delta"],
            "order_stability_no_category_drop_over_0_005": report[
                "deterministic_order_stability"
            ]["no_category_drop_over_0_005"],
            "order_stability_reference_predictions_identical": report[
                "deterministic_order_stability"
            ]["reference_prediction_mismatches_zero"],
            "repeat_mean_delta": repeat_mean_delta,
            "repeat_fold_wins": repeat_wins,
        }
        report[
            "accepted_before_downstream_priors_runtime_schema_and_final_refit"
        ] = all(
            value
            for key, value in report["acceptance"].items()
            if key not in {"repeat_mean_delta", "repeat_fold_wins"}
        )
    report["runtime_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

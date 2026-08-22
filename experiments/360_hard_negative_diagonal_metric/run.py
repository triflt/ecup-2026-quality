from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS = ROOT / "research/first-image-artifacts/extracted/train_embeddings_fp16.npz"
LOCKED = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
OUT = Path(__file__).resolve().parent / "results"

TOP_DIMENSIONS = 256
CONFIDENCE_QUANTILE = 0.98
PAIR_C = 0.1
SEED = 36042


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def family_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    components: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    names, inverse = np.unique(components[positions], return_inverse=True)
    sums = np.zeros((len(names), embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(len(names), dtype=np.int32)
    minima = np.ones(len(names), dtype=np.int8)
    maxima = np.zeros(len(names), dtype=np.int8)
    np.add.at(sums, inverse, embeddings[positions])
    np.add.at(counts, inverse, 1)
    np.minimum.at(minima, inverse, labels[positions])
    np.maximum.at(maxima, inverse, labels[positions])
    consistent = minima == maxima
    prototypes = normalize_rows(sums[consistent] / counts[consistent, None])
    return prototypes, minima[consistent], names[consistent], {
        "families_total": int(len(names)),
        "families_consistent": int(consistent.sum()),
        "families_conflicting_excluded": int((~consistent).sum()),
    }


def mine_pairs(prototypes: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_features = []
    pair_labels = []
    anchor_labels = []
    for start in range(0, len(prototypes), 256):
        block = prototypes[start : start + 256]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            similarity = block @ prototypes.T
        if not np.isfinite(similarity).all():
            raise ValueError("non-finite mining similarity")
        for local, anchor in enumerate(range(start, min(start + len(block), len(prototypes)))):
            same = labels == labels[anchor]
            same[anchor] = False
            opposite = labels != labels[anchor]
            if not same.any() or not opposite.any():
                continue
            positive = np.flatnonzero(same)[np.argmax(similarity[local, same])]
            negative = np.flatnonzero(opposite)[np.argmax(similarity[local, opposite])]
            pair_features.append(np.abs(prototypes[anchor] - prototypes[positive]))
            pair_labels.append(1)
            anchor_labels.append(labels[anchor])
            pair_features.append(np.abs(prototypes[anchor] - prototypes[negative]))
            pair_labels.append(0)
            anchor_labels.append(labels[anchor])
    return (
        np.ascontiguousarray(pair_features, dtype=np.float32),
        np.asarray(pair_labels, dtype=np.int8),
        np.asarray(anchor_labels, dtype=np.int8),
    )


def fit_metric(prototypes: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    pair_x, pair_y, anchor_y = mine_pairs(prototypes, labels)
    anchor_counts = np.bincount(anchor_y, minlength=2)
    pair_counts = np.bincount(pair_y, minlength=2)
    weights = np.asarray([
        len(anchor_y) / (2 * anchor_counts[a]) * len(pair_y) / (2 * pair_counts[p])
        for a, p in zip(anchor_y, pair_y)
    ], dtype=np.float64)
    model = LogisticRegression(
        C=PAIR_C, class_weight=None, solver="liblinear", max_iter=2000,
        random_state=SEED,
    )
    model.fit(pair_x, pair_y, sample_weight=weights)
    metric_weight = np.maximum(-model.coef_[0], 0.0)
    if np.count_nonzero(metric_weight) < TOP_DIMENSIONS:
        selected = np.argsort(model.coef_[0])[:TOP_DIMENSIONS]
        metric_weight = np.maximum(-model.coef_[0, selected], 1e-8)
    else:
        selected = np.argsort(metric_weight)[-TOP_DIMENSIONS:]
        metric_weight = metric_weight[selected]
    scale = np.sqrt(metric_weight / max(float(metric_weight.mean()), 1e-12)).astype(np.float32)
    return selected.astype(np.int32), scale, {
        "pairs": int(len(pair_y)),
        "same_pairs": int((pair_y == 1).sum()),
        "opposite_pairs": int((pair_y == 0).sum()),
        "nonzero_metric_dimensions": int(np.count_nonzero(metric_weight)),
        "pair_training_accuracy": float(model.score(pair_x, pair_y, sample_weight=weights)),
    }


def metric_score(
    query: np.ndarray,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
    selected: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    query = normalize_rows(query[:, selected] * scale[None, :])
    prototypes = normalize_rows(prototypes[:, selected] * scale[None, :])
    positive = prototypes[prototype_labels == 1]
    negative = prototypes[prototype_labels == 0]
    result = np.empty(len(query), dtype=np.float32)
    for start in range(0, len(query), 256):
        block = query[start : start + 256]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pos = block @ positive.T
            neg = block @ negative.T
        if not np.isfinite(pos).all() or not np.isfinite(neg).all():
            raise ValueError("non-finite metric similarity")
        pos_top = np.partition(pos, -3, axis=1)[:, -3:].mean(axis=1)
        neg_top = np.partition(neg, -3, axis=1)[:, -3:].mean(axis=1)
        result[start : start + len(block)] = pos_top - neg_top
    return result


def bootstrap(
    labels: np.ndarray, categories: np.ndarray, baseline: np.ndarray,
    candidate: np.ndarray, safe: np.ndarray, components: np.ndarray,
) -> dict[str, object]:
    groups: dict[str, list[np.ndarray]] = {}
    for category in sorted(np.unique(categories)):
        local: dict[str, list[int]] = {}
        for position in np.flatnonzero(safe & (categories == category)):
            local.setdefault(components[position], []).append(position)
        groups[category] = [np.asarray(rows) for rows in local.values()]
    rng = np.random.default_rng(SEED)
    deltas = np.empty(5000, dtype=np.float64)
    for iteration in range(len(deltas)):
        old, new = [], []
        for category in sorted(groups):
            local = groups[category]
            chosen = rng.integers(0, len(local), size=len(local))
            positions = np.concatenate([local[index] for index in chosen])
            old.append(f1(labels[positions], baseline[positions]))
            new.append(f1(labels[positions], candidate[positions]))
        deltas[iteration] = np.mean(new) - np.mean(old)
    return {
        "iterations": len(deltas), "seed": SEED,
        "probability_delta_positive": float((deltas > 0).mean()),
        "delta_mean": float(deltas.mean()),
        "delta_ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
    }


def main() -> None:
    started = time.monotonic()
    archive = np.load(EMBEDDINGS, allow_pickle=False)
    locked = np.load(LOCKED, allow_pickle=False)
    ids = locked["ids"].astype(str)
    labels = locked["labels"].astype(np.int8)
    categories = locked["categories"].astype(str)
    folds = locked["folds"].astype(np.int8)
    baseline = locked["baseline_nested_predictions"].astype(np.int8)
    if not np.array_equal(archive["ids"].astype(str), ids):
        raise ValueError("embedding id mismatch")
    if not np.array_equal(archive["labels"].astype(np.int8), labels):
        raise ValueError("embedding label mismatch")
    if not np.array_equal(archive["categories"].astype(str), categories):
        raise ValueError("embedding category mismatch")
    embeddings = normalize_rows(archive["embeddings"].astype(np.float32))
    guard = pd.read_csv(GUARD, dtype={"id": str}).set_index("id").loc[ids]
    safe = guard.safe_for_selection.to_numpy(bool)
    components = guard.connected_component.astype(str).to_numpy()
    candidate = baseline.copy()
    scores = np.full(len(labels), np.nan, dtype=np.float32)
    fold_models = []

    for category in sorted(np.unique(categories)):
        for outer_fold in sorted(np.unique(folds[safe])):
            donor = safe & (categories == category) & (folds != outer_fold)
            outer = safe & (categories == category) & (folds == outer_fold)
            prototypes, prototype_labels, prototype_names, family_report = family_prototypes(
                embeddings, labels, components, np.flatnonzero(donor)
            )
            selected, scale, metric_report = fit_metric(prototypes, prototype_labels)
            local_score = metric_score(embeddings[outer], prototypes, prototype_labels, selected, scale)
            scores[outer] = local_score
            confidence = np.abs(local_score)
            threshold = float(np.quantile(confidence, CONFIDENCE_QUANTILE))
            metric_prediction = local_score >= 0
            local_baseline = baseline[outer]
            change = (metric_prediction != local_baseline) & (confidence >= threshold)
            positions = np.flatnonzero(outer)
            candidate[positions[change]] = metric_prediction[change]
            fold_models.append({
                "category": category, "fold": int(outer_fold),
                "outer_rows": int(outer.sum()), "changes": int(change.sum()),
                "confidence_threshold": threshold,
                "selected_dimension_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
                "prototype_names_sha256": hashlib.sha256("\n".join(prototype_names).encode()).hexdigest(),
                **family_report, **metric_report,
            })
            print(f"category={category} fold={outer_fold} changes={change.sum()}", flush=True)

    category_old, category_new = {}, {}
    for category in sorted(np.unique(categories)):
        mask = safe & (categories == category)
        category_old[category] = f1(labels[mask], baseline[mask])
        category_new[category] = f1(labels[mask], candidate[mask])
    old_macro = float(np.mean(list(category_old.values())))
    new_macro = float(np.mean(list(category_new.values())))
    fold_rows = []
    for fold in sorted(np.unique(folds[safe])):
        old, new = [], []
        for category in sorted(np.unique(categories)):
            mask = safe & (categories == category) & (folds == fold)
            old.append(f1(labels[mask], baseline[mask]))
            new.append(f1(labels[mask], candidate[mask]))
        fold_rows.append({
            "fold": int(fold), "baseline_macro_f1": float(np.mean(old)),
            "candidate_macro_f1": float(np.mean(new)),
            "delta": float(np.mean(new) - np.mean(old)),
        })
    changed = safe & (baseline != candidate)
    corrected = changed & (baseline != labels) & (candidate == labels)
    regressed = changed & (baseline == labels) & (candidate != labels)
    category_delta = {key: category_new[key] - category_old[key] for key in category_old}
    audit = {
        "experiment_id": "360",
        "evaluation_version": "connected_hard_negative_metric_screen_v1",
        "recipe_frozen_before_scoring": True,
        "rows": int(safe.sum()),
        "baseline_category_f1": category_old,
        "candidate_category_f1": category_new,
        "baseline_macro_f1": old_macro,
        "candidate_macro_f1": new_macro,
        "delta_macro_f1": new_macro - old_macro,
        "category_delta": category_delta,
        "folds_won": int(sum(row["delta"] > 0 for row in fold_rows)),
        "folds": fold_rows,
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "regressed": int(regressed.sum()),
        "component_bootstrap": bootstrap(labels, categories, baseline, candidate, safe, components),
        "structural_invariants": {
            "changed_unsafe": int(((baseline != candidate) & ~safe).sum()),
        },
        "fold_models": fold_models,
        "runtime_seconds": time.monotonic() - started,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (EMBEDDINGS, LOCKED, GUARD)},
    }
    ratio = audit["corrected"] / max(1, audit["regressed"])
    gates = {
        "delta_at_least_0_001": audit["delta_macro_f1"] >= 0.001,
        "wins_at_least_4_of_5": audit["folds_won"] >= 4,
        "corrected_to_regressed_at_least_1_5": ratio >= 1.5,
        "no_category_drop_over_0_005": min(category_delta.values()) >= -0.005,
        "bootstrap_probability_at_least_0_80": audit["component_bootstrap"]["probability_delta_positive"] >= 0.80,
        "structural_invariants": audit["structural_invariants"]["changed_unsafe"] == 0,
    }
    audit["acceptance"] = gates
    audit["accepted_for_repeated_gate"] = all(gates.values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage1_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(OUT / "stage1_predictions.npz", ids=ids, labels=labels, categories=categories, folds=folds, safe=safe, baseline=baseline, candidate=candidate, metric_scores=scores)
    print(json.dumps({
        "accepted_for_repeated_gate": audit["accepted_for_repeated_gate"],
        "delta_macro_f1": audit["delta_macro_f1"], "folds_won": audit["folds_won"],
        "corrected": audit["corrected"], "regressed": audit["regressed"],
        "bootstrap_probability": audit["component_bootstrap"]["probability_delta_positive"],
        "runtime_seconds": audit["runtime_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

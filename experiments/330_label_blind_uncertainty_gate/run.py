from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
sys.path.insert(0, str(RESEARCH))

from audit_component_decision_survival import (  # noqa: E402
    apply_downstream_priors,
    build_neighbor_graph,
    prepare_frame,
)
from qwen35_locked_190_audit import LOCKED_CONFIG, best_threshold, f1  # noqa: E402


BASE = RESEARCH / "qwen35-hard-5fold-robust-fusion-report.npz"
QWEN3VL = RESEARCH / "lora-hard-5fold-robust-fusion-report.npz"
LOCKED = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
FOLDS = ROOT / "validation/grouped_text_v1/folds.csv"
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
REPEATS = ROOT / "validation/connected_family_repeated_v1/rows.csv"
DATA = RESEARCH / "data.csv"
OUT = Path(__file__).resolve().parent / "results"

ADVANTAGES = (0.0, 0.05, 0.10, 0.15)
BASELINE_CAPS = (0.60, 0.70, 0.80, 0.90)
MIN_OPPOSING = (2, 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_platt(values: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> LogisticRegression:
    values = np.ascontiguousarray(values, dtype=np.float64).reshape(-1, 1)
    labels = np.ascontiguousarray(labels, dtype=np.int8)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("non-finite calibration input")
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=33042)
    model.fit(values, labels, sample_weight=weights)
    return model


def calibrate(
    train_matrix: np.ndarray,
    train_combined: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    valid_matrix: np.ndarray,
    valid_combined: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    for column in range(train_matrix.shape[1]):
        model = fit_platt(train_matrix[:, column], train_labels, train_weights)
        values = np.ascontiguousarray(valid_matrix[:, column], dtype=np.float64).reshape(-1, 1)
        probabilities.append(model.predict_proba(values)[:, 1])
    baseline_model = fit_platt(train_combined, train_labels, train_weights)
    values = np.ascontiguousarray(valid_combined, dtype=np.float64).reshape(-1, 1)
    baseline_probability = baseline_model.predict_proba(values)[:, 1]
    return np.column_stack(probabilities), baseline_probability


def apply_gate(
    baseline: np.ndarray,
    component_probability: np.ndarray,
    baseline_probability: np.ndarray,
    config: tuple[int, float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if config is None:
        return baseline.copy(), np.zeros(len(baseline), dtype=bool)
    minimum_opposing, advantage, baseline_cap = config
    component_predictions = component_probability >= 0.5
    opposing = component_predictions != baseline[:, None]
    opposing_count = opposing.sum(axis=1)
    component_confidence = np.abs(component_probability - 0.5) * 2.0
    opposing_confidence = np.divide(
        (component_confidence * opposing).sum(axis=1),
        np.maximum(1, opposing_count),
    )
    baseline_confidence = np.abs(baseline_probability - 0.5) * 2.0
    flip = (
        (opposing_count >= minimum_opposing)
        & (opposing_confidence - baseline_confidence >= advantage)
        & (baseline_confidence <= baseline_cap)
    )
    result = baseline.copy()
    result[flip] = 1 - result[flip]
    return result, flip


def family_weights(components: np.ndarray) -> np.ndarray:
    counts = Counter(components.tolist())
    return np.asarray([1.0 / counts[value] for value in components], dtype=np.float64)


def nested_topology(
    *,
    name: str,
    matrix: np.ndarray,
    combined: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    components: np.ndarray,
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    baseline_all = np.zeros(len(labels), dtype=np.int8)
    candidate_all = np.zeros(len(labels), dtype=np.int8)
    changed_all = np.zeros(len(labels), dtype=bool)
    fold_rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []

    for category in sorted(np.unique(categories)):
        category_mask = eligible & (categories == category)
        for outer_fold in sorted(np.unique(folds[category_mask])):
            donor = category_mask & (folds != outer_fold)
            outer = category_mask & (folds == outer_fold)
            donor_positions = np.flatnonzero(donor)
            inner_base = np.zeros(donor.sum(), dtype=np.int8)
            inner_component_probability = np.zeros((donor.sum(), 3), dtype=np.float64)
            inner_baseline_probability = np.zeros(donor.sum(), dtype=np.float64)

            for inner_fold in sorted(np.unique(folds[donor])):
                inner_train = donor & (folds != inner_fold)
                inner_valid = donor & (folds == inner_fold)
                destinations = np.searchsorted(donor_positions, np.flatnonzero(inner_valid))
                _, threshold = best_threshold(labels[inner_train], combined[inner_train])
                inner_base[destinations] = (combined[inner_valid] >= threshold).astype(np.int8)
                cp, bp = calibrate(
                    matrix[inner_train], combined[inner_train], labels[inner_train],
                    family_weights(components[inner_train]), matrix[inner_valid], combined[inner_valid],
                )
                inner_component_probability[destinations] = cp
                inner_baseline_probability[destinations] = bp

            inner_labels = labels[donor]
            baseline_inner_f1 = f1(inner_labels, inner_base)
            candidates: list[tuple[tuple[float, int, int, float, float], tuple[int, float, float] | None, int]] = []
            candidates.append(((baseline_inner_f1, 1, 0, 1.0, 1.0), None, 0))
            for minimum_opposing in MIN_OPPOSING:
                for advantage in ADVANTAGES:
                    for baseline_cap in BASELINE_CAPS:
                        predictions, changed = apply_gate(
                            inner_base, inner_component_probability, inner_baseline_probability,
                            (minimum_opposing, advantage, baseline_cap),
                        )
                        score = f1(inner_labels, predictions)
                        preference = (
                            score,
                            int(not changed.any()),
                            -int(changed.sum()),
                            advantage,
                            -baseline_cap,
                        )
                        candidates.append((preference, (minimum_opposing, advantage, baseline_cap), int(changed.sum())))
            preference, config, inner_changes = max(candidates, key=lambda item: item[0])

            _, outer_threshold = best_threshold(labels[donor], combined[donor])
            outer_baseline = (combined[outer] >= outer_threshold).astype(np.int8)
            cp, bp = calibrate(
                matrix[donor], combined[donor], labels[donor], family_weights(components[donor]),
                matrix[outer], combined[outer],
            )
            outer_candidate, outer_changed = apply_gate(outer_baseline, cp, bp, config)
            baseline_all[outer] = outer_baseline
            candidate_all[outer] = outer_candidate
            changed_all[outer] = outer_changed
            old = f1(labels[outer], outer_baseline)
            new = f1(labels[outer], outer_candidate)
            fold_rows.append({
                "category": category,
                "fold": int(outer_fold),
                "baseline_f1": old,
                "candidate_f1": new,
                "delta": new - old,
                "changed": int(outer_changed.sum()),
            })
            selected.append({
                "category": category,
                "fold": int(outer_fold),
                "config": None if config is None else {
                    "minimum_opposing": config[0], "advantage": config[1], "baseline_cap": config[2]
                },
                "inner_baseline_f1": baseline_inner_f1,
                "inner_candidate_f1": preference[0],
                "inner_changes": inner_changes,
            })

    by_category_baseline = {}
    by_category_candidate = {}
    for category in sorted(np.unique(categories)):
        mask = eligible & (categories == category)
        by_category_baseline[category] = f1(labels[mask], baseline_all[mask])
        by_category_candidate[category] = f1(labels[mask], candidate_all[mask])
    baseline_macro = float(np.mean(list(by_category_baseline.values())))
    candidate_macro = float(np.mean(list(by_category_candidate.values())))
    fold_macro = []
    for fold in sorted(np.unique(folds[eligible])):
        rows = [row for row in fold_rows if row["fold"] == fold]
        old = float(np.mean([row["baseline_f1"] for row in rows]))
        new = float(np.mean([row["candidate_f1"] for row in rows]))
        fold_macro.append({"fold": int(fold), "baseline_macro_f1": old, "candidate_macro_f1": new, "delta": new - old})
    corrected = eligible & (baseline_all != labels) & (candidate_all == labels)
    regressed = eligible & (baseline_all == labels) & (candidate_all != labels)
    report = {
        "topology": name,
        "rows": int(eligible.sum()),
        "baseline_category_f1": by_category_baseline,
        "candidate_category_f1": by_category_candidate,
        "baseline_macro_f1": baseline_macro,
        "candidate_macro_f1": candidate_macro,
        "delta_macro_f1": candidate_macro - baseline_macro,
        "category_delta": {key: by_category_candidate[key] - by_category_baseline[key] for key in by_category_baseline},
        "folds_won": int(sum(row["delta"] > 0 for row in fold_macro)),
        "folds": fold_macro,
        "changed": int((eligible & changed_all).sum()),
        "corrected": int(corrected.sum()),
        "regressed": int(regressed.sum()),
        "selection": selected,
        "structural_invariants": {
            "changes_outside_eligible": int((changed_all & ~eligible).sum()),
            "changes_without_two_opponents": 0,
        },
    }
    return baseline_all, candidate_all, report


def component_bootstrap(
    report: dict[str, object], labels: np.ndarray, categories: np.ndarray,
    baseline: np.ndarray, candidate: np.ndarray, eligible: np.ndarray,
    components: np.ndarray, seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[np.ndarray]] = {}
    for category in sorted(np.unique(categories)):
        local: dict[str, list[int]] = {}
        for position in np.flatnonzero(eligible & (categories == category)):
            local.setdefault(components[position], []).append(position)
        groups[category] = [np.asarray(value) for value in local.values()]
    deltas = np.empty(5000, dtype=np.float64)
    for iteration in range(len(deltas)):
        old, new = [], []
        for category in sorted(groups):
            local = groups[category]
            sampled = rng.integers(0, len(local), size=len(local))
            positions = np.concatenate([local[index] for index in sampled])
            old.append(f1(labels[positions], baseline[positions]))
            new.append(f1(labels[positions], candidate[positions]))
        deltas[iteration] = np.mean(new) - np.mean(old)
    report["component_bootstrap"] = {
        "iterations": len(deltas),
        "seed": seed,
        "probability_delta_positive": float((deltas > 0).mean()),
        "delta_mean": float(deltas.mean()),
        "delta_ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
    }


def main() -> None:
    started = time.monotonic()
    base = np.load(BASE, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    locked = np.load(LOCKED, allow_pickle=False)
    ids = base["ids"].astype(str)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    historical_folds = base["folds"].astype(np.int8)
    for source in (qwen3vl, locked):
        for key, expected in (("ids", ids), ("labels", labels), ("categories", categories)):
            if not np.array_equal(source[key].astype(expected.dtype), expected):
                raise ValueError(f"unaligned {key}")
    if not np.array_equal(qwen3vl["folds"].astype(np.int8), historical_folds):
        raise ValueError("Qwen3-VL fold mismatch")
    if not np.array_equal(locked["folds"].astype(np.int8), historical_folds):
        raise ValueError("locked fold mismatch")

    fold_frame = pd.read_csv(FOLDS, dtype={"id": str}).set_index("id").loc[ids]
    guard = pd.read_csv(GUARD, dtype={"id": str}).set_index("id").loc[ids]
    repeats = pd.read_csv(REPEATS, dtype={"id": str}).set_index("id").loc[ids]
    if not np.array_equal(fold_frame.fold.to_numpy(np.int8), historical_folds):
        raise ValueError("fold registry mismatch")
    safe = guard.safe_for_selection.to_numpy(bool)
    components = guard.connected_component.astype(str).to_numpy()

    base_rank = base["base_rank"].astype(np.float32)
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)
    qwen35_rank = base["lora_rank"].astype(np.float32)
    matrix = np.column_stack([base_rank, qwen3vl_rank, qwen35_rank])
    combined = np.zeros(len(labels), dtype=np.float32)
    for category in sorted(np.unique(categories)):
        mask = categories == category
        weights = np.asarray(list(LOCKED_CONFIG[category]["weights"]), dtype=np.float32)
        combined[mask] = np.sum(matrix[mask] * weights[None, :], axis=1, dtype=np.float64)

    topologies: dict[str, dict[str, object]] = {}
    predictions: dict[str, np.ndarray] = {}
    for index, (name, folds, eligible) in enumerate([
        ("historical", historical_folds, np.ones(len(labels), dtype=bool)),
        ("historical_connected_safe", historical_folds, safe),
        ("repeat_0", repeats.repeat_0_fold.to_numpy(np.int8), safe),
        ("repeat_1", repeats.repeat_1_fold.to_numpy(np.int8), safe),
        ("repeat_2", repeats.repeat_2_fold.to_numpy(np.int8), safe),
    ]):
        baseline, candidate, report = nested_topology(
            name=name, matrix=matrix, combined=combined, labels=labels, categories=categories,
            folds=folds, components=components, eligible=eligible,
        )
        component_bootstrap(report, labels, categories, baseline, candidate, eligible, components, 33042 + index)
        topologies[name] = report
        predictions[f"{name}_baseline"] = baseline
        predictions[f"{name}_candidate"] = candidate

    prior_frame = prepare_frame(DATA, locked)
    neighbor_indices, neighbor_scores = build_neighbor_graph(prior_frame)
    historical_baseline = predictions["historical_baseline"]
    historical_candidate = predictions["historical_candidate"]
    baseline_after, baseline_prior = apply_downstream_priors(prior_frame, historical_baseline, neighbor_indices, neighbor_scores)
    candidate_after, candidate_prior = apply_downstream_priors(prior_frame, historical_candidate, neighbor_indices, neighbor_scores)
    after_category = {}
    for category in sorted(np.unique(categories)):
        mask = categories == category
        after_category[category] = {
            "baseline_f1": f1(labels[mask], baseline_after[mask]),
            "candidate_f1": f1(labels[mask], candidate_after[mask]),
        }
    after_baseline_macro = float(np.mean([row["baseline_f1"] for row in after_category.values()]))
    after_candidate_macro = float(np.mean([row["candidate_f1"] for row in after_category.values()]))
    priors = {
        "protocol": "donor-only Public-190 prior replay",
        "category": after_category,
        "baseline_macro_f1": after_baseline_macro,
        "candidate_macro_f1": after_candidate_macro,
        "delta_macro_f1": after_candidate_macro - after_baseline_macro,
        "changed_before": int((historical_baseline != historical_candidate).sum()),
        "changed_after": int((baseline_after != candidate_after).sum()),
        "baseline_prior_hits": baseline_prior,
        "candidate_prior_hits": candidate_prior,
    }

    historical = topologies["historical"]
    connected = topologies["historical_connected_safe"]
    repeat_deltas = [topologies[f"repeat_{index}"]["delta_macro_f1"] for index in range(3)]
    category_drops = list(historical["category_delta"].values()) + list(connected["category_delta"].values())
    gates = {
        "historical_delta_at_least_0_002": historical["delta_macro_f1"] >= 0.002,
        "historical_wins_at_least_4_of_5": historical["folds_won"] >= 4,
        "historical_bootstrap_probability_at_least_0_90": historical["component_bootstrap"]["probability_delta_positive"] >= 0.90,
        "connected_delta_at_least_0_002": connected["delta_macro_f1"] >= 0.002,
        "connected_wins_at_least_4_of_5": connected["folds_won"] >= 4,
        "connected_bootstrap_probability_at_least_0_90": connected["component_bootstrap"]["probability_delta_positive"] >= 0.90,
        "no_category_drop_over_0_005": min(category_drops) >= -0.005,
        "all_repeat_deltas_positive": all(value > 0 for value in repeat_deltas),
        "repeat_mean_delta_at_least_0_002": float(np.mean(repeat_deltas)) >= 0.002,
        "positive_macro_delta_after_priors": priors["delta_macro_f1"] > 0,
        "structural_invariants": all(
            report["structural_invariants"]["changes_outside_eligible"] == 0
            for report in topologies.values()
        ),
    }
    accepted = all(gates.values())
    result = {
        "experiment_id": "330",
        "evaluation_version": "component_transfer_gate_v4_selective_extension",
        "recipe_frozen_before_scoring": True,
        "production_inputs_only": ["robust_base_rank", "qwen3vl_rank", "qwen35_rank"],
        "topologies": topologies,
        "downstream_priors": priors,
        "acceptance": gates,
        "accepted_for_full_refit": accepted,
        "runtime_seconds": time.monotonic() - started,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (BASE, QWEN3VL, LOCKED, FOLDS, GUARD, REPEATS, DATA)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selective_gate_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "prior_replay.json").write_text(json.dumps(priors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        OUT / "selective_gate_predictions.npz", ids=ids, labels=labels, categories=categories,
        historical_folds=historical_folds, **predictions,
        historical_baseline_after_priors=baseline_after,
        historical_candidate_after_priors=candidate_after,
    )
    print(json.dumps({
        "accepted_for_full_refit": accepted,
        "historical_delta": historical["delta_macro_f1"],
        "connected_delta": connected["delta_macro_f1"],
        "repeat_deltas": repeat_deltas,
        "after_priors_delta": priors["delta_macro_f1"],
        "runtime_seconds": result["runtime_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

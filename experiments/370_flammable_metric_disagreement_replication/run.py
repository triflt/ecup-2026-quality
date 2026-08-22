from __future__ import annotations

import hashlib
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_360 = ROOT / "experiments/360_hard_negative_diagonal_metric/run.py"
spec = importlib.util.spec_from_file_location("exp360_metric", SOURCE_360)
metric = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(metric)

EMBEDDINGS = ROOT / "research/first-image-artifacts/extracted/train_embeddings_fp16.npz"
QWEN35 = ROOT / "research/qwen35-hard-5fold-robust-fusion-report.npz"
QWEN3VL = ROOT / "research/lora-hard-5fold-robust-fusion-report.npz"
LOCKED = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
REPEATS = ROOT / "validation/connected_family_repeated_v1/rows.csv"
OUT = Path(__file__).resolve().parent / "results"

BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"
DISAGREEMENT_QUANTILE = 0.90
CONFIG = {
    BAD: {"weights": np.asarray([0.50, 0.25, 0.25]), "threshold": 0.27193570137023926},
    FLAMMABLE: {"weights": np.asarray([0.15, 0.10, 0.75]), "threshold": 0.953912615776062},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def production_baseline(matrix: np.ndarray, categories: np.ndarray) -> np.ndarray:
    result = np.zeros(len(categories), dtype=np.int8)
    for category, config in CONFIG.items():
        mask = categories == category
        score = np.sum(matrix[mask] * config["weights"][None, :], axis=1, dtype=np.float64)
        result[mask] = score >= config["threshold"]
    return result


def score_repeat(
    embeddings: np.ndarray, labels: np.ndarray, categories: np.ndarray,
    folds: np.ndarray, safe: np.ndarray, components: np.ndarray,
    baseline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    candidate = baseline.copy()
    scores = np.full(len(labels), np.nan, dtype=np.float32)
    reports = []
    for outer_fold in sorted(np.unique(folds[safe])):
        donor = safe & (categories == FLAMMABLE) & (folds != outer_fold)
        outer = safe & (categories == FLAMMABLE) & (folds == outer_fold)
        prototypes, prototype_labels, _, family_report = metric.family_prototypes(
            embeddings, labels, components, np.flatnonzero(donor)
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
            selected, scale, metric_report = metric.fit_metric(prototypes, prototype_labels)
        if not np.isfinite(scale).all():
            raise ValueError("non-finite metric scale")
        local_score = metric.metric_score(
            embeddings[outer], prototypes, prototype_labels, selected, scale
        )
        scores[outer] = local_score
        local_baseline = baseline[outer]
        metric_prediction = local_score >= 0
        disagreement = metric_prediction != local_baseline
        disagreement_positions = np.flatnonzero(disagreement)
        change = np.zeros(outer.sum(), dtype=bool)
        if len(disagreement_positions):
            confidence = np.abs(local_score[disagreement_positions])
            threshold = float(np.quantile(confidence, DISAGREEMENT_QUANTILE))
            chosen = disagreement_positions[confidence >= threshold]
            change[chosen] = True
        else:
            threshold = None
        positions = np.flatnonzero(outer)
        candidate[positions[change]] = metric_prediction[change]
        reports.append({
            "fold": int(outer_fold), "outer_rows": int(outer.sum()),
            "disagreements": int(disagreement.sum()), "changes": int(change.sum()),
            "disagreement_confidence_threshold": threshold,
            **family_report, **metric_report,
        })
    return candidate, scores, reports


def summarize(
    name: str, labels: np.ndarray, categories: np.ndarray, folds: np.ndarray,
    safe: np.ndarray, components: np.ndarray, baseline: np.ndarray,
    candidate: np.ndarray, reports: list[dict[str, object]], seed: int,
) -> dict[str, object]:
    old, new = {}, {}
    for category in sorted(np.unique(categories)):
        mask = safe & (categories == category)
        old[category] = metric.f1(labels[mask], baseline[mask])
        new[category] = metric.f1(labels[mask], candidate[mask])
    fold_rows = []
    for fold in sorted(np.unique(folds[safe])):
        old_values, new_values = [], []
        for category in sorted(np.unique(categories)):
            mask = safe & (categories == category) & (folds == fold)
            old_values.append(metric.f1(labels[mask], baseline[mask]))
            new_values.append(metric.f1(labels[mask], candidate[mask]))
        fold_rows.append({
            "fold": int(fold), "baseline_macro_f1": float(np.mean(old_values)),
            "candidate_macro_f1": float(np.mean(new_values)),
            "delta": float(np.mean(new_values) - np.mean(old_values)),
        })
    changed = safe & (baseline != candidate)
    corrected = changed & (baseline != labels) & (candidate == labels)
    regressed = changed & (baseline == labels) & (candidate != labels)
    boot = metric.bootstrap(labels, categories, baseline, candidate, safe, components)
    boot["replication_offset_documentation"] = seed
    return {
        "topology": name,
        "baseline_category_f1": old,
        "candidate_category_f1": new,
        "baseline_macro_f1": float(np.mean(list(old.values()))),
        "candidate_macro_f1": float(np.mean(list(new.values()))),
        "delta_macro_f1": float(np.mean(list(new.values())) - np.mean(list(old.values()))),
        "category_delta": {key: new[key] - old[key] for key in old},
        "folds_won": int(sum(row["delta"] > 0 for row in fold_rows)),
        "folds": fold_rows,
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "regressed": int(regressed.sum()),
        "component_bootstrap": boot,
        "structural_invariants": {
            "changed_bad": int((changed & (categories == BAD)).sum()),
            "changed_unsafe": int(((baseline != candidate) & ~safe).sum()),
        },
        "fold_models": reports,
    }


def main() -> None:
    archive = np.load(EMBEDDINGS, allow_pickle=False)
    qwen35 = np.load(QWEN35, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    locked = np.load(LOCKED, allow_pickle=False)
    ids = locked["ids"].astype(str)
    labels = locked["labels"].astype(np.int8)
    categories = locked["categories"].astype(str)
    for source in (archive, qwen35, qwen3vl):
        if not np.array_equal(source["ids"].astype(str), ids):
            raise ValueError("id mismatch")
        if not np.array_equal(source["labels"].astype(np.int8), labels):
            raise ValueError("label mismatch")
        if not np.array_equal(source["categories"].astype(str), categories):
            raise ValueError("category mismatch")
    embeddings = metric.normalize_rows(archive["embeddings"].astype(np.float32))
    guard = pd.read_csv(GUARD, dtype={"id": str}).set_index("id").loc[ids]
    repeats = pd.read_csv(REPEATS, dtype={"id": str}).set_index("id").loc[ids]
    safe = guard.safe_for_selection.to_numpy(bool)
    components = guard.connected_component.astype(str).to_numpy()
    matrix = np.column_stack([
        qwen35["base_rank"].astype(np.float32),
        qwen3vl["lora_rank"].astype(np.float32),
        qwen35["lora_rank"].astype(np.float32),
    ])
    baseline = production_baseline(matrix, categories)
    topologies = {}
    arrays = {}
    for index in range(3):
        name = f"repeat_{index}"
        folds = repeats[f"repeat_{index}_fold"].to_numpy(np.int8)
        candidate, scores, reports = score_repeat(
            embeddings, labels, categories, folds, safe, components, baseline
        )
        topologies[name] = summarize(
            name, labels, categories, folds, safe, components, baseline,
            candidate, reports, 37042 + index,
        )
        arrays[f"{name}_folds"] = folds
        arrays[f"{name}_candidate"] = candidate
        arrays[f"{name}_scores"] = scores
        print(f"{name} delta={topologies[name]['delta_macro_f1']:.9f} corrected={topologies[name]['corrected']} regressed={topologies[name]['regressed']}", flush=True)

    deltas = [topologies[f"repeat_{index}"]["delta_macro_f1"] for index in range(3)]
    total_corrected = sum(topologies[f"repeat_{index}"]["corrected"] for index in range(3))
    total_regressed = sum(topologies[f"repeat_{index}"]["regressed"] for index in range(3))
    total_wins = sum(topologies[f"repeat_{index}"]["folds_won"] for index in range(3))
    gates = {
        "all_repeat_macro_deltas_positive": all(value > 0 for value in deltas),
        "all_repeat_flammable_deltas_positive": all(topologies[f"repeat_{index}"]["category_delta"][FLAMMABLE] > 0 for index in range(3)),
        "repeat_mean_delta_at_least_0_001": float(np.mean(deltas)) >= 0.001,
        "fold_wins_at_least_11_of_15": total_wins >= 11,
        "total_corrected_to_regressed_at_least_2": total_corrected >= 2 * max(1, total_regressed),
        "each_corrected_to_regressed_at_least_1": all(topologies[f"repeat_{index}"]["corrected"] >= topologies[f"repeat_{index}"]["regressed"] for index in range(3)),
        "each_bootstrap_probability_at_least_0_80": all(topologies[f"repeat_{index}"]["component_bootstrap"]["probability_delta_positive"] >= 0.80 for index in range(3)),
        "structural_invariants": all(all(value == 0 for value in topologies[f"repeat_{index}"]["structural_invariants"].values()) for index in range(3)),
    }
    result = {
        "experiment_id": "370",
        "evaluation_version": "flammable_metric_connected_replication_v1",
        "historical_motivation_not_acceptance": {"corrected": 7, "regressed": 2},
        "recipe_frozen_before_replication": True,
        "topologies": topologies,
        "summary": {
            "repeat_deltas": deltas,
            "mean_delta": float(np.mean(deltas)),
            "total_fold_wins": total_wins,
            "total_corrected": total_corrected,
            "total_regressed": total_regressed,
        },
        "acceptance": gates,
        "accepted_for_full_refit": all(gates.values()),
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (SOURCE_360, EMBEDDINGS, QWEN35, QWEN3VL, LOCKED, GUARD, REPEATS)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replication_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(OUT / "replication_predictions.npz", ids=ids, labels=labels, categories=categories, safe=safe, baseline=baseline, **arrays)
    print(json.dumps({"accepted_for_full_refit": result["accepted_for_full_refit"], **result["summary"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

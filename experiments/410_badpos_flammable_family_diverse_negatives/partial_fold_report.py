from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from qwen35_locked_190_audit import LOCKED_CONFIG  # noqa: E402
from qwen35_seed_ensemble_cv import BASE, QWEN3VL, QWEN35_SEED_A, f1, rank01  # noqa: E402


EXPERIMENT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/260_bad_family_diverse_positives/artifacts/seed_42_diverse_positives"
CANDIDATE = EXPERIMENT / "artifacts/seed_42_negdiv"
LOCKED = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
FLAMMABLE = "Легковоспламеняющиеся"


def aligned_scores(path: Path, ids: np.ndarray, fold: int) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"id": str})
    if frame.id.duplicated().any():
        raise ValueError(f"duplicate ids in {path}")
    if set(frame.id) != set(ids):
        raise ValueError(f"id set mismatch in {path}")
    if not (frame.fold.to_numpy(np.int8) == fold).all():
        raise ValueError(f"fold mismatch in {path}")
    return frame.set_index("id").loc[ids].lora_score.to_numpy(np.float32)


def category_f1(labels: np.ndarray, predictions: np.ndarray, categories: np.ndarray) -> dict:
    return {
        category: f1(labels[categories == category], predictions[categories == category])
        for category in sorted(np.unique(categories))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--parent-root", type=Path, default=PARENT)
    parser.add_argument("--candidate-root", type=Path, default=CANDIDATE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fold = args.fold

    base = np.load(BASE, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    original = np.load(QWEN35_SEED_A, allow_pickle=True)
    locked = np.load(LOCKED, allow_pickle=False)
    ids_all = base["ids"].astype(str)
    labels_all = base["labels"].astype(np.int8)
    categories_all = base["categories"].astype(str)
    folds_all = base["fold_ids"].astype(np.int8)
    positions = np.flatnonzero(folds_all == fold)
    ids = ids_all[positions]
    labels = labels_all[positions]
    categories = categories_all[positions]

    parent_raw = aligned_scores(
        args.parent_root / f"fold_{fold}/lora_holdout_predictions.csv", ids, fold
    )
    candidate_raw = aligned_scores(
        args.candidate_root / f"fold_{fold}/lora_holdout_predictions.csv", ids, fold
    )
    base_rank = original["base_rank"].astype(np.float32)[positions]
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)[positions]
    parent_rank = np.empty(len(positions), dtype=np.float32)
    candidate_rank = np.empty(len(positions), dtype=np.float32)
    for category in sorted(np.unique(categories)):
        local = np.flatnonzero(categories == category)
        parent_rank[local] = rank01(parent_raw[local])
        candidate_rank[local] = rank01(candidate_raw[local])
    for name, values in (
        ("base_rank", base_rank),
        ("qwen3vl_rank", qwen3vl_rank),
        ("parent_rank", parent_rank),
        ("candidate_rank", candidate_rank),
    ):
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError(
                f"invalid {name}: finite={np.isfinite(values).all()} "
                f"range=({values.min()}, {values.max()})"
            )

    config = LOCKED_CONFIG[FLAMMABLE]
    weights = np.asarray(config["weights"], dtype=np.float32)
    threshold = float(config["production_threshold"])
    flammable = categories == FLAMMABLE
    parent_scores = np.sum(
        np.column_stack([base_rank, qwen3vl_rank, parent_rank]) * weights[None, :],
        axis=1,
        dtype=np.float32,
    )
    candidate_scores = np.sum(
        np.column_stack([base_rank, qwen3vl_rank, candidate_rank]) * weights[None, :],
        axis=1,
        dtype=np.float32,
    )
    parent_flammable = (parent_scores >= threshold).astype(np.int8)
    candidate_flammable = (candidate_scores >= threshold).astype(np.int8)

    baseline = np.where(
        flammable,
        parent_flammable,
        locked["baseline_production_predictions"].astype(np.int8)[positions],
    ).astype(np.int8)
    candidate = np.where(flammable, candidate_flammable, baseline).astype(np.int8)
    expected_parent = locked["exp260_production_predictions"].astype(np.int8)[positions]
    if not np.array_equal(parent_flammable[flammable], expected_parent[flammable]):
        raise ValueError("parent 260 production replay mismatch")

    old_scores = category_f1(labels, baseline, categories)
    new_scores = category_f1(labels, candidate, categories)
    old_macro = float(np.mean(list(old_scores.values())))
    new_macro = float(np.mean(list(new_scores.values())))
    changed = baseline != candidate
    positive_flammable = flammable & (labels == 1)
    report = {
        "experiment_id": "410",
        "fold": fold,
        "diagnostic_only": True,
        "reason_not_acceptance_metric": "The fixed production threshold does not replace the frozen nested threshold fitted on the other four folds.",
        "threshold": threshold,
        "baseline_macro_f1": old_macro,
        "candidate_macro_f1": new_macro,
        "delta_macro_f1": new_macro - old_macro,
        "baseline_category_f1": old_scores,
        "candidate_category_f1": new_scores,
        "changed": int(changed.sum()),
        "corrected": int((changed & (candidate == labels) & (baseline != labels)).sum()),
        "regressed": int((changed & (candidate != labels) & (baseline == labels)).sum()),
        "flammable_false_negatives": {
            "baseline": int((baseline[positive_flammable] == 0).sum()),
            "candidate": int((candidate[positive_flammable] == 0).sum()),
        },
    }
    report["flammable_false_negatives"]["delta"] = (
        report["flammable_false_negatives"]["candidate"]
        - report["flammable_false_negatives"]["baseline"]
    )
    path = args.output or EXPERIMENT / f"analysis/partial_fold_{fold}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

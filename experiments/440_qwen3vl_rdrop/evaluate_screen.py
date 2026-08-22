from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from qwen35_locked_190_audit import evaluate_locked
from qwen35_seed_ensemble_cv import (
    BASE,
    QWEN3VL,
    QWEN35_SEED_A,
    f1,
    fold_category_ranks,
    load_seed_predictions,
    rank01,
)

BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"
SCREEN_FOLDS = (0, 3)
PARENT_QWEN35 = (
    ROOT
    / "experiments/260_bad_family_diverse_positives/artifacts/seed_42_diverse_positives"
)
ROUTE400 = (
    ROOT
    / "experiments/400_qwen35_category_routed_adapters/results/routed_predictions.npz"
)
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
DATA = ROOT / "research/data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
SAFETY_PATTERN = re.compile(
    r"(?iu)\b(?:зажигалк|спич|огнив|факел|свеч|горелк|фейерверк|салют|"
    r"петард|бенгал|пиротех|дымогенератор|свеч\w*\s*фонтан|"
    r"фонтан\w*\s+для\s+торт|угол|уголь|дров|брик|розжиг)\w*\b"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_scores(
    labels: np.ndarray,
    predictions: np.ndarray,
    categories: np.ndarray,
) -> dict[str, float]:
    return {
        category: f1(labels[categories == category], predictions[categories == category])
        for category in sorted(np.unique(categories))
    }


def fold_audit(
    *,
    fold: int,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    safe: np.ndarray,
    safety_union: np.ndarray,
) -> dict[str, object]:
    mask = folds == fold
    old = category_scores(labels[mask], baseline[mask], categories[mask])
    new = category_scores(labels[mask], candidate[mask], categories[mask])
    old_macro = float(np.mean(list(old.values())))
    new_macro = float(np.mean(list(new.values())))
    category_delta = {category: new[category] - old[category] for category in old}
    changed = mask & (baseline != candidate)
    corrected = int((changed & (candidate == labels) & (baseline != labels)).sum())
    regressed = int((changed & (candidate != labels) & (baseline == labels)).sum())
    flammable_positive = mask & (categories == FLAMMABLE) & (labels == 1)
    old_fn = int((baseline[flammable_positive] == 0).sum())
    new_fn = int((candidate[flammable_positive] == 0).sum())
    safety = mask & safety_union
    old_safety_fn = int((baseline[safety] == 0).sum())
    new_safety_fn = int((candidate[safety] == 0).sum())
    safe_mask = mask & safe
    safe_old = category_scores(
        labels[safe_mask], baseline[safe_mask], categories[safe_mask]
    )
    safe_new = category_scores(
        labels[safe_mask], candidate[safe_mask], categories[safe_mask]
    )
    safe_delta = float(np.mean(list(safe_new.values())) - np.mean(list(safe_old.values())))
    gates = {
        "routed_macro_delta_positive": new_macro - old_macro > 0,
        "no_category_drop_over_0_005": min(category_delta.values()) >= -0.005,
        "flammable_false_negatives_not_increased": new_fn <= old_fn,
        "corrected_exceeds_regressed": corrected > regressed,
        "safety_union_false_negatives_not_increased": new_safety_fn <= old_safety_fn,
    }
    return {
        "fold": fold,
        "baseline_category_f1": old,
        "candidate_category_f1": new,
        "category_delta": category_delta,
        "baseline_macro_f1": old_macro,
        "candidate_macro_f1": new_macro,
        "delta_macro_f1": new_macro - old_macro,
        "changed": int(changed.sum()),
        "corrected": corrected,
        "regressed": regressed,
        "flammable_false_negatives": {
            "baseline": old_fn,
            "candidate": new_fn,
            "delta": new_fn - old_fn,
        },
        "safety_union_false_negatives": {
            "positive_rows": int(safety.sum()),
            "baseline": old_safety_fn,
            "candidate": new_safety_fn,
            "delta": new_safety_fn - old_safety_fn,
        },
        "connected_safe_delta_macro_f1": safe_delta,
        "gates": gates,
        "passed": all(gates.values()),
    }


def load_candidate_ranks(
    *,
    candidate_paths: dict[int, Path],
    ids: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    parent_rank: np.ndarray,
) -> tuple[np.ndarray, dict[str, str]]:
    hybrid_rank = parent_rank.copy()
    checksums: dict[str, str] = {}
    for fold, path in candidate_paths.items():
        frame = pd.read_csv(path, dtype={"id": str})
        required = {"id", "category", "label", "fold", "lora_score"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"fold {fold} prediction file lacks columns: {missing}")
        expected = np.flatnonzero(folds == fold)
        if not np.array_equal(frame["id"].astype(str).to_numpy(), ids[expected]):
            raise ValueError(f"fold {fold} prediction id/order mismatch")
        if not np.array_equal(frame["label"].to_numpy(np.int8), labels[expected]):
            raise ValueError(f"fold {fold} prediction label mismatch")
        if not np.array_equal(frame["category"].astype(str).to_numpy(), categories[expected]):
            raise ValueError(f"fold {fold} prediction category mismatch")
        if not np.all(frame["fold"].to_numpy(np.int8) == fold):
            raise ValueError(f"fold {fold} prediction file contains another fold")
        scores = frame["lora_score"].to_numpy(np.float32)
        if not np.isfinite(scores).all():
            raise ValueError(f"fold {fold} candidate scores are not finite")
        for category in sorted(np.unique(categories)):
            local_in_frame = frame["category"].astype(str).to_numpy() == category
            local_global = expected[local_in_frame]
            hybrid_rank[local_global] = rank01(scores[local_in_frame])
        checksums[str(path)] = sha256(path)
    return hybrid_rank, checksums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-0", type=Path)
    parser.add_argument("--fold-3", type=Path)
    parser.add_argument("--null-control", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.null_control:
        if args.fold_0 is not None or args.fold_3 is not None:
            raise ValueError("null control does not accept candidate prediction files")
    elif args.fold_0 is None and args.fold_3 is None:
        raise ValueError("candidate screen requires at least one fold prediction file")

    base = np.load(BASE, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    original_qwen35 = np.load(QWEN35_SEED_A, allow_pickle=True)
    ids = base["ids"].astype(str)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    folds = base["fold_ids"].astype(np.int8)
    for name, source in (("qwen3vl", qwen3vl), ("qwen35", original_qwen35)):
        if not np.array_equal(ids, source["ids"].astype(str)):
            raise ValueError(f"{name} id mismatch")
        if not np.array_equal(folds, source["folds"].astype(np.int8)):
            raise ValueError(f"{name} fold mismatch")

    parent_paths = [
        PARENT_QWEN35 / f"fold_{fold}/lora_holdout_predictions.csv"
        for fold in range(5)
    ]
    parent_qwen35_logits = load_seed_predictions(parent_paths, base)
    parent_qwen35_rank = fold_category_ranks(parent_qwen35_logits, folds, categories)
    original_qwen35_rank = original_qwen35["lora_rank"].astype(np.float32)
    routed_qwen35_rank = np.where(
        categories == FLAMMABLE, parent_qwen35_rank, original_qwen35_rank
    ).astype(np.float32)
    parent_qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)
    if args.null_control:
        candidate_qwen3vl_rank = parent_qwen3vl_rank.copy()
        candidate_checksums: dict[str, str] = {}
        evaluated_folds = list(SCREEN_FOLDS)
    else:
        candidate_paths = {
            fold: path
            for fold, path in ((0, args.fold_0), (3, args.fold_3))
            if path is not None
        }
        candidate_qwen3vl_rank, candidate_checksums = load_candidate_ranks(
            candidate_paths=candidate_paths,
            ids=ids,
            labels=labels,
            categories=categories,
            folds=folds,
            parent_rank=parent_qwen3vl_rank,
        )
        evaluated_folds = sorted(candidate_paths)
    base_rank = original_qwen35["base_rank"].astype(np.float32)
    baseline, _, baseline_detail = evaluate_locked(
        np.column_stack([base_rank, parent_qwen3vl_rank, routed_qwen35_rank]),
        labels,
        categories,
        folds,
    )
    candidate, _, candidate_detail = evaluate_locked(
        np.column_stack([base_rank, candidate_qwen3vl_rank, routed_qwen35_rank]),
        labels,
        categories,
        folds,
    )
    frozen = np.load(ROUTE400, allow_pickle=False)
    if not np.array_equal(frozen["ids"].astype(str), ids):
        raise ValueError("experiment 400 routed id mismatch")
    if not np.array_equal(
        frozen["category_routed_nested_predictions"].astype(np.int8), baseline
    ):
        raise ValueError("reconstructed experiment-400 baseline differs from frozen artifact")

    guard = pd.read_csv(GUARD, dtype={"id": str})
    if not np.array_equal(guard["id"].astype(str).to_numpy(), ids):
        raise ValueError("connected guard id mismatch")
    safe = guard["safe_for_selection"].astype(bool).to_numpy()
    data = pd.read_csv(DATA, dtype={"id": str})
    if not np.array_equal(data["id"].astype(str).to_numpy(), ids):
        raise ValueError("data id mismatch")
    text = data["name"].fillna("").astype(str) + "\n" + data["description"].fillna("").astype(str)
    safety_union = (
        (categories == FLAMMABLE)
        & (labels == 1)
        & text.str.contains(SAFETY_PATTERN, na=False).to_numpy()
    )
    fold_reports = [
        fold_audit(
            fold=fold,
            labels=labels,
            categories=categories,
            folds=folds,
            baseline=baseline,
            candidate=candidate,
            safe=safe,
            safety_union=safety_union,
        )
        for fold in evaluated_folds
    ]
    mean_delta = float(np.mean([item["delta_macro_f1"] for item in fold_reports]))
    null_control_passed = bool(np.array_equal(candidate, baseline))
    all_screen_folds_ready = set(evaluated_folds) == set(SCREEN_FOLDS)
    screen_passed = (
        not args.null_control
        and all_screen_folds_ready
        and all(item["passed"] for item in fold_reports)
        and mean_delta >= 0.001
    )
    result = {
        "experiment_id": "440",
        "evaluation_version": "qwen3vl_rdrop_two_fold_screen_v1",
        "purpose": "Reject-only screen; passing authorizes but cannot replace full five-fold evaluation",
        "mode": "null_control" if args.null_control else "candidate",
        "required_screen_folds": list(SCREEN_FOLDS),
        "evaluated_folds": evaluated_folds,
        "all_screen_folds_ready": all_screen_folds_ready,
        "changed_factor": "Qwen3-VL adapter objective only: R-Drop alpha 1.0 symmetric binary KL",
        "threshold_protocol": "Hybrid Qwen3-VL OOF ranks on folds 0 and 3; parent Qwen3-VL ranks on folds 1, 2 and 4; each tested fold is excluded from threshold fitting",
        "weights": {
            BAD: {"robust_base": 0.50, "qwen3vl": 0.25, "qwen35": 0.25},
            FLAMMABLE: {"robust_base": 0.15, "qwen3vl": 0.10, "qwen35": 0.75},
        },
        "selection_uses_screen_labels": False,
        "folds": fold_reports,
        "mean_screen_delta_macro_f1": mean_delta,
        "required_mean_delta_macro_f1": 0.001,
        "screen_passed": screen_passed,
        "full_five_fold_cycle_allowed": screen_passed,
        "null_control_passed": null_control_passed if args.null_control else None,
        "baseline_locked_detail": baseline_detail,
        "candidate_locked_detail": candidate_detail,
        "input_sha256": {
            **candidate_checksums,
            str(ROUTE400.relative_to(ROOT)): sha256(ROUTE400),
            str(GUARD.relative_to(ROOT)): sha256(GUARD),
            str(DATA.relative_to(ROOT)): sha256(DATA),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_name = "null_screen_control.json" if args.null_control else "screen_audit.json"
    npz_name = "null_screen_control.npz" if args.null_control else "screen_predictions.npz"
    (args.output_dir / json_name).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / npz_name,
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        baseline_nested_predictions=baseline,
        candidate_screen_nested_predictions=candidate,
        parent_qwen3vl_rank=parent_qwen3vl_rank,
        candidate_qwen3vl_rank=candidate_qwen3vl_rank,
    )
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "mean_delta": mean_delta,
                "changed": int((baseline != candidate).sum()),
                "screen_passed": screen_passed,
                "null_control_passed": result["null_control_passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

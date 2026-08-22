from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "experiments/440_qwen3vl_rdrop/evaluate_screen.py"
SCREEN_FOLDS = (0, 3)
FLAMMABLE_WEIGHT = 0.75
PAIRWISE_WEIGHT = 0.10
PAIRWISE_MARGIN = 1.0
MIN_CONSTRAINTS = 64


def _load_base():
    spec = importlib.util.spec_from_file_location("_exp560_route400_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load locked route400 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()


def validate_report(predictions: Path, fold: int) -> dict:
    path = predictions.with_name("lora_holdout_report.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    pairwise = report.get("pairwise", {})
    if report.get("experiment_id") != "560" or report.get("holdout_fold") != fold:
        raise ValueError(f"fold {fold} candidate report contract mismatch")
    if pairwise.get("pairwise_weight") != PAIRWISE_WEIGHT:
        raise ValueError("candidate pairwise weight mismatch")
    if pairwise.get("pairwise_margin") != PAIRWISE_MARGIN:
        raise ValueError("candidate pairwise margin mismatch")
    if int(pairwise.get("pairwise_constraints", 0)) < MIN_CONSTRAINTS:
        raise ValueError("candidate did not execute enough pairwise constraints")
    return {"report": str(path), "sha256": base.sha256(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exp560 through locked route400.")
    parser.add_argument("--fold-0", type=Path)
    parser.add_argument("--fold-3", type=Path)
    parser.add_argument("--null-control", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    supplied = {fold: path for fold, path in ((0, args.fold_0), (3, args.fold_3)) if path}
    if args.null_control and supplied:
        raise ValueError("null control accepts no candidate files")
    if not args.null_control and not supplied:
        raise ValueError("candidate mode requires fold predictions")
    output_json = args.output_dir / (
        "null_screen_control.json" if args.null_control else "screen_audit.json"
    )
    output_npz = args.output_dir / (
        "null_screen_control.npz" if args.null_control else "screen_predictions.npz"
    )
    if output_json.exists() or output_npz.exists():
        raise FileExistsError("refusing to overwrite evaluator outputs")

    source = np.load(base.BASE, allow_pickle=True)
    qwen3vl = np.load(base.QWEN3VL, allow_pickle=True)
    original_qwen35 = np.load(base.QWEN35_SEED_A, allow_pickle=True)
    ids = source["ids"].astype(str)
    labels = source["labels"].astype(np.int8)
    categories = source["categories"].astype(str)
    folds = source["fold_ids"].astype(np.int8)
    parent_paths = [
        base.PARENT_QWEN35 / f"fold_{fold}/lora_holdout_predictions.csv" for fold in range(5)
    ]
    parent_logits = base.load_seed_predictions(parent_paths, source)
    parent_flammable_rank = base.fold_category_ranks(parent_logits, folds, categories)
    original_rank = original_qwen35["lora_rank"].astype(np.float32)
    routed_parent_rank = np.where(
        categories == base.FLAMMABLE, parent_flammable_rank, original_rank
    ).astype(np.float32)
    contracts = {}
    if args.null_control:
        candidate_flammable_rank = parent_flammable_rank.copy()
        evaluated_folds = list(SCREEN_FOLDS)
    else:
        for fold, path in supplied.items():
            contracts[str(fold)] = validate_report(path, fold)
        candidate_all_rank, _ = base.load_candidate_ranks(
            candidate_paths=supplied,
            ids=ids,
            labels=labels,
            categories=categories,
            folds=folds,
            parent_rank=parent_flammable_rank,
        )
        candidate_flammable_rank = np.where(
            categories == base.FLAMMABLE, candidate_all_rank, parent_flammable_rank
        )
        evaluated_folds = sorted(supplied)
    routed_candidate_rank = np.where(
        categories == base.FLAMMABLE, candidate_flammable_rank, original_rank
    ).astype(np.float32)
    base_rank = original_qwen35["base_rank"].astype(np.float32)
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)
    baseline, _, baseline_detail = base.evaluate_locked(
        np.column_stack([base_rank, qwen3vl_rank, routed_parent_rank]),
        labels,
        categories,
        folds,
    )
    candidate, _, candidate_detail = base.evaluate_locked(
        np.column_stack([base_rank, qwen3vl_rank, routed_candidate_rank]),
        labels,
        categories,
        folds,
    )
    frozen = np.load(base.ROUTE400, allow_pickle=False)
    frozen_predictions = frozen["category_routed_nested_predictions"].astype(np.int8)
    if not np.array_equal(frozen_predictions, baseline):
        raise ValueError("route400 reconstruction mismatch")
    guard = pd.read_csv(base.GUARD, dtype={"id": str, "connected_component": str})
    if not np.array_equal(guard.id.astype(str).to_numpy(), ids):
        raise ValueError("guard id mismatch")
    safe = guard.safe_for_selection.astype(bool).to_numpy()
    data = pd.read_csv(base.DATA, dtype={"id": str})
    text = data.name.fillna("").astype(str) + "\n" + data.description.fillna("").astype(str)
    safety = (
        (categories == base.FLAMMABLE)
        & (labels == 1)
        & text.str.contains(base.SAFETY_PATTERN, na=False).to_numpy()
    )
    fold_reports = []
    for fold in evaluated_folds:
        item = base.fold_audit(
            fold=fold,
            labels=labels,
            categories=categories,
            folds=folds,
            baseline=baseline,
            candidate=candidate,
            safe=safe,
            safety_union=safety,
        )
        local = folds == fold
        item["gates"].update(
            {
                "flammable_delta_positive": item["category_delta"][base.FLAMMABLE] > 0,
                "bad_predictions_identical": bool(
                    np.array_equal(
                        baseline[local & (categories == base.BAD)],
                        candidate[local & (categories == base.BAD)],
                    )
                ),
                "connected_safe_delta_positive": item["connected_safe_delta_macro_f1"] > 0,
            }
        )
        item["passed"] = all(item["gates"].values())
        fold_reports.append(item)
    mean_delta = float(np.mean([item["delta_macro_f1"] for item in fold_reports]))
    null_pass = bool(np.array_equal(candidate, baseline))
    complete = set(evaluated_folds) == set(SCREEN_FOLDS)
    passed = (
        not args.null_control
        and complete
        and all(item["passed"] for item in fold_reports)
        and mean_delta >= 0.001
    )
    report = {
        "experiment_id": "560",
        "evaluation_version": "qwen35_flammable_hard_pairwise_route400_screen_v1",
        "mode": "null_control" if args.null_control else "candidate",
        "required_screen_folds": list(SCREEN_FOLDS),
        "evaluated_folds": evaluated_folds,
        "same_replacement_weight_frozen_before_training": True,
        "weights": {
            base.BAD: {"robust_base": 0.50, "qwen3vl": 0.25, "qwen35": 0.25},
            base.FLAMMABLE: {
                "robust_base": 0.15,
                "qwen3vl": 0.10,
                "qwen35": FLAMMABLE_WEIGHT,
            },
        },
        "candidate_replaces": "flammable Qwen3.5 rank only",
        "candidate_contracts": contracts,
        "folds": fold_reports,
        "mean_screen_delta_macro_f1": mean_delta,
        "required_mean_delta_macro_f1": 0.001,
        "screen_passed": passed,
        "full_five_fold_cycle_allowed": False,
        "null_control_passed": null_pass if args.null_control else None,
        "baseline_locked_detail": baseline_detail,
        "candidate_locked_detail": candidate_detail,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_npz,
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        baseline_nested_predictions=baseline,
        candidate_screen_nested_predictions=candidate,
        parent_flammable_qwen35_rank=parent_flammable_rank,
        candidate_flammable_qwen35_rank=candidate_flammable_rank,
    )
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "null_control_passed": report["null_control_passed"],
                "screen_passed": passed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

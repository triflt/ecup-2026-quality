from __future__ import annotations

"""Evaluate the locked two-fold H2 screen through the proven route-400 path."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from resolution_contract import (
    CANDIDATE_MAX_EDGE,
    CANDIDATE_MAX_PIXELS,
    SCREEN_FOLDS,
)

ROOT = Path(__file__).resolve().parents[2]
BASE_EVALUATOR_PATH = ROOT / "experiments/440_qwen3vl_rdrop/evaluate_screen.py"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
EVALUATION_VERSION = "qwen3vl_first_image_672_two_fold_screen_v1"
CHANGED_FACTOR = (
    "Qwen3-VL first-image resolution only: edge cap 448 to 672 and "
    "processor max_pixels to 451584 at training and inference"
)


def _load_base_evaluator():
    spec = importlib.util.spec_from_file_location(
        "_qwen3vl_locked_route400_evaluator", BASE_EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load locked evaluator: {BASE_EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_evaluator()


def candidate_report_path(predictions_path: Path, explicit_path: Path | None) -> Path:
    return explicit_path or predictions_path.with_name("lora_holdout_report.json")


def validate_candidate_report(path: Path, *, fold: int) -> dict[str, int]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"fold {fold} candidate report is missing") from error
    if not isinstance(raw, dict):
        raise TypeError(f"fold {fold} candidate report must be a JSON object")
    expected = {
        "holdout_fold": fold,
        "first_image_max_edge": CANDIDATE_MAX_EDGE,
        "first_image_max_pixels": CANDIDATE_MAX_PIXELS,
    }
    for key, value in expected.items():
        observed = raw.get(key)
        if isinstance(observed, bool) or observed != value:
            raise ValueError(
                f"fold {fold} candidate report has {key}={observed!r}; expected {value}"
            )
    if raw.get("full_train") is True:
        raise ValueError(f"fold {fold} candidate report unexpectedly describes full training")
    return expected


def h2_fold_audit(**kwargs) -> dict[str, object]:
    report = base.fold_audit(**kwargs)
    category_delta = report["category_delta"]
    corrected = int(report["corrected"])
    regressed = int(report["regressed"])
    gates = {
        "routed_macro_delta_positive": report["delta_macro_f1"] > 0,
        "bad_f1_drop_not_over_0_003": category_delta[base.BAD] >= -0.003,
        "flammable_false_negatives_not_increased": (
            report["flammable_false_negatives"]["delta"] <= 0
        ),
        "corrected_exceeds_regressed": corrected > regressed,
        "safety_union_false_negatives_not_increased": (
            report["safety_union_false_negatives"]["delta"] <= 0
        ),
    }
    report["gates"] = gates
    report["passed"] = all(gates.values())
    return report


def output_paths(output_dir: Path, *, null_control: bool) -> tuple[Path, Path]:
    if null_control:
        return output_dir / "null_screen_control.json", output_dir / "null_screen_control.npz"
    return output_dir / "screen_audit.json", output_dir / "screen_predictions.npz"


def ensure_output_targets_absent(paths: tuple[Path, Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing screen outputs: "
            + ", ".join(map(str, existing))
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-0", type=Path)
    parser.add_argument("--fold-0-report", type=Path)
    parser.add_argument("--fold-3", type=Path)
    parser.add_argument("--fold-3-report", type=Path)
    parser.add_argument("--null-control", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    candidate_inputs = {
        0: (args.fold_0, args.fold_0_report),
        3: (args.fold_3, args.fold_3_report),
    }
    if args.null_control:
        if any(path is not None for paths in candidate_inputs.values() for path in paths):
            raise ValueError("null control does not accept candidate files or reports")
    else:
        dangling_reports = [
            fold
            for fold, (predictions, report) in candidate_inputs.items()
            if predictions is None and report is not None
        ]
        if dangling_reports:
            raise ValueError(f"reports supplied without predictions: {dangling_reports}")
        if all(predictions is None for predictions, _ in candidate_inputs.values()):
            raise ValueError("candidate screen requires at least one fold prediction file")
    targets = output_paths(args.output_dir, null_control=args.null_control)
    ensure_output_targets_absent(targets)

    source_base = np.load(base.BASE, allow_pickle=True)
    qwen3vl = np.load(base.QWEN3VL, allow_pickle=True)
    original_qwen35 = np.load(base.QWEN35_SEED_A, allow_pickle=True)
    ids = source_base["ids"].astype(str)
    labels = source_base["labels"].astype(np.int8)
    categories = source_base["categories"].astype(str)
    folds = source_base["fold_ids"].astype(np.int8)
    for name, source in (("qwen3vl", qwen3vl), ("qwen35", original_qwen35)):
        if not np.array_equal(ids, source["ids"].astype(str)):
            raise ValueError(f"{name} id mismatch")
        if not np.array_equal(folds, source["folds"].astype(np.int8)):
            raise ValueError(f"{name} fold mismatch")

    parent_paths = [
        base.PARENT_QWEN35 / f"fold_{fold}/lora_holdout_predictions.csv"
        for fold in range(5)
    ]
    parent_qwen35_logits = base.load_seed_predictions(parent_paths, source_base)
    parent_qwen35_rank = base.fold_category_ranks(
        parent_qwen35_logits, folds, categories
    )
    original_qwen35_rank = original_qwen35["lora_rank"].astype(np.float32)
    routed_qwen35_rank = np.where(
        categories == base.FLAMMABLE, parent_qwen35_rank, original_qwen35_rank
    ).astype(np.float32)
    parent_qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)

    report_contracts: dict[str, dict[str, int]] = {}
    input_checksums: dict[str, str] = {}
    if args.null_control:
        candidate_qwen3vl_rank = parent_qwen3vl_rank.copy()
        evaluated_folds = list(SCREEN_FOLDS)
    else:
        candidate_paths: dict[int, Path] = {}
        for fold, (predictions_path, explicit_report_path) in candidate_inputs.items():
            if predictions_path is None:
                continue
            report_path = candidate_report_path(predictions_path, explicit_report_path)
            report_contracts[str(fold)] = validate_candidate_report(report_path, fold=fold)
            candidate_paths[fold] = predictions_path
            input_checksums[f"fold_{fold}_predictions"] = base.sha256(predictions_path)
            input_checksums[f"fold_{fold}_report"] = base.sha256(report_path)
        candidate_qwen3vl_rank, _ = base.load_candidate_ranks(
            candidate_paths=candidate_paths,
            ids=ids,
            labels=labels,
            categories=categories,
            folds=folds,
            parent_rank=parent_qwen3vl_rank,
        )
        evaluated_folds = sorted(candidate_paths)

    base_rank = original_qwen35["base_rank"].astype(np.float32)
    baseline, _, baseline_detail = base.evaluate_locked(
        np.column_stack([base_rank, parent_qwen3vl_rank, routed_qwen35_rank]),
        labels,
        categories,
        folds,
    )
    candidate, _, candidate_detail = base.evaluate_locked(
        np.column_stack([base_rank, candidate_qwen3vl_rank, routed_qwen35_rank]),
        labels,
        categories,
        folds,
    )
    frozen = np.load(base.ROUTE400, allow_pickle=False)
    if not np.array_equal(frozen["ids"].astype(str), ids):
        raise ValueError("experiment-400 routed id mismatch")
    if not np.array_equal(
        frozen["category_routed_nested_predictions"].astype(np.int8), baseline
    ):
        raise ValueError("reconstructed experiment-400 baseline differs from frozen artifact")

    guard = pd.read_csv(base.GUARD, dtype={"id": str})
    if not np.array_equal(guard["id"].astype(str).to_numpy(), ids):
        raise ValueError("connected guard id mismatch")
    safe = guard["safe_for_selection"].astype(bool).to_numpy()
    data = pd.read_csv(base.DATA, dtype={"id": str})
    if not np.array_equal(data["id"].astype(str).to_numpy(), ids):
        raise ValueError("data id mismatch")
    text = (
        data["name"].fillna("").astype(str)
        + "\n"
        + data["description"].fillna("").astype(str)
    )
    safety_union = (
        (categories == base.FLAMMABLE)
        & (labels == 1)
        & text.str.contains(base.SAFETY_PATTERN, na=False).to_numpy()
    )
    fold_reports = [
        h2_fold_audit(
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
    total_screen_changed = int(sum(int(item["changed"]) for item in fold_reports))
    null_control_passed = bool(np.array_equal(candidate, baseline))
    all_screen_folds_ready = set(evaluated_folds) == set(SCREEN_FOLDS)
    screen_passed = (
        not args.null_control
        and all_screen_folds_ready
        and all(item["passed"] for item in fold_reports)
        and mean_delta >= 0.001
        and total_screen_changed >= 5
    )
    result = {
        "experiment_id": "510",
        "evaluation_version": EVALUATION_VERSION,
        "purpose": (
            "Reject-only screen; passing authorizes but cannot replace full five-fold "
            "evaluation"
        ),
        "mode": "null_control" if args.null_control else "candidate",
        "required_screen_folds": list(SCREEN_FOLDS),
        "evaluated_folds": evaluated_folds,
        "all_screen_folds_ready": all_screen_folds_ready,
        "changed_factor": CHANGED_FACTOR,
        "candidate_report_contracts": report_contracts,
        "threshold_protocol": (
            "Hybrid Qwen3-VL OOF ranks on folds 0 and 3; parent Qwen3-VL ranks on "
            "folds 1, 2 and 4; each tested fold is excluded from threshold fitting"
        ),
        "weights": {
            base.BAD: {"robust_base": 0.50, "qwen3vl": 0.25, "qwen35": 0.25},
            base.FLAMMABLE: {
                "robust_base": 0.15,
                "qwen3vl": 0.10,
                "qwen35": 0.75,
            },
        },
        "selection_uses_screen_labels": False,
        "folds": fold_reports,
        "mean_screen_delta_macro_f1": mean_delta,
        "required_mean_delta_macro_f1": 0.001,
        "total_screen_changed_predictions": total_screen_changed,
        "required_changed_predictions": 5,
        "screen_passed": screen_passed,
        "full_five_fold_cycle_allowed": screen_passed,
        "null_control_passed": null_control_passed if args.null_control else None,
        "baseline_locked_detail": baseline_detail,
        "candidate_locked_detail": candidate_detail,
        "input_sha256": {
            **input_checksums,
            str(base.ROUTE400.relative_to(ROOT)): base.sha256(base.ROUTE400),
            str(base.GUARD.relative_to(ROOT)): base.sha256(base.GUARD),
            str(base.DATA.relative_to(ROOT)): base.sha256(base.DATA),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        targets[1],
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

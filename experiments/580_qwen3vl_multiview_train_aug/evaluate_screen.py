from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from multiview_contract import (
    EXPERIMENT_ID,
    FIRST_IMAGE_MAX_EDGE,
    FIRST_IMAGE_MAX_PIXELS,
    INFERENCE_VIEW_POLICY,
    SCREEN_FOLDS,
    SEED,
    TRAINING_VIEW_POLICY,
)

ROOT = Path(__file__).resolve().parents[2]
BASE_EVALUATOR_PATH = ROOT / "experiments/440_qwen3vl_rdrop/evaluate_screen.py"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
EVALUATION_VERSION = "qwen3vl_multiview_train_aug_two_fold_screen_v1"
CHANGED_FACTOR = (
    "Training image view only: one deterministic gallery image per occurrence; "
    "holdout inference remains first image at 448 pixels with one pass"
)


def _load_base_evaluator():
    spec = importlib.util.spec_from_file_location(
        "_exp580_locked_route400_evaluator", BASE_EVALUATOR_PATH
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


def validate_candidate_report(path: Path, *, fold: int) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"fold {fold} candidate report is missing") from error
    if not isinstance(raw, dict):
        raise TypeError(f"fold {fold} candidate report must be a JSON object")
    expected: dict[str, object] = {
        "holdout_fold": fold,
        "training_view_policy": TRAINING_VIEW_POLICY,
        "gallery_seed": SEED,
        "gallery_epoch": 0,
        "training_images_per_occurrence": 1,
        "inference_view_policy": INFERENCE_VIEW_POLICY,
        "inference_image_index": 0,
        "inference_image_count": 1,
        "inference_passes": 1,
        "first_image_max_edge": FIRST_IMAGE_MAX_EDGE,
        "first_image_max_pixels": FIRST_IMAGE_MAX_PIXELS,
        "download_failures": 0,
        "soft_targets": False,
        "last_logit_only": False,
    }
    for key, value in expected.items():
        observed = raw.get(key)
        if observed != value or isinstance(observed, bool) != isinstance(value, bool):
            raise ValueError(
                f"fold {fold} candidate report has {key}={observed!r}; expected {value!r}"
            )
    if raw.get("full_train") is True:
        raise ValueError(f"fold {fold} candidate report unexpectedly describes full training")
    plan = raw.get("gallery_plan")
    if not isinstance(plan, dict):
        raise TypeError(f"fold {fold} candidate report lacks gallery_plan")
    if plan.get("training_occurrences") != raw.get("train_records"):
        raise ValueError(f"fold {fold} gallery plan changes the parent step count")
    return expected


def apply_acceptance(result: dict[str, object]) -> dict[str, object]:
    folds = result["folds"]
    if not isinstance(folds, list):
        raise TypeError("fold report list is missing")
    for fold in folds:
        gates = {
            "fold_delta_positive": fold["delta_macro_f1"] > 0,
            "flammable_false_negatives_not_increased": (
                fold["flammable_false_negatives"]["delta"] <= 0
            ),
            "safety_union_false_negatives_not_increased": (
                fold["safety_union_false_negatives"]["delta"] <= 0
            ),
            "corrected_exceeds_regressed": fold["corrected"] > fold["regressed"],
        }
        fold["gates"] = gates
        fold["passed"] = all(gates.values())
    evaluated = result["evaluated_folds"]
    all_ready = set(evaluated) == set(SCREEN_FOLDS)
    mean_delta = float(result["mean_screen_delta_macro_f1"])
    mode = result["mode"]
    passed = (
        mode == "candidate"
        and all_ready
        and all(fold["passed"] for fold in folds)
        and mean_delta >= 0.001
    )
    result.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "evaluation_version": EVALUATION_VERSION,
            "changed_factor": CHANGED_FACTOR,
            "required_screen_folds": list(SCREEN_FOLDS),
            "all_screen_folds_ready": all_ready,
            "required_mean_delta_macro_f1": 0.001,
            "screen_passed": passed,
            "full_five_fold_cycle_allowed": passed,
        }
    )
    return result


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
        if any(path is not None for pair in candidate_inputs.values() for path in pair):
            raise ValueError("null control does not accept candidate files or reports")
    else:
        if all(predictions is None for predictions, _ in candidate_inputs.values()):
            raise ValueError("candidate screen requires at least one fold prediction file")
        dangling = [
            fold
            for fold, (predictions, report) in candidate_inputs.items()
            if predictions is None and report is not None
        ]
        if dangling:
            raise ValueError(f"reports supplied without predictions: {dangling}")

    report_contracts: dict[str, dict[str, object]] = {}
    report_checksums: dict[str, str] = {}
    base_argv = [str(BASE_EVALUATOR_PATH)]
    if args.null_control:
        base_argv.append("--null-control")
    else:
        for fold, (predictions, explicit_report) in candidate_inputs.items():
            if predictions is None:
                continue
            report = candidate_report_path(predictions, explicit_report)
            report_contracts[str(fold)] = validate_candidate_report(report, fold=fold)
            report_checksums[f"fold_{fold}_candidate_report"] = base.sha256(report)
            base_argv.extend([f"--fold-{fold}", str(predictions)])
    base_argv.extend(["--output-dir", str(args.output_dir)])
    previous_argv = sys.argv
    try:
        sys.argv = base_argv
        base.main()
    finally:
        sys.argv = previous_argv

    name = "null_screen_control.json" if args.null_control else "screen_audit.json"
    path = args.output_dir / name
    result = json.loads(path.read_text(encoding="utf-8"))
    result = apply_acceptance(result)
    result["candidate_report_contracts"] = report_contracts
    result["input_sha256"].update(report_checksums)
    result["selection_uses_screen_labels"] = False
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "mean_delta": result["mean_screen_delta_macro_f1"],
                "screen_passed": result["screen_passed"],
                "null_control_passed": result["null_control_passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

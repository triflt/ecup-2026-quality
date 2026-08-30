from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from contract import INFERENCE_VIEW_POLICY, SCREEN_FOLDS, TRAINING_VIEW_POLICY

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "experiments/440_qwen3vl_rdrop/evaluate_screen.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("_exp590_locked_route400", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load locked route400 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()


def validate_candidate_report(predictions: Path, fold: int) -> dict[str, str]:
    path = predictions.with_name("lora_holdout_report.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": "590",
        "holdout_fold": fold,
        "training_view_policy": TRAINING_VIEW_POLICY,
        "inference_view_policy": INFERENCE_VIEW_POLICY,
        "inference_image_index": 0,
        "inference_image_count": 1,
        "inference_passes": 1,
        "first_image_max_edge": 448,
        "first_image_max_pixels": 262144,
        "preflight_decision": "GO",
        "record_multiset_unchanged": True,
        "steps_unchanged": True,
    }
    mismatch = {
        key: (value, report.get(key)) for key, value in expected.items() if report.get(key) != value
    }
    if mismatch:
        raise ValueError(f"fold {fold} candidate report mismatch: {mismatch}")
    if report.get("recombined_fraction_of_repeats") != 0.25:
        raise ValueError("candidate recombination rate mismatch")
    return {"path": str(path), "sha256": base.sha256(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exp590 through exact route400.")
    parser.add_argument("--fold-0", type=Path)
    parser.add_argument("--fold-3", type=Path)
    parser.add_argument("--null-control", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = {0: args.fold_0, 3: args.fold_3}
    if args.null_control and any(candidates.values()):
        raise ValueError("null control accepts no candidate files")
    if not args.null_control and not any(candidates.values()):
        raise ValueError("candidate mode requires predictions")
    contracts = {}
    base_argv = [str(BASE_PATH)]
    if args.null_control:
        base_argv.append("--null-control")
    else:
        for fold, path in candidates.items():
            if path is not None:
                contracts[str(fold)] = validate_candidate_report(path, fold)
                base_argv.extend([f"--fold-{fold}", str(path)])
    base_argv.extend(["--output-dir", str(args.output_dir)])
    old_argv = sys.argv
    try:
        sys.argv = base_argv
        base.main()
    finally:
        sys.argv = old_argv
    filename = "null_screen_control.json" if args.null_control else "screen_audit.json"
    path = args.output_dir / filename
    result = json.loads(path.read_text(encoding="utf-8"))
    for fold in result["folds"]:
        gates = {
            "fold_delta_positive": fold["delta_macro_f1"] > 0,
            "flammable_false_negatives_not_increased": fold["flammable_false_negatives"]["delta"]
            <= 0,
            "safety_union_false_negatives_not_increased": fold["safety_union_false_negatives"][
                "delta"
            ]
            <= 0,
            "corrected_exceeds_regressed": fold["corrected"] > fold["regressed"],
            "bad_drop_at_most_0_003": fold["category_delta"][base.BAD] >= -0.003,
        }
        fold["gates"] = gates
        fold["passed"] = all(gates.values())
    complete = set(result["evaluated_folds"]) == set(SCREEN_FOLDS)
    passed = (
        result["mode"] == "candidate"
        and complete
        and all(fold["passed"] for fold in result["folds"])
        and result["mean_screen_delta_macro_f1"] >= 0.001
    )
    result.update(
        {
            "experiment_id": "590",
            "evaluation_version": "qwen3vl_cross_listing_route400_screen_v1",
            "required_screen_folds": list(SCREEN_FOLDS),
            "all_screen_folds_ready": complete,
            "required_mean_delta_macro_f1": 0.001,
            "candidate_replaces": "Qwen3-VL rank only at the existing route400 weights",
            "candidate_report_contracts": contracts,
            "screen_passed": passed,
            "full_five_fold_cycle_allowed": False,
        }
    )
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "null_control_passed": result["null_control_passed"],
                "screen_passed": passed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

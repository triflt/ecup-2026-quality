from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT_EXPERIMENT = ROOT / "experiments/520_qwen35_grounded_auxiliary_sft"
ENGINE_PATH = ROOT / "experiments/560_qwen35_flammable_hard_pairwise/evaluate_screen.py"
SCREEN_FOLDS = (0, 3)
EXPERIMENT_ID = "570"
FORMAT_VERSION = "grounded_auxiliary_target_v1"
LOSS_CONTRACT_VERSION = "weighted_grounded_auxiliary_loss_v1"
LOSS_CONTRACT_SHA256 = "0419aa40b5c9aa4866f963548d2cd77a096260e6a972948d4535eb286dbed728"
VERDICT_TOKEN_WEIGHT = 1.0
EVIDENCE_TOKEN_WEIGHT = 0.05


def _load_engine():
    spec = importlib.util.spec_from_file_location("_exp570_route400_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load locked route400 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = _load_engine()


def validate_report(predictions: Path, fold: int) -> dict:
    report_path = predictions.with_name("lora_holdout_report.json")
    grounded_runtime_path = predictions.with_name("grounded_target_audit.runtime.json")
    loss_runtime_path = predictions.with_name("weighted_loss_audit.runtime.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runtime = json.loads(grounded_runtime_path.read_text(encoding="utf-8"))
    loss_runtime = json.loads(loss_runtime_path.read_text(encoding="utf-8"))
    frozen = json.loads(
        (
            PARENT_EXPERIMENT / f"analysis/coverage_fold_{fold}.json"
        ).read_text(encoding="utf-8")
    )
    expected_runtime = {
        "experiment_id": EXPERIMENT_ID,
        "holdout_fold": fold,
        "seed": 42,
        "training_records": 5390,
        "format_version": FORMAT_VERSION,
        "decision": "GO",
        "outer_validation_training_occurrences": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": runtime.get(key)}
        for key, value in expected_runtime.items()
        if runtime.get(key) != value
    }
    if mismatches:
        raise ValueError(f"fold {fold} grounded runtime contract mismatch: {mismatches}")
    if runtime.get("format_failures") or runtime.get("gates", {}).get("failures"):
        raise ValueError("grounded runtime audit contains failures")
    for key in ("target_plan_sha256", "record_multiset_sha256"):
        if runtime.get(key) != frozen.get(key):
            raise ValueError(f"fold {fold} {key} differs from frozen experiment 520")

    expected_loss = {
        "experiment_id": EXPERIMENT_ID,
        "contract_version": LOSS_CONTRACT_VERSION,
        "contract_sha256": LOSS_CONTRACT_SHA256,
        "verdict_token_weight": VERDICT_TOKEN_WEIGHT,
        "evidence_token_weight": EVIDENCE_TOKEN_WEIGHT,
        "batches": 1348,
        "training_occurrences": 5390,
        "verdict_tokens": 5390,
        "decision": "GO",
    }
    loss_mismatches = {
        key: {"expected": value, "actual": loss_runtime.get(key)}
        for key, value in expected_loss.items()
        if loss_runtime.get(key) != value
    }
    if loss_mismatches:
        raise ValueError(f"fold {fold} weighted loss contract mismatch: {loss_mismatches}")
    if loss_runtime.get("failures"):
        raise ValueError("weighted loss runtime audit contains failures")
    if int(loss_runtime.get("evidence_tokens", 0)) <= 0:
        raise ValueError("weighted loss runtime audit observed no evidence tokens")
    null_control = loss_runtime.get("exact_null_control", {})
    if not null_control.get("executed") or not null_control.get("bit_exact"):
        raise ValueError("weighted loss exact null control did not pass")

    if report.get("holdout_fold") != fold or report.get("train_records") != 5390:
        raise ValueError(f"fold {fold} parent training report mismatch")
    if report.get("download_failures") != 0:
        raise ValueError("candidate report contains image download failures")
    selection = report.get("flammable_selection", {})
    grounded = selection.get("grounded_auxiliary_sft", {})
    weighted = selection.get("weighted_grounded_auxiliary", {})
    if grounded.get("format_version") != FORMAT_VERSION:
        raise ValueError("candidate report grounded target version mismatch")
    if grounded.get("target_plan_sha256") != runtime.get("target_plan_sha256"):
        raise ValueError("candidate report target plan mismatch")
    if weighted.get("contract_version") != LOSS_CONTRACT_VERSION:
        raise ValueError("candidate report weighted loss version mismatch")
    if weighted.get("verdict_token_weight") != VERDICT_TOKEN_WEIGHT:
        raise ValueError("candidate report verdict token weight mismatch")
    if weighted.get("evidence_token_weight") != EVIDENCE_TOKEN_WEIGHT:
        raise ValueError("candidate report evidence token weight mismatch")
    return {
        "report": str(report_path),
        "report_sha256": engine.base.sha256(report_path),
        "grounded_runtime_audit": str(grounded_runtime_path),
        "grounded_runtime_audit_sha256": engine.base.sha256(grounded_runtime_path),
        "loss_runtime_audit": str(loss_runtime_path),
        "loss_runtime_audit_sha256": engine.base.sha256(loss_runtime_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate experiment 570 via frozen route400.")
    parser.add_argument("--fold-0", type=Path)
    parser.add_argument("--fold-3", type=Path)
    parser.add_argument("--null-control", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def engine_output_names(null_control: bool) -> tuple[str, str]:
    if null_control:
        return "null_screen_control", "null_screen_control"
    return "screen_audit", "screen_predictions"


def main() -> int:
    args = parse_args()
    supplied = {fold: path for fold, path in ((0, args.fold_0), (3, args.fold_3)) if path}
    if args.null_control and supplied:
        raise ValueError("null control accepts no candidate files")
    if not args.null_control and set(supplied) != set(SCREEN_FOLDS):
        raise ValueError("candidate screen requires exactly folds 0 and 3")
    output_name = "null_screen_control" if args.null_control else "screen_audit"
    output_json = args.output_dir / f"{output_name}.json"
    output_npz = args.output_dir / f"{output_name}.npz"
    if output_json.exists() or output_npz.exists():
        raise FileExistsError("refusing to overwrite evaluator outputs")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    engine.validate_report = validate_report
    previous_argv = sys.argv
    try:
        with tempfile.TemporaryDirectory(
            prefix=".exp570-route400-", dir=args.output_dir.parent
        ) as temporary:
            argv = ["evaluate_screen.py"]
            if args.null_control:
                argv.append("--null-control")
            else:
                argv.extend(
                    [
                        "--fold-0",
                        str(args.fold_0),
                        "--fold-3",
                        str(args.fold_3),
                    ]
                )
            argv.extend(["--output-dir", temporary])
            sys.argv = argv
            result = engine.main()
            source_name, source_npz_name = engine_output_names(args.null_control)
            report_path = Path(temporary) / f"{source_name}.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.update(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "evaluation_version": "qwen35_weighted_grounded_route400_screen_v1",
                    "training_parent": "520_qwen35_grounded_auxiliary_sft",
                    "candidate_replaces": "flammable Qwen3.5 rank only",
                    "acceptance": {
                        "each_fold_delta_strictly_positive": True,
                        "mean_macro_delta_minimum": 0.001,
                        "flammable_false_negative_increase_maximum": 0,
                        "safety_union_false_negative_increase_maximum": 0,
                    },
                }
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            shutil.copyfile(Path(temporary) / f"{source_npz_name}.npz", output_npz)
    finally:
        sys.argv = previous_argv
    return result


if __name__ == "__main__":
    raise SystemExit(main())

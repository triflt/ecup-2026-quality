from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = ROOT / "experiments/560_qwen35_flammable_hard_pairwise/evaluate_screen.py"
SCREEN_FOLDS = (0, 3)
FORMAT_VERSION = "grounded_auxiliary_target_v1"


def _load_engine():
    spec = importlib.util.spec_from_file_location("_exp520_route400_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load locked route400 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = _load_engine()


def validate_report(predictions: Path, fold: int) -> dict:
    report_path = predictions.with_name("lora_holdout_report.json")
    runtime_path = predictions.with_name("grounded_target_audit.runtime.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    frozen = json.loads(
        (Path(__file__).parent / f"analysis/coverage_fold_{fold}.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "experiment_id": "520",
        "holdout_fold": fold,
        "seed": 42,
        "training_records": 5390,
        "format_version": FORMAT_VERSION,
        "decision": "GO",
        "outer_validation_training_occurrences": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": runtime.get(key)}
        for key, value in expected.items()
        if runtime.get(key) != value
    }
    if mismatches:
        raise ValueError(f"fold {fold} grounded runtime contract mismatch: {mismatches}")
    if runtime.get("format_failures") or runtime.get("gates", {}).get("failures"):
        raise ValueError("grounded runtime audit contains failures")
    for key in ("target_plan_sha256", "record_multiset_sha256"):
        if runtime.get(key) != frozen.get(key):
            raise ValueError(f"fold {fold} {key} differs from the frozen coverage audit")
    if report.get("holdout_fold") != fold or report.get("train_records") != 5390:
        raise ValueError(f"fold {fold} parent training report mismatch")
    if report.get("download_failures") != 0:
        raise ValueError("candidate report contains image download failures")
    grounded = report.get("flammable_selection", {}).get("grounded_auxiliary_sft", {})
    if grounded.get("format_version") != FORMAT_VERSION:
        raise ValueError("candidate report grounded target version mismatch")
    if grounded.get("target_plan_sha256") != runtime.get("target_plan_sha256"):
        raise ValueError("candidate report target plan mismatch")
    return {
        "report": str(report_path),
        "report_sha256": engine.base.sha256(report_path),
        "runtime_audit": str(runtime_path),
        "runtime_audit_sha256": engine.base.sha256(runtime_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exp520 through locked route400.")
    parser.add_argument("--fold-0", required=True, type=Path)
    parser.add_argument("--fold-3", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_json = args.output_dir / "screen_audit.json"
    output_npz = args.output_dir / "screen_predictions.npz"
    if output_json.exists() or output_npz.exists():
        raise FileExistsError("refusing to overwrite evaluator outputs")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    engine.validate_report = validate_report
    previous_argv = sys.argv
    try:
        with tempfile.TemporaryDirectory(
            prefix=".exp520-route400-", dir=args.output_dir.parent
        ) as temporary:
            sys.argv = [
                "evaluate_screen.py",
                "--fold-0",
                str(args.fold_0),
                "--fold-3",
                str(args.fold_3),
                "--output-dir",
                temporary,
            ]
            result = engine.main()
            report_path = Path(temporary) / "screen_audit.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.update(
                {
                    "experiment_id": "520",
                    "evaluation_version": "qwen35_grounded_auxiliary_route400_screen_v1",
                    "candidate_replaces": "flammable Qwen3.5 rank only",
                }
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            shutil.copyfile(Path(temporary) / "screen_predictions.npz", output_npz)
    finally:
        sys.argv = previous_argv
    return result


if __name__ == "__main__":
    raise SystemExit(main())

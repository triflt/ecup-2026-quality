from __future__ import annotations

"""Evaluate the two-fold position-augmentation screen against route 400."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT_EVALUATOR = (
    ROOT
    / "experiments"
    / "420_qwen35_flammable_recall_continuation"
    / "evaluate_screen.py"
)


def load_parent_evaluator():
    spec = importlib.util.spec_from_file_location("exp500_parent_screen_evaluator", PARENT_EVALUATOR)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load evaluator: {PARENT_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-0", type=Path, required=True)
    parser.add_argument("--fold-3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    targets = (
        args.output_dir / "screen_audit.json",
        args.output_dir / "screen_predictions.npz",
    )
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite exp500 screen outputs")

    parent = load_parent_evaluator()
    parent.OUT = args.output_dir
    original_argv = sys.argv
    try:
        sys.argv = [
            str(PARENT_EVALUATOR),
            "--fold-0",
            str(args.fold_0),
            "--fold-3",
            str(args.fold_3),
        ]
        parent.main()
    finally:
        sys.argv = original_argv

    report_path = args.output_dir / "screen_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "experiment_id": "500",
            "evaluation_version": "transaction_scope_position_two_fold_screen_v1",
            "changed_factor": (
                "whole-sentence position augmentation on deterministic outer-train "
                "flammable rows; inference is unchanged"
            ),
        }
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

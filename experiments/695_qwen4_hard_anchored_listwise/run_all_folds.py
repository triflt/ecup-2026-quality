from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ("hard_bce_control", "rank_candidate")


def commit_fold_output(staging: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(f"refusing to replace committed fold output: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)


def main() -> None:
    parser = argparse.ArgumentParser(description="One job: paired train+eval for all five folds.")
    for name in (
        "runtime-root",
        "teacher-root",
        "baseline-root",
        "images",
        "model-root",
        "vendor",
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--teacher-acceptance", type=Path, required=True)
    parser.add_argument("--teacher-acceptance-sha256", required=True)
    parser.add_argument("--runtime-bundle-sha256", required=True)
    parser.add_argument("--baseline-bundle-sha256", required=True)
    parser.add_argument(
        "--runtime-backend", choices=("verified_fast_path", "legacy_eager"), default="legacy_eager"
    )
    parser.add_argument("--technical-smoke", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty output")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        for mode in MODES:
            arm = "control" if mode == MODES[0] else "candidate"
            final_output = args.output_root / arm / f"fold{fold}"
            staging_output = (
                args.output_root.parent
                / ".qwen4_staging"
                / args.output_root.name
                / arm
                / f"fold{fold}"
            )
            command = [
                sys.executable,
                str(HERE / "train_fold.py"),
                "--fold",
                str(fold),
                "--runtime-dir",
                str(args.runtime_root / f"fold{fold}"),
                "--teacher-root",
                str(args.teacher_root),
                "--teacher-acceptance",
                str(args.teacher_acceptance),
                "--teacher-acceptance-sha256",
                args.teacher_acceptance_sha256,
                "--images",
                str(args.images),
                "--model-root",
                str(args.model_root),
                "--model-revision",
                args.model_revision,
                "--vendor",
                str(args.vendor),
                "--output-dir",
                str(staging_output),
                "--runtime-backend",
                args.runtime_backend,
                "--micro-batch-size-override",
                "2",
                "--mode",
                mode,
            ]
            if args.technical_smoke:
                command.append("--technical-smoke")
            subprocess.run(command, check=True)
            commit_fold_output(staging_output, final_output)
    evaluation_command = [
        sys.executable,
        str(HERE / "evaluate.py"),
        "--runtime-root",
        str(args.runtime_root),
        "--teacher-root",
        str(args.teacher_root),
        "--teacher-acceptance",
        str(args.teacher_acceptance),
        "--teacher-acceptance-sha256",
        args.teacher_acceptance_sha256,
        "--runtime-bundle-sha256",
        args.runtime_bundle_sha256,
        "--baseline-bundle-sha256",
        args.baseline_bundle_sha256,
        "--baseline-root",
        str(args.baseline_root),
        "--control-root",
        str(args.output_root / "control"),
        "--candidate-root",
        str(args.output_root / "candidate"),
        "--output",
        str(args.output_root / "evaluation.json"),
    ]
    subprocess.run(evaluation_command, check=True)


if __name__ == "__main__":
    main()

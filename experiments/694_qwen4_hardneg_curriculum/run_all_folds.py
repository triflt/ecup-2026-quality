from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ("hard_bce_control", "hardneg_candidate")


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
    parser.add_argument("--submission-limit-minutes", type=float)
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
                str(args.output_root / arm / f"fold{fold}"),
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
        "--baseline-root",
        str(args.baseline_root),
        "--control-root",
        str(args.output_root / "control"),
        "--candidate-root",
        str(args.output_root / "candidate"),
        "--output",
        str(args.output_root / "evaluation.json"),
    ]
    if args.submission_limit_minutes is not None:
        evaluation_command.extend(
            ["--submission-limit-minutes", str(args.submission_limit_minutes)]
        )
    subprocess.run(evaluation_command, check=True)


if __name__ == "__main__":
    main()

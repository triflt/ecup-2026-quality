from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ("causal_candidate", "hard_bce_control")


def checked_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="One job: paired train+eval for all five folds.")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--teacher-acceptance", type=Path, required=True)
    parser.add_argument("--teacher-acceptance-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--runtime-backend", choices=("verified_fast_path", "legacy_eager"), default="legacy_eager"
    )
    parser.add_argument("--technical-smoke", action="store_true")
    args = parser.parse_args()
    checked_empty(args.output_root)
    for fold in range(5):
        for mode in MODES:
            arm = "control" if mode == "hard_bce_control" else "candidate"
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
    subprocess.run(
        [
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
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

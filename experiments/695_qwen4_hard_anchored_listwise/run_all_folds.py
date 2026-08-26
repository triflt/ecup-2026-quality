from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ("hard_bce_control", "rank_candidate")


def main() -> None:
    parser = argparse.ArgumentParser(description="One job: paired train+eval for all five folds.")
    for name in ("runtime-root", "baseline-root", "images", "model-root", "vendor", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
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
                str(args.runtime_root / f"fold_{fold}"),
                "--images",
                str(args.images),
                "--model-root",
                str(args.model_root),
                "--model-revision",
                args.model_revision,
                "--vendor",
                str(args.vendor),
                "--output-dir",
                str(args.output_root / arm / f"fold_{fold}"),
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

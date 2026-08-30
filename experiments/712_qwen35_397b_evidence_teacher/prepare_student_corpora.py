from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    run(
        sys.executable,
        "-u",
        str(args.code_root / "712" / "build_handoff_artifact.py"),
        "--data", str(args.data),
        "--folds", str(args.folds),
        "--fold-manifest", str(args.fold_manifest),
        "--oof", str(args.oof),
        "--teacher-root", str(args.teacher_root),
        "--image-root", str(args.image_root),
        "--output-root", str(args.output_root),
        "--code-root", str(args.code_root),
    )
    run(
        sys.executable,
        "-u",
        str(args.code_root / "712" / "verify_handoff_artifact.py"),
        "--data", str(args.data),
        "--handoff-root", str(args.output_root),
        "--image-root", str(args.image_root),
        "--model-root", str(args.model_root),
        "--code-root", str(args.code_root),
        "--output-root", str(args.output_root),
        "--max-length", "2304",
        "--batch-size", "8",
    )


if __name__ == "__main__":
    main()

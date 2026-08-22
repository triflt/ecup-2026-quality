from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def resolved(value: Path) -> str:
    return str(value.expanduser().resolve())


def common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, choices=(42, 31415), default=42)
    parser.add_argument("--full-train", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local interface for experiment 500.")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Build the immutable augmentation audit.")
    common_parser(audit)
    train = commands.add_parser("train", help="Run the frozen parent with augmentation.")
    common_parser(train)
    train.add_argument("--audit-json", required=True, type=Path)
    train.add_argument("--image-manifest", required=True, type=Path)
    train.add_argument("--images", required=True, type=Path)
    train.add_argument("--model-root", required=True, type=Path)
    train.add_argument("--vendor", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "audit":
        command = [
            sys.executable,
            "-u",
            str(HERE / "build_manifest.py"),
            "--data",
            resolved(args.data),
            "--oof",
            resolved(args.oof),
            "--output-dir",
            resolved(args.output_dir),
            "--fold",
            str(args.fold),
            "--seed",
            str(args.seed),
        ]
        if args.full_train:
            command.append("--full-train")
        return subprocess.run(command, cwd=HERE.parents[1], check=False).returncode

    environment = os.environ.copy()
    environment.update(
        {
            "ECUP_DATA": resolved(args.data),
            "ECUP_OOF": resolved(args.oof),
            "ECUP_OUTPUT_DIR": resolved(args.output_dir),
            "ECUP_MANIFEST": resolved(args.image_manifest),
            "ECUP_IMAGES": resolved(args.images),
            "ECUP_MODEL_ROOT": resolved(args.model_root),
            "ECUP_VENDOR": resolved(args.vendor),
            "ECUP_AUGMENT_AUDIT_JSON": resolved(args.audit_json),
            "HOLDOUT_FOLD": str(args.fold),
            "SEED": str(args.seed),
            "FULL_TRAIN": "1" if args.full_train else "0",
        }
    )
    return subprocess.run(
        [sys.executable, "-u", str(HERE / "trainer.py")],
        cwd=HERE.parents[1],
        env=environment,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

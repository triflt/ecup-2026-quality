from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def resolved(path: Path) -> str:
    return str(path.expanduser().resolve())


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--guard", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int, choices=(0, 3))
    parser.add_argument("--seed", default=42, type=int, choices=(42,))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local exp540 interface.")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    add_common(audit)
    audit.add_argument("--output-dir", required=True, type=Path)
    train = commands.add_parser("train")
    add_common(train)
    train.add_argument("--audit-json", required=True, type=Path)
    train.add_argument("--image-manifest", required=True, type=Path)
    train.add_argument("--images", required=True, type=Path)
    train.add_argument("--model-root", required=True, type=Path)
    train.add_argument("--vendor", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "audit":
        return subprocess.run(
            [
                sys.executable,
                "-u",
                str(HERE / "build_manifest.py"),
                "--data",
                resolved(args.data),
                "--oof",
                resolved(args.oof),
                "--guard",
                resolved(args.guard),
                "--output-dir",
                resolved(args.output_dir),
                "--fold",
                str(args.fold),
                "--seed",
                str(args.seed),
            ],
            cwd=HERE.parents[1],
            check=False,
        ).returncode
    audit_path = args.audit_json.resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("decision") != "GO" or audit.get("holdout_fold") != args.fold:
        raise ValueError("training requires a GO pair audit for the same fold")
    environment = os.environ.copy()
    environment.update(
        {
            "ECUP_DATA": resolved(args.data),
            "ECUP_OOF": resolved(args.oof),
            "ECUP_CONNECTED_GUARD": resolved(args.guard),
            "ECUP_PAIR_AUDIT_JSON": resolved(args.audit_json),
            "ECUP_MANIFEST": resolved(args.image_manifest),
            "ECUP_IMAGES": resolved(args.images),
            "ECUP_MODEL_ROOT": resolved(args.model_root),
            "ECUP_VENDOR": resolved(args.vendor),
            "ECUP_OUTPUT_DIR": resolved(args.output_dir),
            "HOLDOUT_FOLD": str(args.fold),
            "SEED": str(args.seed),
            "FULL_TRAIN": "0",
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

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _resolved(path: Path) -> str:
    return str(path.expanduser().resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local fail-closed exp590 interface.")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--data", required=True, type=Path)
    audit.add_argument("--oof", required=True, type=Path)
    audit.add_argument("--sealed-dir", required=True, type=Path)
    audit.add_argument("--draft-dir", required=True, type=Path)
    audit.add_argument("--reviews", type=Path)
    audit.add_argument("--output-dir", required=True, type=Path)
    train = commands.add_parser("train")
    train.add_argument("--fold", required=True, type=int, choices=(0, 3))
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--oof", required=True, type=Path)
    train.add_argument("--preflight-audit", required=True, type=Path)
    train.add_argument("--recombination-manifest", required=True, type=Path)
    train.add_argument("--image-manifest", required=True, type=Path)
    train.add_argument("--images", required=True, type=Path)
    train.add_argument("--model-root", required=True, type=Path)
    train.add_argument("--vendor", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "audit":
        command = [
            sys.executable,
            "-u",
            str(HERE / "build_preflight.py"),
            "--data",
            _resolved(args.data),
            "--oof",
            _resolved(args.oof),
            "--sealed-dir",
            _resolved(args.sealed_dir),
            "--draft-dir",
            _resolved(args.draft_dir),
            "--output-dir",
            _resolved(args.output_dir),
        ]
        if args.reviews:
            command.extend(["--reviews", _resolved(args.reviews)])
        return subprocess.run(command, cwd=HERE.parents[1], check=False).returncode
    audit = json.loads(args.preflight_audit.resolve().read_text(encoding="utf-8"))
    if audit.get("decision") != "GO" or audit.get("holdout_fold") != args.fold:
        raise ValueError("training is blocked until the same-fold preflight is GO")
    environment = os.environ.copy()
    environment.update(
        {
            "ECUP_DATA": _resolved(args.data),
            "ECUP_OOF": _resolved(args.oof),
            "ECUP_MANIFEST": _resolved(args.image_manifest),
            "ECUP_IMAGES": _resolved(args.images),
            "ECUP_MODEL_ROOT": _resolved(args.model_root),
            "ECUP_VENDOR": _resolved(args.vendor),
            "ECUP_OUTPUT_DIR": _resolved(args.output_dir),
        }
    )
    return subprocess.run(
        [
            sys.executable,
            "-u",
            str(HERE / "train_screen.py"),
            "--fold",
            str(args.fold),
            "--preflight-audit",
            _resolved(args.preflight_audit),
            "--recombination-manifest",
            _resolved(args.recombination_manifest),
        ],
        cwd=HERE.parents[1],
        env=environment,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

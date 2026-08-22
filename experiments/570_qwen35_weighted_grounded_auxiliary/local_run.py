from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _resolved(value: Path) -> str:
    return str(value.expanduser().resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local interface for experiment 570.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Run one frozen screen fold.")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--oof", required=True, type=Path)
    train.add_argument("--audit-json", required=True, type=Path)
    train.add_argument("--image-manifest", required=True, type=Path)
    train.add_argument("--images", required=True, type=Path)
    train.add_argument("--model-root", required=True, type=Path)
    train.add_argument("--vendor", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--fold", required=True, type=int, choices=(0, 3))
    train.add_argument("--seed", type=int, default=42, choices=(42,))
    screen = commands.add_parser("screen", help="Evaluate both folds via frozen route400.")
    screen.add_argument("--fold-0", required=True, type=Path)
    screen.add_argument("--fold-3", required=True, type=Path)
    screen.add_argument("--output-dir", required=True, type=Path)
    null = commands.add_parser("null-control", help="Check exact route400 reconstruction.")
    null.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command in {"screen", "null-control"}:
        command = [sys.executable, "-u", str(HERE / "evaluate_screen.py")]
        if args.command == "null-control":
            command.append("--null-control")
        else:
            command.extend(
                ["--fold-0", _resolved(args.fold_0), "--fold-3", _resolved(args.fold_3)]
            )
        command.extend(["--output-dir", _resolved(args.output_dir)])
        return subprocess.run(command, cwd=HERE.parents[1], check=False).returncode

    environment = os.environ.copy()
    environment.update(
        {
            "ECUP_DATA": _resolved(args.data),
            "ECUP_OOF": _resolved(args.oof),
            "ECUP_OUTPUT_DIR": _resolved(args.output_dir),
            "ECUP_MANIFEST": _resolved(args.image_manifest),
            "ECUP_IMAGES": _resolved(args.images),
            "ECUP_MODEL_ROOT": _resolved(args.model_root),
            "ECUP_VENDOR": _resolved(args.vendor),
            "ECUP_GROUNDED_AUX_AUDIT_JSON": _resolved(args.audit_json),
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

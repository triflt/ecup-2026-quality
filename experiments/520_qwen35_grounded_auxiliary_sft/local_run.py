from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def resolved(value: Path) -> str:
    return str(value.expanduser().resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local interface for experiment 520.")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Audit grounded targets on outer-train records.")
    audit.add_argument("--data", required=True, type=Path)
    audit.add_argument("--oof", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    audit.add_argument("--fold", required=True, type=int, choices=(0, 3))
    audit.add_argument("--seed", type=int, default=42, choices=(42,))
    audit.add_argument("--cross-check-jsonl", type=Path)

    train = commands.add_parser("train", help="Run one frozen screen fold after a GO audit.")
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

    evaluate = commands.add_parser("evaluate", help="Audit already generated structured outputs.")
    evaluate.add_argument("--input", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--rendered-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "audit":
        command = [
            sys.executable,
            "-u",
            str(HERE / "coverage_audit.py"),
            "--data",
            resolved(args.data),
            "--oof",
            resolved(args.oof),
            "--output",
            resolved(args.output),
            "--fold",
            str(args.fold),
            "--seed",
            str(args.seed),
        ]
        if args.cross_check_jsonl:
            command.extend(["--cross-check-jsonl", resolved(args.cross_check_jsonl)])
        return subprocess.run(command, cwd=HERE.parents[1], check=False).returncode

    if args.command == "evaluate":
        command = [
            sys.executable,
            "-u",
            str(HERE / "evaluate_outputs.py"),
            "--input",
            resolved(args.input),
            "--output",
            resolved(args.output),
        ]
        if args.rendered_output:
            command.extend(["--rendered-output", resolved(args.rendered_output)])
        return subprocess.run(command, cwd=HERE.parents[1], check=False).returncode

    audit_path = args.audit_json.resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("decision") != "GO" or audit.get("holdout_fold") != args.fold:
        raise ValueError("training requires a GO coverage audit for the same fold")
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
            "ECUP_GROUNDED_AUX_AUDIT_JSON": resolved(args.audit_json),
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

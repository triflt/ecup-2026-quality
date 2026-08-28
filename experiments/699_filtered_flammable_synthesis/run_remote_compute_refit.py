from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runtime-map", type=Path, required=True)
    parser.add_argument("--runtime-map-contract", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.run_root.exists():
        raise FileExistsError("refusing to overwrite remote compute refit run")
    args.run_root.mkdir(parents=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.vendor.resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    runtime = args.run_root / "runtime" / "full_refit"
    ranked_args = [
        item
        for fold in range(5)
        for item in ("--ranked", f"{fold}={args.data_root / f'fold{fold}_ranked.jsonl'}")
    ]
    commands = [
        [
            sys.executable,
            str(Path(__file__).with_name("build_refit_runtime.py")),
            "--parent-root",
            str(args.data_root / "parent_runtime"),
            "--runtime-map",
            str(args.runtime_map),
            "--runtime-map-contract",
            str(args.runtime_map_contract),
            "--oof",
            str(args.oof),
            *ranked_args,
            "--epochs",
            str(args.epochs),
            "--output-dir",
            str(runtime),
        ],
        [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("train_synth_fold.py")),
            "--architecture",
            "qwen35_4b",
            "--fold",
            "-1",
            "--refit",
            "--runtime-dir",
            str(runtime),
            "--images",
            str(args.image_cache),
            "--model-root",
            "/models/qwen35_4b",
            "--model-revision",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "--vendor",
            str(args.vendor),
            "--output-dir",
            str(args.run_root / "output" / "full_refit"),
            "--epochs",
            str(args.epochs),
        ],
    ]
    for command in commands:
        print(json.dumps({"phase": "command", "argv": command}), flush=True)
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()

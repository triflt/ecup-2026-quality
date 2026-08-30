from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("qwen35_4b", "qwen3vl_2b"), required=True)
    parser.add_argument("--source", choices=("v1", "v2", "both"), required=True)
    parser.add_argument("--mode", choices=("positive_only", "balanced"), required=True)
    parser.add_argument("--cap", type=int, choices=(5, 10, 19, 40, 80), required=True)
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--synthetic-repeat", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    args = parser.parse_args()
    if args.run_root.exists():
        raise FileExistsError("refusing to overwrite remote compute run")
    args.run_root.mkdir(parents=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.vendor.resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    runtime = args.run_root / "runtime" / f"fold{args.fold}"
    commands = [
        [
            sys.executable,
            str(Path(__file__).with_name("build_append_runtime.py")),
            "--parent-dir", str(args.data_root / "parent_runtime" / f"fold{args.fold}"),
            "--ranked", str(args.data_root / f"fold{args.fold}_ranked.jsonl"),
            "--output-dir", str(runtime),
            "--fold", str(args.fold),
            "--source", args.source,
            "--mode", args.mode,
            "--cap", str(args.cap),
            "--epochs", str(args.epochs),
            "--synthetic-repeat", str(args.synthetic_repeat),
        ],
        [
            sys.executable, "-u", str(Path(__file__).with_name("train_synth_fold.py")),
            "--architecture", args.architecture,
            "--fold", str(args.fold),
            "--runtime-dir", str(runtime),
            "--images", str(args.image_cache),
            "--model-root", "/models/qwen35_4b" if args.architecture == "qwen35_4b" else "/models/qwen3vl_2b",
            "--model-revision", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" if args.architecture == "qwen35_4b" else "e2378df056d88153dc44616229fa371fcb87e236",
            "--vendor", str(args.vendor),
            "--output-dir", str(args.run_root / "output" / f"fold{args.fold}"),
            "--epochs", str(args.epochs),
        ],
    ]
    for command in commands:
        print(json.dumps({"phase": "command", "argv": command}), flush=True)
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()

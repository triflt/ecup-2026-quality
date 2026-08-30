from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODELS = {
    "qwen35_4b": (
        "/models/qwen35_4b",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    ),
    "qwen3vl_2b": (
        "/models/qwen3vl_2b",
        "e2378df056d88153dc44616229fa371fcb87e236",
    ),
}


def run(command: list[str], environment: dict[str, str]) -> None:
    print(json.dumps({"phase": "command", "argv": command}), flush=True)
    subprocess.run(command, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=tuple(MODELS), required=True)
    parser.add_argument("--source", choices=("v1", "v2", "both"), required=True)
    parser.add_argument("--mode", choices=("positive_only", "balanced"), required=True)
    parser.add_argument("--cap", type=int, choices=(19, 20, 40, 80), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), action="append", required=True)
    args = parser.parse_args()
    if args.run_root.exists():
        raise FileExistsError("refusing to overwrite remote compute run")
    args.run_root.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    model_root, revision = MODELS[args.architecture]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(args.vendor.resolve())
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    for fold in args.fold:
        runtime = args.run_root / "runtime" / f"fold{fold}"
        run(
            [
                sys.executable,
                str(repo / "experiments/699_filtered_flammable_synthesis/build_synth_runtime.py"),
                "--parent-dir",
                str(args.data_root / "parent_runtime" / f"fold{fold}"),
                "--ranked",
                str(args.data_root / f"fold{fold}_ranked.jsonl"),
                "--output-dir",
                str(runtime),
                "--fold",
                str(fold),
                "--source",
                args.source,
                "--mode",
                args.mode,
                "--cap",
                str(args.cap),
            ],
            environment,
        )
        run(
            [
                sys.executable,
                "-u",
                str(repo / "experiments/699_filtered_flammable_synthesis/train_synth_fold.py"),
                "--architecture",
                args.architecture,
                "--fold",
                str(fold),
                "--runtime-dir",
                str(runtime),
                "--images",
                str(args.image_cache),
                "--model-root",
                model_root,
                "--model-revision",
                revision,
                "--vendor",
                str(args.vendor),
                "--output-dir",
                str(args.run_root / "output" / f"fold{fold}"),
            ],
            environment,
        )


if __name__ == "__main__":
    main()

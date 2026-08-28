from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], env: dict[str, str]) -> None:
    print(json.dumps({"phase": "command", "argv": command}), flush=True)
    subprocess.run(command, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--reference-code", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--ranked-root", type=Path, required=True)
    parser.add_argument("--mounted-image-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--synth-cap", type=int, choices=(5, 10), required=True)
    parser.add_argument(
        "--preprocessing",
        choices=("thumbnail448", "area_cap262144"),
        required=True,
    )
    parser.add_argument("--reuse-image-cache", action="store_true")
    args = parser.parse_args()
    if args.run_root.exists():
        raise FileExistsError("refusing to overwrite corrected remote compute refit run")
    if (
        args.image_cache.exists()
        and any(args.image_cache.iterdir())
        and not args.reuse_image_cache
    ):
        raise FileExistsError("corrected Qwen3.5 image cache must start empty")
    args.run_root.mkdir(parents=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.vendor.resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    runtime = args.run_root / "runtime" / "solution140_full_support_refit"
    ranked_args = [
        item
        for fold in range(5)
        for item in (
            "--ranked",
            f"{fold}={args.ranked_root / f'fold{fold}_ranked.jsonl'}",
        )
    ]
    run_command(
        [
            sys.executable,
            str(
                Path(__file__).with_name(
                    "build_solution140_full_support_refit.py"
                )
            ),
            "--data",
            str(args.data),
            "--oof",
            str(args.oof),
            "--reference-code",
            str(args.reference_code),
            "--reference-report",
            str(args.reference_report),
            *ranked_args,
            "--synth-cap",
            str(args.synth_cap),
            "--preprocessing",
            args.preprocessing,
            "--epochs",
            "1",
            "--output-dir",
            str(runtime),
        ],
        env,
    )
    run_command(
        [
            sys.executable,
            str(
                Path(__file__).with_name(
                    "prepare_solution140_qwen35_image_cache.py"
                )
            ),
            "--runtime-dir",
            str(runtime),
            "--mounted-image-root",
            str(args.mounted_image_root),
            "--cache-root",
            str(args.image_cache),
            "--report",
            str(args.run_root / "image_cache_acceptance.json"),
            "--preprocessing",
            args.preprocessing,
            *(["--reuse-existing"] if args.reuse_image_cache else []),
        ],
        env,
    )
    run_command(
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
            str(args.model_root),
            "--model-revision",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "--vendor",
            str(args.vendor),
            "--output-dir",
            str(args.run_root / "output" / "full_refit"),
            "--epochs",
            "1",
        ],
        env,
    )


if __name__ == "__main__":
    main()

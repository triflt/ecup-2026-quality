from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

REVISIONS = {
    "qwen35_4b": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "qwen3vl_2b": "e2378df056d88153dc44616229fa371fcb87e236",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    if sha256(destination) != expected_sha256:
        raise ValueError(f"download SHA mismatch: {destination.name}")


def run_command(command: list[str]) -> None:
    print(json.dumps({"phase": "command", "argv": command}), flush=True)
    subprocess.run(command, check=True)


def safe_extract(bundle: Path, destination: Path) -> None:
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe bundle member: {member.name}")
        archive.extractall(destination, members=members, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=tuple(REVISIONS), required=True)
    parser.add_argument("--source", choices=("v1", "v2", "both"), required=True)
    parser.add_argument("--mode", choices=("positive_only", "balanced"), required=True)
    parser.add_argument("--cap", type=int, choices=(19, 20, 40, 80), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.bundle.exists():
        download(os.environ["BUNDLE_URL"], args.bundle, args.bundle_sha256)
    if sha256(args.bundle) != args.bundle_sha256:
        raise ValueError("screen bundle SHA mismatch")
    root = Path("/work/exp699")
    root.mkdir(parents=True, exist_ok=False)
    safe_extract(args.bundle, root)
    vendor = root / "vendor"
    run_command(
        [sys.executable, "-m", "zipfile", "-e", str(root / "peft-0.20.0.zip"), str(vendor)]
    )
    ranked_dir = root / "ranked"
    ranked_dir.mkdir(parents=True, exist_ok=False)
    for fold in (0, 3):
        ranked = ranked_dir / f"fold{fold}_ranked.jsonl"
        download(
            os.environ[f"FILTER_FOLD{fold}_URL"],
            ranked,
            os.environ[f"FILTER_FOLD{fold}_SHA256"],
        )
        runtime = root / "runtime" / f"fold{fold}"
        run_command(
            [
                sys.executable,
                str(root / "code" / "build_synth_runtime.py"),
                "--parent-dir",
                str(root / "parent_runtime" / f"fold{fold}"),
                "--ranked",
                str(ranked),
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
            ]
        )
        run_command(
            [
                sys.executable,
                "-u",
                str(root / "code" / "train_synth_fold.py"),
                "--architecture",
                args.architecture,
                "--fold",
                str(fold),
                "--runtime-dir",
                str(runtime),
                "--images",
                str(root / "images" / f"fold{fold}"),
                "--model-root",
                str(args.model_root),
                "--model-revision",
                REVISIONS[args.architecture],
                "--vendor",
                str(vendor),
                "--output-dir",
                str(args.output_root / f"fold{fold}"),
            ]
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
METHODS = {
    "causal": "experiments/693_qwen4_causal_distillation/run_all_folds.py",
    "hardneg": "experiments/694_qwen4_hardneg_curriculum/run_all_folds.py",
    "rank": "experiments/695_qwen4_hard_anchored_listwise/run_all_folds.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(name: str) -> PurePosixPath:
    value = PurePosixPath(name.removeprefix("./"))
    if (
        not value.parts
        or value.is_absolute()
        or ".." in value.parts
        or any(
            part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
            for part in value.parts
        )
    ):
        raise ValueError(f"unsafe archive member: {name}")
    return value


def write_member(source: BinaryIO, target: Path, mode: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    target.chmod(mode & 0o777)


def extract_archive(archive: Path, destination: Path, expected_sha256: str) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to replace extraction root: {destination}")
    if sha256_file(archive) != expected_sha256:
        raise ValueError(f"archive SHA-256 mismatch: {archive.name}")
    destination.mkdir(parents=True)
    names: set[str] = set()
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                relative = safe_relative(member.name)
                key = relative.as_posix()
                if key in names or not (member.isdir() or member.isfile()):
                    raise ValueError("duplicate or special tar member")
                names.add(key)
                target = destination.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    payload = source.extractfile(member)
                    if payload is None:
                        raise ValueError("tar member is unreadable")
                    write_member(payload, target, member.mode)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                relative = safe_relative(member.filename)
                key = relative.as_posix()
                if key in names or member.is_dir() and member.file_size != 0:
                    raise ValueError("duplicate or invalid zip member")
                names.add(key)
                target = destination.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    with source.open(member) as payload:
                        write_member(payload, target, member.external_attr >> 16 or 0o644)
    else:
        raise ValueError("unsupported archive format")


def require_fivefold(root: Path, filename: str) -> None:
    for fold in range(5):
        path = root / f"fold{fold}" / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"fivefold input is incomplete: {path}")


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def completed_outputs(output: Path) -> list[str]:
    return [
        f"{arm}/fold{fold}"
        for arm in ("control", "candidate")
        for fold in range(5)
        if (output / arm / f"fold{fold}" / "output_contract.json").is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--acceptance-sha256", required=True)
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--winner-sha256", required=True)
    parser.add_argument("--vendor-archive", type=Path, required=True)
    parser.add_argument("--vendor-sha256", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty job output")
    args.output.mkdir(parents=True, exist_ok=True)
    work = args.output.parent / ".qwen4_inputs"
    if work.exists():
        raise FileExistsError("refusing to replace staged job inputs")
    try:
        extract_archive(args.runtime_archive, work / "runtime_bundle", args.runtime_sha256)
        extract_archive(args.baseline_archive, work / "baseline_bundle", args.baseline_sha256)
        extract_archive(args.vendor_archive, work / "vendor_bundle", args.vendor_sha256)
        runtime = work / "runtime_bundle" / "runtime"
        baseline = work / "baseline_bundle"
        vendor = work / "vendor_bundle"
        require_fivefold(runtime, "runtime_audit.json")
        require_fivefold(baseline, "predictions.jsonl")
        if not (vendor / "peft" / "__init__.py").is_file():
            raise ValueError("PEFT vendor archive lacks peft/__init__.py at archive root")
        if sha256_file(args.acceptance) != args.acceptance_sha256:
            raise ValueError("exp692 acceptance file SHA-256 mismatch")
        if sha256_file(args.winner) != args.winner_sha256:
            raise ValueError("exp692 winner gate file SHA-256 mismatch")
        winner = json.loads(args.winner.read_text(encoding="utf-8"))
        selected_id = str(winner.get("selected_teacher_experiment_id"))
        selected_root = args.teacher_root / ("fivefold" if selected_id == "696" else "")
        if selected_id not in {"691", "696"}:
            raise ValueError("winner gate selected unknown teacher")
        require_fivefold(selected_root, "report.json")
        command = [
            "python3",
            "-u",
            str(args.code_root / METHODS[args.method]),
            "--runtime-root",
            str(runtime),
            "--teacher-root",
            str(args.teacher_root),
            "--teacher-acceptance",
            str(args.acceptance),
            "--teacher-acceptance-sha256",
            args.acceptance_sha256,
            "--teacher-winner",
            str(args.winner),
            "--teacher-winner-sha256",
            args.winner_sha256,
            "--runtime-bundle-sha256",
            args.runtime_sha256,
            "--baseline-bundle-sha256",
            args.baseline_sha256,
            "--baseline-root",
            str(baseline),
            "--images",
            str(work / "images"),
            "--model-root",
            str(args.model_root),
            "--model-revision",
            MODEL_REVISION,
            "--vendor",
            str(vendor),
            "--output-root",
            str(args.output),
            "--runtime-backend",
            "legacy_eager",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = ":".join(
            (
                str(args.code_root / "experiments/693_qwen4_causal_distillation"),
                str(args.code_root / "experiments/645_qwen_scale_2x3_gate"),
                str(vendor),
            )
        )
        subprocess.run(command, check=True, env=env)
        atomic_json(
            args.output / "JOB_SUCCESS.json",
            {"method": args.method, "completed_outputs": completed_outputs(args.output)},
        )
    except Exception as error:
        atomic_json(
            args.output / "JOB_FAILURE.json",
            {
                "method": args.method,
                "error_type": type(error).__name__,
                "completed_outputs": completed_outputs(args.output),
            },
        )
        raise


if __name__ == "__main__":
    main()

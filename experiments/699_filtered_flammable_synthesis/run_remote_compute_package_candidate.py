from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def regular_files(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"candidate contains a symlink: {path.relative_to(root)}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def safe_extract_exact(archive: Path, destination: Path) -> dict[str, str]:
    if destination.exists():
        raise FileExistsError("refusing to overwrite extracted submission")
    members: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            relative = PurePosixPath(info.filename)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or "" in relative.parts
                or "\\" in info.filename
            ):
                raise ValueError(f"unsafe ZIP member: {info.filename!r}")
            if info.filename in members:
                raise ValueError(f"duplicate ZIP member: {info.filename}")
            unix_mode = info.external_attr >> 16
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise ValueError(f"ZIP symlink is forbidden: {info.filename}")
            file_type = stat.S_IFMT(unix_mode)
            if not info.is_dir() and file_type not in {0, stat.S_IFREG}:
                raise ValueError(f"non-regular ZIP member: {info.filename}")
            members[info.filename] = info
        if not members or "run.py" not in members:
            raise ValueError("submission ZIP does not contain run.py")
        destination.mkdir(parents=True)
        bundle.extractall(destination)
    return regular_files(destination)


def run_capture(argv: list[str], *, path: Path, env: dict[str, str] | None = None) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite process log: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def wait_for(path: Path, timeout_seconds: int) -> None:
    started = time.monotonic()
    while not path.is_file():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(10)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke_root.exists():
        raise FileExistsError("refusing to overwrite package smoke root")
    wait_for(args.refit_output / "output_contract.json", args.wait_timeout_seconds)
    args.smoke_root.mkdir(parents=True)
    build_log = args.smoke_root / "package_build.json"
    run_capture(
        [
            sys.executable,
            str(args.package_builder),
            "--source",
            str(args.source),
            "--refit-output",
            str(args.refit_output),
            "--destination",
            str(args.candidate_dir),
            "--archive",
            str(args.archive),
            "--expected-source-run-sha256",
            args.expected_source_run_sha256,
            "--expected-runtime-contract",
            args.expected_runtime_contract,
        ],
        path=build_log,
    )
    build_report = json.loads(build_log.read_text(encoding="utf-8"))
    if build_report.get("decision") != "GO_PACKAGE_SMOKE":
        raise ValueError("package builder did not open runtime smoke")
    if build_report.get("archive_sha256") != sha256_file(args.archive):
        raise ValueError("package archive binding mismatch")
    extracted = args.smoke_root / "extracted"
    extracted_files = safe_extract_exact(args.archive, extracted)
    candidate_files = regular_files(args.candidate_dir)
    if extracted_files != candidate_files:
        raise ValueError("ZIP extraction differs from candidate directory")
    manifest = json.loads(
        (extracted / "exp699_candidate_manifest.json").read_text(encoding="utf-8")
    )
    manifest_payload = dict(manifest)
    manifest_self = manifest_payload.pop("self_sha256", None)
    if manifest_self != canonical_sha256(manifest_payload):
        raise ValueError("candidate manifest self-hash mismatch")

    output = args.smoke_root / "submission.csv"
    environment = dict(os.environ)
    environment.update(
        {
            "QWEN_EMBED_MODEL_PATH": "/models/qwen3vl_embedding_2b",
            "QWEN_INSTRUCT_MODEL_PATH": "/models/qwen3vl_2b",
            "QWEN35_MODEL_PATH": "/models/qwen35_4b",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
    run_capture(
        [
            sys.executable,
            "-u",
            str(extracted / "run.py"),
            "-i",
            str(args.smoke_input / "data.csv"),
            "-o",
            str(output),
        ],
        path=args.smoke_root / "runtime.log",
        env=environment,
    )
    verify_log = args.smoke_root / "smoke_acceptance.json"
    run_capture(
        [
            sys.executable,
            str(args.smoke_verifier),
            "verify",
            "--input-dir",
            str(args.smoke_input),
            "--output",
            str(output),
        ],
        path=verify_log,
    )
    acceptance = json.loads(verify_log.read_text(encoding="utf-8"))
    if acceptance.get("decision") != "ACCEPT_PACKAGE_RUNTIME_SMOKE":
        raise ValueError("package runtime smoke was not accepted")
    acceptance_payload = dict(acceptance)
    acceptance_self = acceptance_payload.pop("self_sha256", None)
    if acceptance_self != canonical_sha256(acceptance_payload):
        raise ValueError("smoke acceptance self-hash mismatch")
    report: dict[str, Any] = {
        "schema": "exp699_package_terminal_v1",
        "experiment_id": "699",
        "archive_sha256": sha256_file(args.archive),
        "archive_size": args.archive.stat().st_size,
        "candidate_manifest_self_sha256": manifest_self,
        "refit_output_contract_sha256": build_report["refit_output_contract_sha256"],
        "smoke_acceptance_self_sha256": acceptance_self,
        "smoke_output_sha256": acceptance["output_sha256"],
        "rows": acceptance["rows"],
        "public_used": False,
        "decision": "GO_PUBLIC_SUBMIT",
    }
    report["self_sha256"] = canonical_sha256(report)
    terminal = args.smoke_root / "terminal.json"
    terminal.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--refit-output", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--smoke-input", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--package-builder", type=Path, required=True)
    parser.add_argument("--smoke-verifier", type=Path, required=True)
    parser.add_argument("--expected-source-run-sha256", required=True)
    parser.add_argument("--expected-runtime-contract", required=True)
    parser.add_argument("--wait-timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

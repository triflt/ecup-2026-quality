from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import tempfile
from pathlib import Path, PurePosixPath

MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPERIMENT_DIR = "experiments/693_qwen4_causal_distillation"
JOBS = (
    ("qwen4-causal", "causal"),
    ("qwen4-hardneg", "hardneg"),
    ("qwen4-rank", "rank"),
)


def hex_sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def git_sha(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Git revision must be a full lowercase SHA")
    return value


def safe_token(value: str, label: str) -> str:
    if not value or any(character.isspace() for character in value):
        raise ValueError(f"unsafe {label}")
    return value


def safe_bucket(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{1,61}[A-Za-z0-9]", value):
        raise ValueError("unsafe bucket")
    return value


def safe_s3_prefix(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value.startswith("/")
        or value.startswith("//")
        or ".." in path.parts
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"unsafe S3 prefix: {value}")
    return value.rstrip("/")


def safe_file(value: str) -> str:
    if PurePosixPath(value).name != value or value in {".", ".."}:
        raise ValueError("remote file must be one safe basename")
    return value


def time_limit_minutes(value: str) -> float:
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?", value)
    if match is None or not any(match.groups()):
        raise ValueError("time limit must use HhMm syntax")
    minutes = int(match.group(1) or 0) * 60 + int(match.group(2) or 0)
    if minutes <= 0:
        raise ValueError("time limit must be positive")
    return float(minutes)


def input_spec(*, bucket: str, src: str, dst: str, file: str | None = None) -> list[str]:
    lines = [
        "    - type: s3msk",
        f"      src: {json.dumps(src)}",
    ]
    if file is not None:
        lines.append(f"      file: {json.dumps(file)}")
    lines.extend(
        (
            f"      dst: {json.dumps(dst)}",
            f"      bucket: {json.dumps(bucket)}",
        )
    )
    return lines


def output_spec(*, bucket: str, dst: str) -> list[str]:
    # Intentionally no upload_policies: remote compute uploads on every terminal state.
    return [
        "    - type: s3msk",
        "      src: /work/output",
        f"      dst: {json.dumps(dst)}",
        f"      bucket: {json.dumps(bucket)}",
        "      allow_empty: true",
    ]


def extraction_bootstrap(archive: str, expected_sha256: str) -> str:
    script = (
        "import pathlib,sys,tarfile;"
        "a=pathlib.Path(sys.argv[1]);d=pathlib.Path(sys.argv[2]);"
        "t=tarfile.open(a);m=t.getmembers();"
        "p=[pathlib.PurePosixPath(x.name) for x in m];"
        "n=[x.as_posix().removeprefix('./') for x in p];"
        "b=[x.name for x,y,z in zip(m,p,n) if y.is_absolute() or '..' in y.parts "
        "or not z or not (x.isfile() or x.isdir()) "
        "or any(q == '__MACOSX' or q == '.DS_Store' or q.startswith('._') for q in y.parts)];"
        "b and (_ for _ in ()).throw(ValueError('unsafe code archive'));"
        "len(n)!=len(set(n)) and (_ for _ in ()).throw(ValueError('duplicate members'));"
        "d.mkdir(parents=True);t.extractall(d,members=m)"
    )
    return (
        f"test \"$(sha256sum {shlex.quote(archive)} | cut -d' ' -f1)\" = "
        f'"{expected_sha256}" && '
        f"python3 -c {shlex.quote(script)} {shlex.quote(archive)} /work/code"
    )


def render(args: argparse.Namespace, *, job_name: str, method: str) -> str:
    code_archive = f"/work/input/code/{args.code_bundle_file}"
    runtime_archive = f"/work/input/runtime/{args.runtime_bundle_file}"
    baseline_archive = f"/work/input/baseline/{args.baseline_bundle_file}"
    vendor_archive = f"/work/input/vendor/{args.vendor_bundle_file}"
    acceptance = f"/work/input/acceptance/{args.acceptance_file}"
    bootstrap = extraction_bootstrap(code_archive, args.code_bundle_sha256)
    command = (
        f"set -euo pipefail; {bootstrap} && "
        f"python3 -u /work/code/{EXPERIMENT_DIR}/build_code_bundle.py verify "
        f"--root /work/code --expected-revision {args.code_revision} "
        f"--archive {shlex.quote(code_archive)} "
        f"--expected-bundle-sha256 {args.code_bundle_sha256} && "
        f"python3 -u /work/code/{EXPERIMENT_DIR}/remote_entrypoint.py "
        f"--method {method} --code-root /work/code "
        f"--runtime-archive {shlex.quote(runtime_archive)} "
        f"--runtime-sha256 {args.runtime_bundle_sha256} "
        f"--baseline-archive {shlex.quote(baseline_archive)} "
        f"--baseline-sha256 {args.baseline_bundle_sha256} "
        "--teacher-root /work/input/teacher "
        f"--acceptance {shlex.quote(acceptance)} "
        f"--acceptance-sha256 {args.acceptance_sha256} "
        f"--vendor-archive {shlex.quote(vendor_archive)} "
        f"--vendor-sha256 {args.vendor_bundle_sha256} "
        "--model-root /hf_models --output /work/output"
    )
    lines = [
        "job:",
        f"  generate_name: {job_name}",
        f"  time_limit: {args.time_limit}",
        f"  flavor: {json.dumps(args.h100_flavor)}",
        f"  region: {json.dumps(args.region)}",
        f"  image: {json.dumps(args.image)}",
        f"  preemption: {json.dumps(args.preemption)}",
        "  work_dir: /work",
        "  env:",
        '    TOKENIZERS_PARALLELISM: "false"',
        '    PYTORCH_ALLOC_CONF: "expandable_segments:True"',
        "  entrypoint: /bin/bash",
        "  args:",
        "    - -lc",
        f"    - {json.dumps(command)}",
        "  input:",
    ]
    for spec in (
        input_spec(
            bucket=args.input_bucket,
            src=args.code_bundle_src,
            file=args.code_bundle_file,
            dst=code_archive,
        ),
        input_spec(
            bucket=args.input_bucket,
            src=args.runtime_bundle_src,
            file=args.runtime_bundle_file,
            dst=runtime_archive,
        ),
        input_spec(
            bucket=args.input_bucket,
            src=args.baseline_bundle_src,
            file=args.baseline_bundle_file,
            dst=baseline_archive,
        ),
        input_spec(
            bucket=args.input_bucket,
            src=args.qwen27_output_src,
            dst="/work/input/teacher",
        ),
        input_spec(
            bucket=args.input_bucket,
            src=args.acceptance_output_src,
            file=args.acceptance_file,
            dst=acceptance,
        ),
        input_spec(
            bucket=args.input_bucket,
            src=args.vendor_bundle_src,
            file=args.vendor_bundle_file,
            dst=vendor_archive,
        ),
    ):
        lines.extend(spec)
    lines.extend(
        (
            "    - type: model_registry",
            f"      mrid: {json.dumps(args.model_mrid)}",
            "      dst: /hf_models/",
            "  output:",
        )
    )
    lines.extend(
        output_spec(
            bucket=args.output_bucket,
            dst=f"{args.output_prefix}/{job_name}",
        )
    )
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build exactly three secret-free remote compute student presets."
    )
    result.add_argument("--project", required=True)
    result.add_argument("--region", required=True)
    result.add_argument("--image", required=True)
    result.add_argument("--h100-flavor", required=True)
    result.add_argument("--time-limit", required=True)
    result.add_argument("--preemption", choices=("allowed", "forbidden"), required=True)
    result.add_argument("--input-bucket", required=True)
    result.add_argument("--output-bucket", required=True)
    result.add_argument("--code-bundle-src", required=True)
    result.add_argument("--code-bundle-file", required=True)
    result.add_argument("--code-bundle-sha256", required=True)
    result.add_argument("--code-revision", required=True)
    result.add_argument("--runtime-bundle-src", required=True)
    result.add_argument("--runtime-bundle-file", required=True)
    result.add_argument("--runtime-bundle-sha256", required=True)
    result.add_argument("--baseline-bundle-src", required=True)
    result.add_argument("--baseline-bundle-file", required=True)
    result.add_argument("--baseline-bundle-sha256", required=True)
    result.add_argument("--qwen27-output-src", required=True)
    result.add_argument("--acceptance-output-src", required=True)
    result.add_argument("--acceptance-file", required=True)
    result.add_argument("--acceptance-sha256", required=True)
    result.add_argument("--vendor-bundle-src", required=True)
    result.add_argument("--vendor-bundle-file", required=True)
    result.add_argument("--vendor-bundle-sha256", required=True)
    result.add_argument("--model-mrid", required=True)
    result.add_argument("--output-prefix", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def validate(args: argparse.Namespace) -> None:
    safe_token(args.project, "project")
    safe_token(args.region, "region")
    safe_token(args.image, "image")
    safe_token(args.h100_flavor, "H100 flavor")
    if "h100" not in args.h100_flavor.casefold():
        raise ValueError("H100 flavor argument must identify H100")
    time_limit_minutes(args.time_limit)
    args.input_bucket = safe_bucket(args.input_bucket)
    args.output_bucket = safe_bucket(args.output_bucket)
    for field in (
        "code_bundle_src",
        "runtime_bundle_src",
        "baseline_bundle_src",
        "qwen27_output_src",
        "acceptance_output_src",
        "vendor_bundle_src",
        "output_prefix",
    ):
        setattr(args, field, safe_s3_prefix(getattr(args, field)))
    for field in (
        "code_bundle_file",
        "runtime_bundle_file",
        "baseline_bundle_file",
        "acceptance_file",
        "vendor_bundle_file",
    ):
        setattr(args, field, safe_file(getattr(args, field)))
    for field in (
        "code_bundle_sha256",
        "runtime_bundle_sha256",
        "baseline_bundle_sha256",
        "acceptance_sha256",
        "vendor_bundle_sha256",
    ):
        setattr(args, field, hex_sha256(getattr(args, field)))
    args.code_revision = git_sha(args.code_revision)
    safe_token(args.model_mrid, "model MRID")
    if PurePosixPath(args.model_mrid).name != MODEL_REVISION:
        raise ValueError("model MRID must bind the frozen Qwen3.5-4B revision")


def build_all(args: argparse.Namespace) -> dict[str, object]:
    validate(args)
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite preset output directory")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    rendered = {
        f"{job_name}.yaml": render(args, job_name=job_name, method=method)
        for job_name, method in JOBS
    }
    with tempfile.TemporaryDirectory(
        prefix="qwen4_presets_", dir=args.output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "presets"
        staging.mkdir()
        for name, payload in rendered.items():
            (staging / name).write_text(payload, encoding="utf-8")
        os.replace(staging, args.output_dir)
    return {
        "schema_version": "qwen4_three_job_preset_build_v1",
        "jobs": [name for name, _ in JOBS],
        "files": sorted(rendered),
        "project_supplied": True,
        "model_revision": MODEL_REVISION,
        "presets_sha256": {
            name: hashlib.sha256(payload.encode()).hexdigest() for name, payload in rendered.items()
        },
        "submission_authorized": False,
    }


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(build_all(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

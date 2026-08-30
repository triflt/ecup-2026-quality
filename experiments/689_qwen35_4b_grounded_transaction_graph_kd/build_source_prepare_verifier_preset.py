"""Build a credential-free remote-first CPU verifier preset for exp689 PREPARE."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path, PurePosixPath

from build_source_prepare_preset import ALLOWED_REGIONS, input_lines, safe_extract, safe_key

EXPERIMENT = "experiments/689_qwen35_4b_grounded_transaction_graph_kd"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def s3_ref(bucket: str, key: str) -> str:
    key = safe_key(key)
    return f"s3://{bucket}{key}"


def directory_input_lines(name: str, bucket: str, key: str, dst: str) -> list[str]:
    return [
        "    - type: s3msk",
        f"      name: {json.dumps(name)}",
        f"      src: {json.dumps(safe_key(key))}",
        f"      dst: {json.dumps(dst)}",
        f"      bucket: {json.dumps(bucket)}",
    ]


def build(args: argparse.Namespace) -> str:
    if args.region not in ALLOWED_REGIONS:
        raise ValueError("source PREPARE verifier region is not approved")
    if not HEX40.fullmatch(args.revision):
        raise ValueError("revision must be exact lowercase Git SHA")
    hashes = (
        args.bundle_sha256,
        args.manifest_sha256,
        args.source_f03_sha256,
        args.source_f124_sha256,
        args.exclusion_670_sha256,
        args.exclusion_672_sha256,
        args.terminal_metadata_sha256,
    )
    if any(not HEX64.fullmatch(value) for value in hashes):
        raise ValueError("all verifier inputs require exact lowercase SHA-256")
    keys = tuple(
        safe_key(value)
        for value in (
            args.bundle_key,
            args.manifest_key,
            args.source_f03_key,
            args.source_f124_key,
            args.exclusion_670_key,
            args.exclusion_672_key,
            args.prepared_prefix,
            args.terminal_metadata_key,
            args.output_prefix,
        )
    )
    (
        bundle_key,
        manifest_key,
        source_f03_key,
        source_f124_key,
        exclusion_670_key,
        exclusion_672_key,
        prepared_prefix,
        terminal_metadata_key,
        output_prefix,
    ) = keys
    approved_ref = s3_ref(args.bucket, prepared_prefix)
    source_f03_ref = s3_ref(args.bucket, source_f03_key)
    source_f124_ref = s3_ref(args.bucket, source_f124_key)
    segments = [
        safe_extract(
            "/work/input/code/source_prepare_bundle.tar.gz",
            "/work/code",
            args.bundle_sha256,
        ),
        (
            "test \"$(sha256sum /work/input/manifest/bundle_manifest.json | cut -d' ' -f1)\" "
            f'= "{args.manifest_sha256}"'
        ),
        safe_extract(
            "/work/input/source_f03/source_f03.tar.gz",
            "/work/source_f03",
            args.source_f03_sha256,
        ),
        safe_extract(
            "/work/input/source_f124/source_f124.tar.gz",
            "/work/source_f124",
            args.source_f124_sha256,
        ),
        (
            f"PYTHONPATH=/work/code/{EXPERIMENT} python3 -u /work/code/{EXPERIMENT}/"
            "verify_source_prepare.py "
            "--prepare-dir /work/input/prepared/prepared "
            "--runtime-dir /work/source_f03/experiments/641_qwen35_4b_class_only_lora/.local/runtime/fold0 "
            "--runtime-dir /work/source_f124/runtime/fold1 "
            "--runtime-dir /work/source_f124/runtime/fold2 "
            "--runtime-dir /work/source_f03/experiments/641_qwen35_4b_class_only_lora/.local/runtime/fold3 "
            "--runtime-dir /work/source_f124/runtime/fold4 "
            "--runtime-archive /work/input/source_f03/source_f03.tar.gz "
            "--runtime-archive /work/input/source_f124/source_f124.tar.gz "
            f"--runtime-archive-ref {shlex.quote(source_f03_ref)} "
            f"--runtime-archive-ref {shlex.quote(source_f124_ref)} "
            "--exclusion-670 /work/input/exclusion_670/exp670.csv "
            "--exclusion-672 /work/input/exclusion_672/exp672.json "
            f"--exclusion-670-sha256 {args.exclusion_670_sha256} "
            f"--exclusion-672-sha256 {args.exclusion_672_sha256} "
            "--bundle-root /work/code "
            "--bundle-manifest /work/input/manifest/bundle_manifest.json "
            f"--bundle-manifest-sha256 {args.manifest_sha256} "
            f"--builder-revision {args.revision} "
            f"--terminal-metadata /work/input/terminal/{PurePosixPath(terminal_metadata_key).name} "
            f"--terminal-metadata-sha256 {args.terminal_metadata_sha256} "
            f"--approved-s3-output-ref {shlex.quote(approved_ref)} "
            "--acceptance /work/output/source_prepare_acceptance.json"
        ),
    ]
    lines = [
        "job:",
        "  generate_name: exp689-source-verify",
        "  time_limit: 2h",
        "  flavor: 8cpu-128ram",
        f"  region: {args.region}",
        "  image: odsai/ecup26-quality-baseline:1.0",
        "  preemption: forbidden",
        "  work_dir: /work",
        "  entrypoint: /bin/bash",
        "  args:",
        "    - -lc",
        "    - >-",
        f"      {' && '.join(segments)}",
        "  input:",
    ]
    file_specs = (
        ("code", bundle_key, "/work/input/code", "source_prepare_bundle.tar.gz"),
        ("manifest", manifest_key, "/work/input/manifest", "bundle_manifest.json"),
        ("source_f03", source_f03_key, "/work/input/source_f03", "source_f03.tar.gz"),
        (
            "source_f124",
            source_f124_key,
            "/work/input/source_f124",
            "source_f124.tar.gz",
        ),
        ("exclusion_670", exclusion_670_key, "/work/input/exclusion_670", "exp670.csv"),
        ("exclusion_672", exclusion_672_key, "/work/input/exclusion_672", "exp672.json"),
        (
            "terminal",
            terminal_metadata_key,
            "/work/input/terminal",
            PurePosixPath(terminal_metadata_key).name,
        ),
    )
    for name, key, dst, filename in file_specs:
        lines.extend(input_lines(name, args.bucket, key, dst, filename))
    lines.extend(
        directory_input_lines(
            "prepared", args.bucket, prepared_prefix, "/work/input/prepared"
        )
    )
    lines.extend(
        [
            "  output:",
            "    - type: s3msk",
            "      name: source_prepare_acceptance",
            "      src: /work/output",
            f"      dst: {json.dumps(output_prefix)}",
            f"      bucket: {json.dumps(args.bucket)}",
            "      upload_policies:",
            "        - when: on_job_status=succeeded",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--bundle-key", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-f03-key", required=True)
    parser.add_argument("--source-f03-sha256", required=True)
    parser.add_argument("--source-f124-key", required=True)
    parser.add_argument("--source-f124-sha256", required=True)
    parser.add_argument("--exclusion-670-key", required=True)
    parser.add_argument("--exclusion-670-sha256", required=True)
    parser.add_argument("--exclusion-672-key", required=True)
    parser.add_argument("--exclusion-672-sha256", required=True)
    parser.add_argument("--prepared-prefix", required=True)
    parser.add_argument("--terminal-metadata-key", required=True)
    parser.add_argument("--terminal-metadata-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite clean verifier preset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args), encoding="utf-8")


if __name__ == "__main__":
    main()

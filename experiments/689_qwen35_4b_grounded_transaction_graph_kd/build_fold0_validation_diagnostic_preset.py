"""Build the secret-free CPU-only exp689 fold0 binding diagnostic preset."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path, PurePosixPath

import extract_source_archive_transport as transport
from build_source_prepare_preset import ALLOWED_REGIONS, input_lines, safe_extract, safe_key

EXPERIMENT = "experiments/689_qwen35_4b_grounded_transaction_graph_kd"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _disjoint(output: str, inputs: list[str]) -> None:
    target = PurePosixPath(output)
    if any(
        target == PurePosixPath(value)
        or target.is_relative_to(PurePosixPath(value))
        or PurePosixPath(value).is_relative_to(target)
        for value in inputs
    ):
        raise ValueError("diagnostic output must be disjoint from every input")


def build(args: argparse.Namespace) -> str:
    if args.region not in ALLOWED_REGIONS:
        raise ValueError("diagnostic region is not approved")
    if not HEX40.fullmatch(args.revision):
        raise ValueError("diagnostic revision must be an exact Git SHA")
    hashes = (
        args.bundle_sha256,
        args.manifest_sha256,
        args.manifest_self_sha256,
        args.source_f03_sha256,
    )
    if any(not HEX64.fullmatch(value) for value in hashes):
        raise ValueError("diagnostic inputs require exact SHA-256 bindings")
    profile = transport.PROFILES["source_f03"]
    if (
        args.source_f03_sha256 != profile["sha256"]
        or args.source_f03_size_bytes != profile["size_bytes"]
    ):
        raise ValueError("diagnostic f03 archive differs from frozen profile")
    keys = {
        name: safe_key(getattr(args, name))
        for name in ("bundle_key", "manifest_key", "source_f03_key", "output_prefix")
    }
    _disjoint(
        keys["output_prefix"],
        [keys["bundle_key"], keys["manifest_key"], keys["source_f03_key"]],
    )
    runner = f"/work/code/{EXPERIMENT}/diagnose_fold0_validation_binding.py"
    segments = [
        safe_extract(
            "/work/input/code/fold0_validation_diagnostic_bundle.tar.gz",
            "/work/code",
            args.bundle_sha256,
        ),
        (
            "test \"$(sha256sum /work/input/manifest/bundle_manifest.json "
            "| cut -d' ' -f1)\" "
            f'= "{args.manifest_sha256}"'
        ),
        (
            "test \"$(sha256sum /work/input/source_f03/source_f03.tar.gz "
            "| cut -d' ' -f1)\" "
            f'= "{args.source_f03_sha256}"'
        ),
        (
            "test \"$(stat -c%s /work/input/source_f03/source_f03.tar.gz)\" "
            f'= "{args.source_f03_size_bytes}"'
        ),
        " ".join(
            shlex.quote(value)
            for value in (
                "python3",
                "-u",
                runner,
                "--archive",
                "/work/input/source_f03/source_f03.tar.gz",
                "--bundle-root",
                "/work/code",
                "--bundle-sha256",
                args.bundle_sha256,
                "--manifest",
                "/work/input/manifest/bundle_manifest.json",
                "--manifest-sha256",
                args.manifest_sha256,
                "--manifest-self-sha256",
                args.manifest_self_sha256,
                "--revision",
                args.revision,
                "--work-dir",
                "/work/diagnostic-work",
                "--output",
                "/work/output/fold0_validation_binding.json",
            )
        ),
    ]
    lines = [
        "job:",
        "  generate_name: exp689-fold0-bind-diag",
        "  time_limit: 30m",
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
    for name, key, destination, filename in (
        (
            "diag_code",
            keys["bundle_key"],
            "/work/input/code",
            "fold0_validation_diagnostic_bundle.tar.gz",
        ),
        (
            "diag_manifest",
            keys["manifest_key"],
            "/work/input/manifest",
            "bundle_manifest.json",
        ),
        (
            "source_f03",
            keys["source_f03_key"],
            "/work/input/source_f03",
            "source_f03.tar.gz",
        ),
    ):
        lines.extend(input_lines(name, args.bucket, key, destination, filename))
    lines.extend(
        [
            "  output:",
            "    - type: s3msk",
            "      name: fold0_bind_report",
            "      src: /work/output",
            f"      dst: {json.dumps(keys['output_prefix'])}",
            f"      bucket: {json.dumps(args.bucket)}",
            "      upload_policies:",
            "        - when: on_job_status=succeeded",
        ]
    )
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--region", required=True)
    value.add_argument("--bucket", required=True)
    value.add_argument("--revision", required=True)
    value.add_argument("--bundle-key", required=True)
    value.add_argument("--bundle-sha256", required=True)
    value.add_argument("--manifest-key", required=True)
    value.add_argument("--manifest-sha256", required=True)
    value.add_argument("--manifest-self-sha256", required=True)
    value.add_argument("--source-f03-key", required=True)
    value.add_argument("--source-f03-sha256", required=True)
    value.add_argument("--source-f03-size-bytes", type=int, required=True)
    value.add_argument("--output-prefix", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite diagnostic preset")
    preset = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(preset, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Build exp689 PREPARE retry preset with the exact AppleDouble transport fix."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

from build_source_prepare_preset import ALLOWED_REGIONS, input_lines, safe_extract, safe_key

EXPERIMENT = "experiments/689_qwen35_4b_grounded_transaction_graph_kd"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def build(args: argparse.Namespace) -> str:
    if args.region not in ALLOWED_REGIONS:
        raise ValueError("source PREPARE retry region is not approved")
    if not HEX40.fullmatch(args.revision):
        raise ValueError("revision must be an exact lowercase Git SHA")
    for value in (
        args.bundle_sha256,
        args.manifest_sha256,
        args.source_f03_sha256,
        args.source_f124_sha256,
        args.exclusion_670_sha256,
        args.exclusion_672_sha256,
        args.archive_acceptance_sha256,
    ):
        if not HEX64.fullmatch(value):
            raise ValueError("all retry inputs require exact lowercase SHA-256")
    keys = tuple(
        safe_key(value)
        for value in (
            args.bundle_key,
            args.manifest_key,
            args.source_f03_key,
            args.source_f124_key,
            args.exclusion_670_key,
            args.exclusion_672_key,
            args.archive_acceptance_key,
            args.output_prefix,
            args.transport_report_prefix,
        )
    )
    (
        bundle_key,
        manifest_key,
        source_f03_key,
        source_f124_key,
        exclusion_670_key,
        exclusion_672_key,
        archive_acceptance_key,
        output_prefix,
        transport_report_prefix,
    ) = keys
    segments = [
        safe_extract(
            "/work/input/code/source_prepare_retry_bundle.tar.gz",
            "/work/code",
            args.bundle_sha256,
        ),
        (
            'test "$(sha256sum /work/input/manifest/bundle_manifest.json | cut -d\' \' -f1)" '
            f'= "{args.manifest_sha256}"'
        ),
        (
            f"PYTHONPATH=/work/code/{EXPERIMENT} python3 -c "
            + shlex.quote(
                "from pathlib import Path;import verify_source_prepare as v;"
                f"v._validate_bundle(Path('/work/code'),Path('/work/input/manifest/bundle_manifest.json'),"
                f"'{args.manifest_sha256}','{args.revision}')"
            )
        ),
        (
            f"PYTHONPATH=/work/code/{EXPERIMENT} python3 -c "
            + shlex.quote(
                "from pathlib import Path;import extract_source_archive_transport as x;"
                "x.validate_diagnostic_acceptance("
                "Path('/work/input/archive_acceptance/source_archive_audit_acceptance.json'),"
                f"'{args.archive_acceptance_sha256}')"
            )
        ),
        (
            f"PYTHONPATH=/work/code/{EXPERIMENT} python3 -u /work/code/{EXPERIMENT}/"
            "extract_source_archive_transport.py "
            "--archive /work/input/source_f03/source_f03.tar.gz "
            "--archive-id source_f03 --destination /work/source_f03 "
            "--report /work/transport_reports/source_f03_extraction.json"
        ),
        (
            f"PYTHONPATH=/work/code/{EXPERIMENT} python3 -u /work/code/{EXPERIMENT}/"
            "extract_source_archive_transport.py "
            "--archive /work/input/source_f124/source_f124.tar.gz "
            "--archive-id source_f124 --destination /work/source_f124 "
            "--report /work/transport_reports/source_f124_extraction.json"
        ),
        (
            f"PYTHONPATH=/work/code/{EXPERIMENT} python3 -u /work/code/{EXPERIMENT}/"
            "prepare_source_universe.py "
            "--runtime-dir /work/source_f03/experiments/641_qwen35_4b_class_only_lora/.local/runtime/fold0 "
            "--runtime-dir /work/source_f124/runtime/fold1 "
            "--runtime-dir /work/source_f124/runtime/fold2 "
            "--runtime-dir /work/source_f03/experiments/641_qwen35_4b_class_only_lora/.local/runtime/fold3 "
            "--runtime-dir /work/source_f124/runtime/fold4 "
            "--exclusion-670 /work/input/exclusion_670/exp670.csv "
            "--exclusion-672 /work/input/exclusion_672/exp672.json "
            f"--exclusion-670-sha256 {args.exclusion_670_sha256} "
            f"--exclusion-672-sha256 {args.exclusion_672_sha256} "
            f"--builder-revision {args.revision} --output-dir /work/output/prepared"
        ),
    ]
    lines = [
        "job:",
        "  generate_name: exp689-source-prepare-retry",
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
    for name, key, dst, filename in (
        ("code", bundle_key, "/work/input/code", "source_prepare_retry_bundle.tar.gz"),
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
            "archive_accept",
            archive_acceptance_key,
            "/work/input/archive_acceptance",
            "source_archive_audit_acceptance.json",
        ),
    ):
        lines.extend(input_lines(name, args.bucket, key, dst, filename))
    lines.extend(
        [
            "  output:",
            "    - type: s3msk",
            "      name: source_prepare",
            "      src: /work/output",
            f"      dst: {json.dumps(output_prefix)}",
            f"      bucket: {json.dumps(args.bucket)}",
            "      upload_policies:",
            "        - when: on_job_status=succeeded",
            "    - type: s3msk",
            "      name: transport_reports",
            "      src: /work/transport_reports",
            f"      dst: {json.dumps(transport_report_prefix)}",
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
    value.add_argument("--source-f03-key", required=True)
    value.add_argument("--source-f03-sha256", required=True)
    value.add_argument("--source-f124-key", required=True)
    value.add_argument("--source-f124-sha256", required=True)
    value.add_argument("--exclusion-670-key", required=True)
    value.add_argument("--exclusion-670-sha256", required=True)
    value.add_argument("--exclusion-672-key", required=True)
    value.add_argument("--exclusion-672-sha256", required=True)
    value.add_argument("--archive-acceptance-key", required=True)
    value.add_argument("--archive-acceptance-sha256", required=True)
    value.add_argument("--output-prefix", required=True)
    value.add_argument("--transport-report-prefix", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite source PREPARE retry preset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args), encoding="utf-8")


if __name__ == "__main__":
    main()

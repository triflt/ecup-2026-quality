"""Build a transport-aware remote CPU verifier preset for exp689 PREPARE retry."""

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


def s3_ref(bucket: str, key: str) -> str:
    return f"s3://{bucket}{safe_key(key)}"


def directory_input_lines(name: str, bucket: str, key: str, dst: str) -> list[str]:
    return [
        "    - type: s3msk",
        f"      name: {json.dumps(name)}",
        f"      src: {json.dumps(safe_key(key))}",
        f"      dst: {json.dumps(dst)}",
        f"      bucket: {json.dumps(bucket)}",
    ]


def _non_overlapping(*keys: str) -> None:
    paths = [PurePosixPath(key) for key in keys]
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("verifier input/output prefixes must not overlap")


def build(args: argparse.Namespace) -> str:
    if args.region not in ALLOWED_REGIONS:
        raise ValueError("transport-aware verifier region is not approved")
    if not HEX40.fullmatch(args.verifier_revision) or not HEX40.fullmatch(
        args.retry_revision
    ):
        raise ValueError("verifier revisions must be exact lowercase Git SHAs")
    hash_fields = (
        "verifier_bundle_sha256",
        "verifier_manifest_sha256",
        "verifier_manifest_self_sha256",
        "verifier_runner_sha256",
        "retry_bundle_sha256",
        "retry_manifest_sha256",
        "retry_manifest_self_sha256",
        "extractor_sha256",
        "source_f03_sha256",
        "source_f124_sha256",
        "exclusion_670_sha256",
        "exclusion_672_sha256",
        "diagnostic_acceptance_sha256",
        "diagnostic_acceptance_self_sha256",
        "diagnostic_verifier_terminal_metadata_sha256",
        "retry_contract_sha256",
        "retry_contract_self_sha256",
        "retry_gate_sha256",
        "retry_gate_self_sha256",
        "retry_preset_builder_sha256",
        "terminal_metadata_sha256",
    )
    if any(not HEX64.fullmatch(getattr(args, field)) for field in hash_fields):
        raise ValueError("all transport-aware verifier inputs require exact SHA-256")
    if (
        args.source_f03_sha256 != transport.PROFILES["source_f03"]["sha256"]
        or args.source_f124_sha256
        != transport.PROFILES["source_f124"]["sha256"]
        or args.exclusion_670_sha256
        != transport.EXCLUSION_BINDINGS["exp670_audit_csv_sha256"]
        or args.exclusion_672_sha256
        != transport.EXCLUSION_BINDINGS["exp672_private_manifest_sha256"]
    ):
        raise ValueError("frozen source archive/exclusion binding mismatch")
    key_fields = (
        "verifier_bundle_key",
        "verifier_manifest_key",
        "retry_bundle_key",
        "retry_manifest_key",
        "source_f03_key",
        "source_f124_key",
        "exclusion_670_key",
        "exclusion_672_key",
        "prepared_prefix",
        "prepare_transport_report_prefix",
        "terminal_metadata_key",
        "diagnostic_acceptance_key",
        "retry_contract_key",
        "retry_gate_key",
        "output_prefix",
    )
    keys = {field: safe_key(getattr(args, field)) for field in key_fields}
    _non_overlapping(
        keys["prepared_prefix"],
        keys["prepare_transport_report_prefix"],
        keys["output_prefix"],
    )
    refs = {
        field.removesuffix("_key") + "_ref": s3_ref(args.bucket, value)
        for field, value in keys.items()
        if field.endswith("_key")
    }
    prepare_output_ref = s3_ref(args.bucket, keys["prepared_prefix"])
    runner_path = f"/work/verifier_code/{EXPERIMENT}/verify_source_prepare_retry.py"
    terminal_name = PurePosixPath(keys["terminal_metadata_key"]).name
    segments = [
        safe_extract(
            "/work/input/verifier_code/verifier_bundle.tar.gz",
            "/work/verifier_code",
            args.verifier_bundle_sha256,
        ),
        (
            "test \"$(sha256sum /work/input/verifier_manifest/bundle_manifest.json "
            "| cut -d' ' -f1)\" "
            f'= "{args.verifier_manifest_sha256}"'
        ),
        (
            f'test "$(sha256sum {runner_path} | cut -d\' \' -f1)" '
            f'= "{args.verifier_runner_sha256}"'
        ),
        safe_extract(
            "/work/input/retry_code/source_prepare_retry_bundle.tar.gz",
            "/work/retry_code",
            args.retry_bundle_sha256,
        ),
        (
            "test \"$(sha256sum /work/input/retry_manifest/bundle_manifest.json "
            "| cut -d' ' -f1)\" "
            f'= "{args.retry_manifest_sha256}"'
        ),
    ]
    command_args = [
        "python3",
        "-u",
        runner_path,
        "--verifier-bundle-root",
        "/work/verifier_code",
        "--verifier-bundle-sha256",
        args.verifier_bundle_sha256,
        "--verifier-manifest",
        "/work/input/verifier_manifest/bundle_manifest.json",
        "--verifier-manifest-sha256",
        args.verifier_manifest_sha256,
        "--verifier-manifest-self-sha256",
        args.verifier_manifest_self_sha256,
        "--verifier-revision",
        args.verifier_revision,
        "--verifier-runner-sha256",
        args.verifier_runner_sha256,
        "--retry-bundle-root",
        "/work/retry_code",
        "--retry-bundle-sha256",
        args.retry_bundle_sha256,
        "--retry-manifest",
        "/work/input/retry_manifest/bundle_manifest.json",
        "--retry-manifest-sha256",
        args.retry_manifest_sha256,
        "--retry-manifest-self-sha256",
        args.retry_manifest_self_sha256,
        "--retry-revision",
        args.retry_revision,
        "--extractor-sha256",
        args.extractor_sha256,
    ]
    file_arguments = (
        (
            "source-f03",
            "/work/input/source_f03/source_f03.tar.gz",
            args.source_f03_sha256,
            refs["source_f03_ref"],
        ),
        (
            "source-f124",
            "/work/input/source_f124/source_f124.tar.gz",
            args.source_f124_sha256,
            refs["source_f124_ref"],
        ),
        (
            "exclusion-670",
            "/work/input/exclusion_670/exp670.csv",
            args.exclusion_670_sha256,
            refs["exclusion_670_ref"],
        ),
        (
            "exclusion-672",
            "/work/input/exclusion_672/exp672.json",
            args.exclusion_672_sha256,
            refs["exclusion_672_ref"],
        ),
        (
            "diagnostic-acceptance",
            "/work/input/diagnostic/source_archive_audit_acceptance.json",
            args.diagnostic_acceptance_sha256,
            refs["diagnostic_acceptance_ref"],
        ),
        (
            "retry-contract",
            "/work/input/retry_contract/retry_preset_contract.json",
            args.retry_contract_sha256,
            refs["retry_contract_ref"],
        ),
        (
            "retry-gate",
            "/work/input/retry_gate/transport_retry_gate.json",
            args.retry_gate_sha256,
            refs["retry_gate_ref"],
        ),
        (
            "terminal-metadata",
            f"/work/input/terminal/{terminal_name}",
            args.terminal_metadata_sha256,
            refs["terminal_metadata_ref"],
        ),
    )
    for name, path, sha, reference in file_arguments:
        command_args.extend([f"--{name}", path, f"--{name}-sha256", sha, f"--{name}-ref", reference])
    command_args.extend(
        [
            "--diagnostic-acceptance-self-sha256",
            args.diagnostic_acceptance_self_sha256,
            "--diagnostic-verifier-terminal-metadata-sha256",
            args.diagnostic_verifier_terminal_metadata_sha256,
            "--retry-contract-self-sha256",
            args.retry_contract_self_sha256,
            "--retry-gate-self-sha256",
            args.retry_gate_self_sha256,
            "--retry-preset-builder-sha256",
            args.retry_preset_builder_sha256,
            "--retry-bundle-ref",
            refs["retry_bundle_ref"],
            "--retry-manifest-ref",
            refs["retry_manifest_ref"],
            "--prepare-output-prefix",
            keys["prepared_prefix"],
            "--prepare-transport-report-prefix",
            keys["prepare_transport_report_prefix"],
            "--prepare-output-ref",
            prepare_output_ref,
            "--prepare-dir",
            "/work/input/prepared/prepared",
            "--runtime-root",
            "/work/runtime",
            "--transport-report-dir",
            "/work/output/transport_reports",
            "--base-acceptance",
            "/work/output/source_prepare_acceptance.json",
            "--acceptance",
            "/work/output/source_prepare_retry_acceptance.json",
        ]
    )
    segments.append(" ".join(shlex.quote(value) for value in command_args))
    lines = [
        "job:",
        "  generate_name: exp689-source-retry-verify",
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
        (
            "verifier_code",
            keys["verifier_bundle_key"],
            "/work/input/verifier_code",
            "verifier_bundle.tar.gz",
        ),
        (
            "verifier_manifest",
            keys["verifier_manifest_key"],
            "/work/input/verifier_manifest",
            "bundle_manifest.json",
        ),
        (
            "retry_code",
            keys["retry_bundle_key"],
            "/work/input/retry_code",
            "source_prepare_retry_bundle.tar.gz",
        ),
        (
            "retry_manifest",
            keys["retry_manifest_key"],
            "/work/input/retry_manifest",
            "bundle_manifest.json",
        ),
        (
            "source_f03",
            keys["source_f03_key"],
            "/work/input/source_f03",
            "source_f03.tar.gz",
        ),
        (
            "source_f124",
            keys["source_f124_key"],
            "/work/input/source_f124",
            "source_f124.tar.gz",
        ),
        (
            "exclusion_670",
            keys["exclusion_670_key"],
            "/work/input/exclusion_670",
            "exp670.csv",
        ),
        (
            "exclusion_672",
            keys["exclusion_672_key"],
            "/work/input/exclusion_672",
            "exp672.json",
        ),
        (
            "diagnostic",
            keys["diagnostic_acceptance_key"],
            "/work/input/diagnostic",
            "source_archive_audit_acceptance.json",
        ),
        (
            "retry_contract",
            keys["retry_contract_key"],
            "/work/input/retry_contract",
            "retry_preset_contract.json",
        ),
        (
            "retry_gate",
            keys["retry_gate_key"],
            "/work/input/retry_gate",
            "transport_retry_gate.json",
        ),
        (
            "terminal",
            keys["terminal_metadata_key"],
            "/work/input/terminal",
            terminal_name,
        ),
    )
    for name, key, dst, filename in file_specs:
        lines.extend(input_lines(name, args.bucket, key, dst, filename))
    lines.extend(
        directory_input_lines(
            "prepared", args.bucket, keys["prepared_prefix"], "/work/input/prepared"
        )
    )
    lines.extend(
        [
            "  output:",
            "    - type: s3msk",
            "      name: source_prepare_retry_acceptance",
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
    for name in ("verifier", "retry"):
        value.add_argument(f"--{name}-bundle-key", required=True)
        value.add_argument(f"--{name}-bundle-sha256", required=True)
        value.add_argument(f"--{name}-manifest-key", required=True)
        value.add_argument(f"--{name}-manifest-sha256", required=True)
        value.add_argument(f"--{name}-manifest-self-sha256", required=True)
        value.add_argument(f"--{name}-revision", required=True)
    value.add_argument("--verifier-runner-sha256", required=True)
    value.add_argument("--extractor-sha256", required=True)
    for name in ("source-f03", "source-f124", "exclusion-670", "exclusion-672"):
        value.add_argument(f"--{name}-key", required=True)
        value.add_argument(f"--{name}-sha256", required=True)
    value.add_argument("--prepared-prefix", required=True)
    value.add_argument("--prepare-transport-report-prefix", required=True)
    value.add_argument("--terminal-metadata-key", required=True)
    value.add_argument("--terminal-metadata-sha256", required=True)
    value.add_argument("--diagnostic-acceptance-key", required=True)
    value.add_argument("--diagnostic-acceptance-sha256", required=True)
    value.add_argument("--diagnostic-acceptance-self-sha256", required=True)
    value.add_argument(
        "--diagnostic-verifier-terminal-metadata-sha256", required=True
    )
    value.add_argument("--retry-contract-key", required=True)
    value.add_argument("--retry-contract-sha256", required=True)
    value.add_argument("--retry-contract-self-sha256", required=True)
    value.add_argument("--retry-gate-key", required=True)
    value.add_argument("--retry-gate-sha256", required=True)
    value.add_argument("--retry-gate-self-sha256", required=True)
    value.add_argument("--retry-preset-builder-sha256", required=True)
    value.add_argument("--output-prefix", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite transport-aware verifier preset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args), encoding="utf-8")


if __name__ == "__main__":
    main()

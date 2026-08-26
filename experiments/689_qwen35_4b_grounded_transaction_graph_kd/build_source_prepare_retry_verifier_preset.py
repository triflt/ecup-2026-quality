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


def _non_overlapping(*, outputs: list[str], inputs: list[str]) -> None:
    output_paths = [PurePosixPath(key) for key in outputs]
    input_paths = [PurePosixPath(key) for key in inputs]
    pairs = [
        (left, right)
        for index, left in enumerate(output_paths)
        for right in output_paths[index + 1 :]
    ] + [(output, input_path) for output in output_paths for input_path in input_paths]
    if any(
        left == right or left.is_relative_to(right) or right.is_relative_to(left)
        for left, right in pairs
    ):
        raise ValueError("verifier outputs must be disjoint from all inputs and outputs")


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
        "submit_receipt_sha256",
        "submit_receipt_self_sha256",
        "live_go_sha256",
        "live_go_self_sha256",
        "materialization_receipt_sha256",
        "materialization_receipt_self_sha256",
        "resolved_terminal_metadata_sha256",
        "resolved_terminal_metadata_self_sha256",
        "terminal_transport_f03_sha256",
        "terminal_transport_f03_self_sha256",
        "terminal_transport_f124_sha256",
        "terminal_transport_f124_self_sha256",
        "expected_resolved_command_sha256",
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
        "diagnostic_acceptance_key",
        "retry_contract_key",
        "retry_gate_key",
        "submit_receipt_key",
        "live_go_key",
        "materialization_receipt_key",
        "resolved_terminal_metadata_key",
        "terminal_transport_f03_key",
        "terminal_transport_f124_key",
        "output_prefix",
    )
    keys = {field: safe_key(getattr(args, field)) for field in key_fields}
    expected_terminal_keys = {
        "terminal_transport_f03_key": (
            keys["prepare_transport_report_prefix"].rstrip("/")
            + "/source_f03_extraction.json"
        ),
        "terminal_transport_f124_key": (
            keys["prepare_transport_report_prefix"].rstrip("/")
            + "/source_f124_extraction.json"
        ),
    }
    if any(keys[field] != expected for field, expected in expected_terminal_keys.items()):
        raise ValueError("terminal transport report key differs from exact output inventory")
    _non_overlapping(
        outputs=[keys["output_prefix"]],
        inputs=[value for field, value in keys.items() if field != "output_prefix"],
    )
    refs = {
        field.removesuffix("_key") + "_ref": s3_ref(args.bucket, value)
        for field, value in keys.items()
        if field.endswith("_key")
    }
    prepare_output_ref = s3_ref(args.bucket, keys["prepared_prefix"])
    runner_path = f"/work/verifier_code/{EXPERIMENT}/verify_source_prepare_retry.py"
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
    )
    for name, path, sha, reference in file_arguments:
        command_args.extend(
            [f"--{name}", path, f"--{name}-sha256", sha, f"--{name}-ref", reference]
        )
    provenance_arguments = (
        (
            "submit-receipt",
            "/work/input/submit_receipt/submit_receipt.json",
            args.submit_receipt_sha256,
            args.submit_receipt_self_sha256,
            refs["submit_receipt_ref"],
        ),
        (
            "live-go",
            "/work/input/live_go/live_go.json",
            args.live_go_sha256,
            args.live_go_self_sha256,
            refs["live_go_ref"],
        ),
        (
            "materialization-receipt",
            "/work/input/materialization/materialization_receipt.json",
            args.materialization_receipt_sha256,
            args.materialization_receipt_self_sha256,
            refs["materialization_receipt_ref"],
        ),
        (
            "resolved-terminal-metadata",
            "/work/input/resolved_terminal/resolved_terminal_metadata.json",
            args.resolved_terminal_metadata_sha256,
            args.resolved_terminal_metadata_self_sha256,
            refs["resolved_terminal_metadata_ref"],
        ),
        (
            "terminal-transport-f03",
            "/work/input/terminal_transport_f03/source_f03_extraction.json",
            args.terminal_transport_f03_sha256,
            args.terminal_transport_f03_self_sha256,
            refs["terminal_transport_f03_ref"],
        ),
        (
            "terminal-transport-f124",
            "/work/input/terminal_transport_f124/source_f124_extraction.json",
            args.terminal_transport_f124_sha256,
            args.terminal_transport_f124_self_sha256,
            refs["terminal_transport_f124_ref"],
        ),
    )
    for name, path, file_sha, self_sha, reference in provenance_arguments:
        command_args.extend(
            [
                f"--{name}",
                path,
                f"--{name}-sha256",
                file_sha,
                f"--{name}-self-sha256",
                self_sha,
                f"--{name}-ref",
                reference,
            ]
        )
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
            "--expected-resolved-command-sha256",
            args.expected_resolved_command_sha256,
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
            "submit_receipt",
            keys["submit_receipt_key"],
            "/work/input/submit_receipt",
            "submit_receipt.json",
        ),
        (
            "live_go",
            keys["live_go_key"],
            "/work/input/live_go",
            "live_go.json",
        ),
        (
            "materialization",
            keys["materialization_receipt_key"],
            "/work/input/materialization",
            "materialization_receipt.json",
        ),
        (
            "resolved_terminal",
            keys["resolved_terminal_metadata_key"],
            "/work/input/resolved_terminal",
            "resolved_terminal_metadata.json",
        ),
        (
            "transport_f03",
            keys["terminal_transport_f03_key"],
            "/work/input/terminal_transport_f03",
            "source_f03_extraction.json",
        ),
        (
            "transport_f124",
            keys["terminal_transport_f124_key"],
            "/work/input/terminal_transport_f124",
            "source_f124_extraction.json",
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
            "      name: src_retry_accept",
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
    for name in (
        "diagnostic-acceptance",
        "retry-contract",
        "retry-gate",
        "submit-receipt",
        "live-go",
        "materialization-receipt",
        "resolved-terminal-metadata",
        "terminal-transport-f03",
        "terminal-transport-f124",
    ):
        value.add_argument(f"--{name}-key", required=True)
        value.add_argument(f"--{name}-sha256", required=True)
        value.add_argument(f"--{name}-self-sha256", required=True)
    value.add_argument(
        "--diagnostic-verifier-terminal-metadata-sha256", required=True
    )
    value.add_argument("--retry-preset-builder-sha256", required=True)
    value.add_argument("--expected-resolved-command-sha256", required=True)
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

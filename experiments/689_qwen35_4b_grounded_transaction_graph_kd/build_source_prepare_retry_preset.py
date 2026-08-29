"""Build exp689 PREPARE retry preset with the exact AppleDouble transport fix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any

import extract_source_archive_transport as transport
from build_source_prepare_preset import ALLOWED_REGIONS, input_lines, safe_extract, safe_key

EXPERIMENT = "experiments/689_qwen35_4b_grounded_transaction_graph_kd"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def preset_semantic_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze launch semantics without the gate hashes that authorize them.

    Binding the final preset SHA inside the gate and the gate SHA inside that same
    preset is a cryptographic cycle.  The independent gate therefore binds this
    complete semantic contract.  The sealed preset embeds the exact gate file and
    self hashes; a materialization receipt then binds the final preset bytes.
    """

    return {
        "schema_version": "exp689_source_prepare_retry_preset_contract_v1",
        "experiment_id": "689",
        "scope": "source_prepare_transport_retry_only",
        "retry_attempt": 1,
        "job": {
            "flavor": "8cpu-128ram",
            "time_limit": "2h",
            "region": args.region,
            "image": "odsai/ecup26-quality-baseline:1.0",
            "preemption": "forbidden",
            "gpu_count": 0,
        },
        "bucket": args.bucket,
        "revision": args.revision,
        "inputs": {
            "bundle": {"key": args.bundle_key, "sha256": args.bundle_sha256},
            "manifest": {
                "key": args.manifest_key,
                "sha256": args.manifest_sha256,
                "self_sha256": args.manifest_self_sha256,
            },
            "source_f03": {
                "key": args.source_f03_key,
                "sha256": args.source_f03_sha256,
            },
            "source_f124": {
                "key": args.source_f124_key,
                "sha256": args.source_f124_sha256,
            },
            "exclusion_670": {
                "key": args.exclusion_670_key,
                "sha256": args.exclusion_670_sha256,
            },
            "exclusion_672": {
                "key": args.exclusion_672_key,
                "sha256": args.exclusion_672_sha256,
            },
            "diagnostic_acceptance": {
                "key": args.archive_acceptance_key,
                "sha256": args.archive_acceptance_sha256,
                "self_sha256": args.archive_acceptance_self_sha256,
            },
            "diagnostic_verifier_terminal_metadata_sha256": (
                args.archive_verifier_terminal_metadata_sha256
            ),
            "preset_builder_sha256": args.retry_preset_builder_sha256,
            "preset_contract_key": args.retry_preset_contract_key,
            "transport_retry_gate_key": args.transport_retry_gate_key,
        },
        "outputs": {
            "source_prepare": args.output_prefix,
            "transport_reports": args.transport_report_prefix,
        },
        "command_plan": [
            "safe_extract_exact_retry_bundle",
            "validate_exact_bundle_manifest",
            "validate_diagnostic_acceptance_false",
            "validate_independent_transport_retry_gate_true",
            "extract_frozen_f03_skipping_exact_apple_metadata",
            "extract_frozen_f124_skipping_zero_apple_metadata",
            "run_unchanged_prepare_source_universe",
        ],
        "max_jobs": 1,
        "teacher_authorized": False,
        "model_authorized": False,
        "review_authorized": False,
        "student_gpu_authorized": False,
        "public_used": False,
    }


def preset_semantic_sha256(args: argparse.Namespace) -> str:
    return preset_contract_document(args)["self_sha256"]


def preset_contract_document(args: argparse.Namespace) -> dict[str, Any]:
    value = preset_semantic_contract(args)
    value["self_sha256"] = None
    value["self_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


def build(args: argparse.Namespace) -> str:
    if args.region not in ALLOWED_REGIONS:
        raise ValueError("source PREPARE retry region is not approved")
    if not HEX40.fullmatch(args.revision):
        raise ValueError("revision must be an exact lowercase Git SHA")
    for value in (
        args.bundle_sha256,
        args.manifest_sha256,
        args.manifest_self_sha256,
        args.source_f03_sha256,
        args.source_f124_sha256,
        args.exclusion_670_sha256,
        args.exclusion_672_sha256,
        args.archive_acceptance_sha256,
        args.archive_acceptance_self_sha256,
        args.archive_verifier_terminal_metadata_sha256,
        args.transport_retry_gate_sha256,
        args.transport_retry_gate_self_sha256,
        args.retry_preset_builder_sha256,
        args.retry_preset_contract_file_sha256,
        args.retry_preset_contract_sha256,
    ):
        if not HEX64.fullmatch(value):
            raise ValueError("all retry inputs require exact lowercase SHA-256")
    if (
        args.source_f03_sha256 != transport.PROFILES["source_f03"]["sha256"]
        or args.source_f124_sha256
        != transport.PROFILES["source_f124"]["sha256"]
        or args.exclusion_670_sha256
        != transport.EXCLUSION_BINDINGS["exp670_audit_csv_sha256"]
        or args.exclusion_672_sha256
        != transport.EXCLUSION_BINDINGS["exp672_private_manifest_sha256"]
    ):
        raise ValueError("retry preset frozen source/exclusion binding mismatch")
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
            args.retry_preset_contract_key,
            args.transport_retry_gate_key,
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
        retry_preset_contract_key,
        transport_retry_gate_key,
        output_prefix,
        transport_report_prefix,
    ) = keys
    semantic_sha256 = preset_semantic_sha256(args)
    if args.retry_preset_contract_sha256 != semantic_sha256:
        raise ValueError("retry preset semantic contract SHA mismatch")
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
                "c=x.validate_retry_preset_contract("
                "Path('/work/input/preset_contract/retry_preset_contract.json'),"
                f"expected_file_sha256='{args.retry_preset_contract_file_sha256}',"
                f"expected_self_sha256='{semantic_sha256}');"
                "d=x.validate_diagnostic_acceptance("
                "Path('/work/input/archive_acceptance/source_archive_audit_acceptance.json'),"
                f"'{args.archive_acceptance_sha256}');"
                "x.validate_transport_retry_gate("
                "Path('/work/input/retry_gate/transport_retry_gate.json'),"
                f"expected_file_sha256='{args.transport_retry_gate_sha256}',"
                f"expected_self_sha256='{args.transport_retry_gate_self_sha256}',"
                "diagnostic_acceptance=d,"
                f"diagnostic_acceptance_file_sha256='{args.archive_acceptance_sha256}',"
                "expected_verifier_terminal_metadata_sha256="
                f"'{args.archive_verifier_terminal_metadata_sha256}',"
                f"expected_retry_code_commit='{args.revision}',"
                f"expected_retry_code_bundle_sha256='{args.bundle_sha256}',"
                "expected_retry_bundle_manifest_file_sha256="
                f"'{args.manifest_sha256}',"
                "expected_retry_bundle_manifest_self_sha256="
                f"'{args.manifest_self_sha256}',"
                "expected_retry_preset_builder_sha256="
                f"'{args.retry_preset_builder_sha256}',"
                "retry_preset_contract=c,"
                "retry_preset_contract_file_sha256="
                f"'{args.retry_preset_contract_file_sha256}',"
                "expected_retry_preset_contract_sha256="
                f"'{semantic_sha256}',"
                f"source_prepare_spec_path=Path('/work/code/{EXPERIMENT}/source_prepare_spec_v1.json'),"
                f"expected_output_prefix='{output_prefix}',"
                "expected_transport_report_prefix="
                f"'{transport_report_prefix}')"
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
        (
            "preset_contract",
            retry_preset_contract_key,
            "/work/input/preset_contract",
            "retry_preset_contract.json",
        ),
        (
            "retry_gate",
            transport_retry_gate_key,
            "/work/input/retry_gate",
            "transport_retry_gate.json",
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
    value.add_argument("--manifest-self-sha256", required=True)
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
    value.add_argument("--archive-acceptance-self-sha256", required=True)
    value.add_argument(
        "--archive-verifier-terminal-metadata-sha256", required=True
    )
    value.add_argument("--transport-retry-gate-key", required=True)
    value.add_argument("--transport-retry-gate-sha256", required=True)
    value.add_argument("--transport-retry-gate-self-sha256", required=True)
    value.add_argument("--retry-preset-builder-sha256", required=True)
    value.add_argument("--retry-preset-contract-key", required=True)
    value.add_argument("--retry-preset-contract-file-sha256", required=True)
    value.add_argument("--retry-preset-contract-sha256", required=True)
    value.add_argument("--output-prefix", required=True)
    value.add_argument("--transport-report-prefix", required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--materialization-receipt-output", type=Path, required=True)
    return value


def materialization_receipt(
    args: argparse.Namespace, preset_text: str
) -> dict[str, Any]:
    encoded = preset_text.encode("utf-8")
    receipt = {
        "schema_version": "exp689_source_prepare_retry_materialization_v1",
        "experiment_id": "689",
        "scope": "source_prepare_transport_retry_only",
        "retry_attempt": 1,
        "diagnostic_acceptance_file_sha256": args.archive_acceptance_sha256,
        "diagnostic_acceptance_self_sha256": args.archive_acceptance_self_sha256,
        "transport_retry_gate_file_sha256": args.transport_retry_gate_sha256,
        "transport_retry_gate_self_sha256": args.transport_retry_gate_self_sha256,
        "preset_contract_file_sha256": args.retry_preset_contract_file_sha256,
        "preset_semantic_contract_sha256": args.retry_preset_contract_sha256,
        "final_preset_sha256": hashlib.sha256(encoded).hexdigest(),
        "final_preset_size_bytes": len(encoded),
        "retry_preset_builder_sha256": args.retry_preset_builder_sha256,
        "controlled_prepare_retry_authorized": True,
        "max_jobs": 1,
        "teacher_authorized": False,
        "student_gpu_authorized": False,
        "public_used": False,
        "self_sha256": None,
    }
    receipt["self_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    return receipt


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() or args.materialization_receipt_output.exists():
        raise FileExistsError("refusing to overwrite retry materialization outputs")
    text = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    receipt = materialization_receipt(args, text)
    args.materialization_receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.materialization_receipt_output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

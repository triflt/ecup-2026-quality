"""Build the independent, one-job transport authorization for exp689 PREPARE.

This document is intentionally separate from the immutable diagnostic acceptance.
The diagnostic proves the cause and keeps ``prepare_retry_authorized=false``.  Only
an independent integrator may issue this gate after accepting the terminal remote
verifier and the frozen retry packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import extract_source_archive_transport as transport

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _exact_sha(value: str, name: str) -> str:
    if not HEX64.fullmatch(value):
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.issuer_role != "independent_integrator":
        raise ValueError("transport retry gate requires the independent integrator")
    if not HEX40.fullmatch(args.retry_code_commit):
        raise ValueError("retry code commit must be an exact lowercase Git SHA")
    exact_sha_fields = (
        "diagnostic_acceptance_file_sha256",
        "diagnostic_verifier_terminal_metadata_sha256",
        "retry_code_bundle_sha256",
        "retry_bundle_manifest_file_sha256",
        "retry_bundle_manifest_self_sha256",
        "retry_preset_builder_sha256",
        "retry_preset_contract_file_sha256",
        "retry_preset_contract_self_sha256",
    )
    for field in exact_sha_fields:
        _exact_sha(getattr(args, field), field)
    diagnostic = transport.validate_diagnostic_acceptance(
        args.diagnostic_acceptance,
        args.diagnostic_acceptance_file_sha256,
    )
    _, selector_sha = transport.validate_source_prepare_spec(
        args.source_prepare_spec
    )
    preset_contract = transport.validate_retry_preset_contract(
        args.retry_preset_contract,
        expected_file_sha256=args.retry_preset_contract_file_sha256,
        expected_self_sha256=args.retry_preset_contract_self_sha256,
    )
    contract_inputs = preset_contract["inputs"]
    if (
        preset_contract["revision"] != args.retry_code_commit
        or contract_inputs["bundle"]["sha256"]
        != args.retry_code_bundle_sha256
        or contract_inputs["manifest"]["sha256"]
        != args.retry_bundle_manifest_file_sha256
        or contract_inputs["manifest"]["self_sha256"]
        != args.retry_bundle_manifest_self_sha256
        or contract_inputs["preset_builder_sha256"]
        != args.retry_preset_builder_sha256
        or contract_inputs["source_f03"]["sha256"]
        != transport.PROFILES["source_f03"]["sha256"]
        or contract_inputs["source_f124"]["sha256"]
        != transport.PROFILES["source_f124"]["sha256"]
        or contract_inputs["exclusion_670"]["sha256"]
        != transport.EXCLUSION_BINDINGS["exp670_audit_csv_sha256"]
        or contract_inputs["exclusion_672"]["sha256"]
        != transport.EXCLUSION_BINDINGS["exp672_private_manifest_sha256"]
        or contract_inputs["diagnostic_acceptance"]["sha256"]
        != args.diagnostic_acceptance_file_sha256
        or contract_inputs["diagnostic_acceptance"]["self_sha256"]
        != diagnostic["self_sha256"]
        or contract_inputs["diagnostic_verifier_terminal_metadata_sha256"]
        != args.diagnostic_verifier_terminal_metadata_sha256
        or preset_contract["outputs"]["source_prepare"] != args.output_prefix
        or preset_contract["outputs"]["transport_reports"]
        != args.transport_report_prefix
    ):
        raise ValueError("retry preset contract does not match the gate packet")
    archives = {
        archive_id: {
            "sha256": profile["sha256"],
            "size_bytes": profile["size_bytes"],
        }
        for archive_id, profile in transport.PROFILES.items()
    }
    gate = {
        "schema_version": "exp689_source_prepare_transport_retry_gate_v1",
        "experiment_id": "689",
        "scope": "source_prepare_transport_retry_only",
        "issuer_role": args.issuer_role,
        "decision": "OPEN_CONTROLLED_SOURCE_PREPARE_RETRY",
        "diagnostic_acceptance_file_sha256": (
            args.diagnostic_acceptance_file_sha256
        ),
        "diagnostic_acceptance_self_sha256": diagnostic["self_sha256"],
        "diagnostic_verifier_terminal_metadata_sha256": (
            args.diagnostic_verifier_terminal_metadata_sha256
        ),
        "frozen_archives": archives,
        "retry_code_commit": args.retry_code_commit,
        "retry_code_bundle_sha256": args.retry_code_bundle_sha256,
        "retry_bundle_manifest_file_sha256": (
            args.retry_bundle_manifest_file_sha256
        ),
        "retry_bundle_manifest_self_sha256": (
            args.retry_bundle_manifest_self_sha256
        ),
        "retry_preset_builder_sha256": args.retry_preset_builder_sha256,
        "retry_preset_contract_file_sha256": (
            args.retry_preset_contract_file_sha256
        ),
        "retry_preset_contract_sha256": preset_contract["self_sha256"],
        "source_prepare_spec_file_sha256": (
            transport.SOURCE_PREPARE_SPEC_FILE_SHA256
        ),
        "source_prepare_spec_self_sha256": (
            transport.SOURCE_PREPARE_SPEC_SELF_SHA256
        ),
        "selector_contract_sha256": selector_sha,
        "exclusion_bindings": transport.EXCLUSION_BINDINGS,
        "runtime_bindings": transport.RUNTIME_BINDINGS,
        "output_prefix": args.output_prefix,
        "transport_report_prefix": args.transport_report_prefix,
        "controlled_prepare_retry_authorized": True,
        "max_jobs": 1,
        "teacher_authorized": False,
        "model_authorized": False,
        "review_authorized": False,
        "student_gpu_authorized": False,
        "public_used": False,
        "self_sha256": None,
    }
    gate["self_sha256"] = hashlib.sha256(
        transport.canonical_json_bytes(gate)
    ).hexdigest()
    return gate


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--issuer-role", required=True)
    value.add_argument("--diagnostic-acceptance", type=Path, required=True)
    value.add_argument("--diagnostic-acceptance-file-sha256", required=True)
    value.add_argument(
        "--diagnostic-verifier-terminal-metadata-sha256", required=True
    )
    value.add_argument("--retry-code-commit", required=True)
    value.add_argument("--retry-code-bundle-sha256", required=True)
    value.add_argument("--retry-bundle-manifest-file-sha256", required=True)
    value.add_argument("--retry-bundle-manifest-self-sha256", required=True)
    value.add_argument("--retry-preset-builder-sha256", required=True)
    value.add_argument("--retry-preset-contract", type=Path, required=True)
    value.add_argument("--retry-preset-contract-file-sha256", required=True)
    value.add_argument("--retry-preset-contract-self-sha256", required=True)
    value.add_argument("--source-prepare-spec", type=Path, required=True)
    value.add_argument("--output-prefix", required=True)
    value.add_argument("--transport-report-prefix", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite transport retry gate")
    gate = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Issue an independently auditable promotion gate for the exp689 teacher smoke."""

from __future__ import annotations

import argparse
import json
import re
import stat
import zipfile
from pathlib import Path
from typing import Any

from build_target_audit import (
    ContractError,
    canonical_json_bytes,
    expect_exact_keys,
    load_json,
    require_hex64,
    sha256_bytes,
    sha256_file,
    validate_self_hash,
    with_self_hash,
    write_json,
)
from run_teacher import MODEL_ID, MODEL_REVISION, REMOTE_SMOKE_ACCEPTANCE_FIELDS
from verify_teacher_run import (
    RECEIPT_FIELDS,
    validate_code_bundle,
    validate_model_contract,
)

VERIFIER_BUNDLE_MEMBERS = {
    "build_target_audit.py",
    "build_teacher_model_contract.py",
    "build_teacher_smoke_promotion_gate.py",
    "run_teacher.py",
    "verify_teacher_run.py",
}
TERMINAL_METADATA_FIELDS = {
    "schema_version",
    "job_identity_sha256",
    "terminal_state",
    "exit_code",
    "finished_at_utc",
    "receipt_sha256",
    "output_ref_sha256",
    "state_source",
    "self_sha256",
}
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def validate_verifier_bundle(
    path: Path,
    *,
    expected_sha256: str,
    expected_verifier_sha256: str,
    expected_gate_builder_sha256: str,
) -> dict[str, str]:
    for value, context in (
        (expected_sha256, "verifier-bundle SHA"),
        (expected_verifier_sha256, "verifier-code SHA"),
        (expected_gate_builder_sha256, "gate-builder SHA"),
    ):
        require_hex64(value, context)
    if sha256_file(path) != expected_sha256:
        raise ContractError("verifier bundle SHA mismatch")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != VERIFIER_BUNDLE_MEMBERS:
                raise ContractError("verifier bundle exact member whitelist mismatch")
            members: dict[str, str] = {}
            for info in infos:
                member = Path(info.filename)
                mode = info.external_attr >> 16
                if (
                    info.is_dir()
                    or member.is_absolute()
                    or ".." in member.parts
                    or stat.S_ISLNK(mode)
                ):
                    raise ContractError("verifier bundle contains an unsafe member")
                members[info.filename] = sha256_bytes(archive.read(info))
            if archive.testzip() is not None:
                raise ContractError("verifier bundle integrity check failed")
    except (OSError, zipfile.BadZipFile) as error:
        raise ContractError("verifier bundle is unreadable or corrupt") from error
    if members["verify_teacher_run.py"] != expected_verifier_sha256:
        raise ContractError("verifier bundle verifier-code SHA mismatch")
    if members["build_teacher_smoke_promotion_gate.py"] != expected_gate_builder_sha256:
        raise ContractError("verifier bundle gate-builder SHA mismatch")
    return dict(sorted(members.items()))


def validate_terminal_metadata(
    value: dict[str, Any],
    *,
    file_sha256: str,
    receipt: dict[str, Any],
    receipt_file_sha256: str,
    output_ref_sha256: str,
) -> None:
    expect_exact_keys(value, TERMINAL_METADATA_FIELDS, "teacher terminal metadata")
    validate_self_hash(value, "teacher terminal metadata")
    if value["schema_version"] != "exp689_teacher_terminal_metadata_v1":
        raise ContractError("teacher terminal metadata schema mismatch")
    if (
        value["terminal_state"] != "SUCCESS"
        or value["exit_code"] != 0
        or value["state_source"] != "remote_compute_remote_api_independent"
    ):
        raise ContractError("teacher terminal metadata is not independent SUCCESS")
    if value["job_identity_sha256"] != receipt["job_identity_sha256"]:
        raise ContractError("teacher terminal metadata job binding mismatch")
    if (
        not isinstance(value["finished_at_utc"], str)
        or not TIMESTAMP.fullmatch(value["finished_at_utc"])
        or value["finished_at_utc"] != receipt["finished_at_utc"]
    ):
        raise ContractError("teacher terminal metadata timestamp mismatch")
    if value["receipt_sha256"] != receipt_file_sha256:
        raise ContractError("teacher terminal metadata receipt SHA mismatch")
    if value["output_ref_sha256"] != output_ref_sha256:
        raise ContractError("teacher terminal metadata output-ref SHA mismatch")
    require_hex64(file_sha256, "terminal-metadata file SHA")


def build(
    *,
    remote_acceptance_path: Path,
    expected_remote_acceptance_sha256: str,
    remote_receipt_path: Path,
    terminal_metadata_path: Path,
    teacher_code_bundle_path: Path,
    model_contract_path: Path,
    verifier_bundle_path: Path,
    expected_verifier_bundle_sha256: str,
    expected_verifier_sha256: str,
    expected_gate_builder_sha256: str,
    expected_commit: str,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite immutable teacher smoke gate")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ContractError("expected smoke commit must be exact lowercase Git SHA")
    require_hex64(expected_remote_acceptance_sha256, "remote acceptance file SHA")
    if sha256_file(remote_acceptance_path) != expected_remote_acceptance_sha256:
        raise ContractError("remote acceptance file SHA mismatch")
    acceptance = load_json(remote_acceptance_path, "teacher remote acceptance")
    expect_exact_keys(
        acceptance, REMOTE_SMOKE_ACCEPTANCE_FIELDS, "teacher remote acceptance"
    )
    validate_self_hash(acceptance, "teacher remote acceptance")
    expected_acceptance = {
        "schema_version": "exp689_teacher_remote_acceptance_v1",
        "experiment_id": "689",
        "scope": "technical_smoke",
        "status": "accepted",
        "decision": "OPEN_FULL_TEACHER",
        "technical_only": True,
        "quality_evaluated": False,
        "student_gpu_authorized": False,
        "terminal_job_metadata_bound": True,
        "approved_remote_output_bound": True,
        "full_teacher_technical_gate_open": True,
        "accepted_smoke_self_sha256": None,
        "accepted_smoke_promotion_gate_self_sha256": None,
        "commit_sha": expected_commit,
        "labels_read": 0,
        "sealed_rows": 0,
        "public_used": False,
        "jobs_launched_by_verifier": 0,
        "uploads_by_verifier": 0,
        "presets_built_by_verifier": 0,
        "bundles_built_by_verifier": 0,
    }
    for field, expected in expected_acceptance.items():
        if acceptance[field] != expected:
            raise ContractError(f"teacher remote acceptance {field} mismatch")

    teacher_members = validate_code_bundle(
        teacher_code_bundle_path,
        expected_sha256=acceptance["code_bundle_sha256"],
        expected_runner_sha256=acceptance["runner_sha256"],
    )
    if teacher_members != acceptance["code_bundle_members"]:
        raise ContractError("teacher remote acceptance code members mismatch")

    model_contract = load_json(model_contract_path, "teacher model contract")
    if sha256_file(model_contract_path) != acceptance["model_contract_sha256"]:
        raise ContractError("teacher remote acceptance model-contract file mismatch")
    validate_model_contract(model_contract, acceptance["model_contract_sha256"])
    model_expected = {
        "model_contract_self_sha256": model_contract["self_sha256"],
        "model_tree_sha256": model_contract["model_tree_sha256"],
        "processor_sha256": model_contract["processor_sha256"],
    }
    for field, expected in model_expected.items():
        if acceptance[field] != expected:
            raise ContractError(f"teacher remote acceptance {field} mismatch")

    receipt_file_sha = sha256_file(remote_receipt_path)
    if receipt_file_sha != acceptance["remote_receipt_sha256"]:
        raise ContractError("teacher remote acceptance receipt file SHA mismatch")
    receipt = load_json(remote_receipt_path, "teacher remote receipt")
    expect_exact_keys(receipt, RECEIPT_FIELDS, "teacher remote receipt")
    validate_self_hash(receipt, "teacher remote receipt")
    if receipt["self_sha256"] != acceptance["remote_receipt_self_sha256"]:
        raise ContractError("teacher remote acceptance receipt self SHA mismatch")
    if (
        receipt["terminal_state"] != "SUCCESS"
        or receipt["exit_code"] != 0
        or receipt["failure_reason"] is not None
        or receipt["scope"] != "technical_smoke"
        or receipt["commit_sha"] != expected_commit
        or receipt["code_bundle_sha256"] != acceptance["code_bundle_sha256"]
        or receipt["code_bundle_members"] != teacher_members
        or receipt["runner_sha256"] != acceptance["runner_sha256"]
        or receipt["teacher_request_sha256"] != acceptance["source_sha256"]
        or receipt["image_manifest_sha256"] != acceptance["image_manifest_sha256"]
        or receipt["pixel_set_sha256"] != acceptance["pixel_set_sha256"]
        or receipt["labels_read"] != 0
        or receipt["sealed_rows"] != 0
        or receipt["public_used"] is not False
    ):
        raise ContractError("teacher remote receipt identity/provenance mismatch")
    registry = receipt["model_registry_input"]
    registry_expected = {
        "input_identity_sha256": acceptance["model_registry_input_identity_sha256"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_contract_sha256": acceptance["model_contract_sha256"],
        "model_contract_self_sha256": acceptance["model_contract_self_sha256"],
        "model_tree_sha256": acceptance["model_tree_sha256"],
        "processor_sha256": acceptance["processor_sha256"],
    }
    if any(registry.get(field) != expected for field, expected in registry_expected.items()):
        raise ContractError("teacher remote receipt model-registry binding mismatch")
    output_ref_sha = sha256_bytes(canonical_json_bytes(receipt["output_ref"]))
    if output_ref_sha != acceptance["remote_output_ref_sha256"]:
        raise ContractError("teacher remote acceptance output-ref SHA mismatch")
    objects = receipt["output_ref"].get("objects")
    inventory = acceptance["runner_output_inventory"]
    if acceptance["runner_output_inventory_sha256"] != sha256_bytes(
        canonical_json_bytes(inventory)
    ):
        raise ContractError("teacher remote acceptance output-inventory SHA mismatch")
    if (
        not isinstance(objects, list)
        or {item.get("name"): item.get("sha256") for item in objects} != inventory
        or any(not item.get("version_id") for item in objects)
    ):
        raise ContractError("teacher remote receipt versioned output inventory mismatch")

    terminal_file_sha = sha256_file(terminal_metadata_path)
    terminal = load_json(terminal_metadata_path, "teacher terminal metadata")
    validate_terminal_metadata(
        terminal,
        file_sha256=terminal_file_sha,
        receipt=receipt,
        receipt_file_sha256=receipt_file_sha,
        output_ref_sha256=output_ref_sha,
    )
    verifier_members = validate_verifier_bundle(
        verifier_bundle_path,
        expected_sha256=expected_verifier_bundle_sha256,
        expected_verifier_sha256=expected_verifier_sha256,
        expected_gate_builder_sha256=expected_gate_builder_sha256,
    )

    result = with_self_hash(
        {
            "schema_version": "exp689_teacher_smoke_promotion_gate_v1",
            "experiment_id": "689",
            "scope": "technical_smoke_to_full_teacher",
            "status": "accepted",
            "decision": "OPEN_FULL_TEACHER",
            "independent_integrator_required": True,
            "remote_acceptance_sha256": expected_remote_acceptance_sha256,
            "remote_acceptance_self_sha256": acceptance["self_sha256"],
            "smoke_commit_sha": expected_commit,
            "teacher_code_bundle_sha256": acceptance["code_bundle_sha256"],
            "teacher_code_bundle_members": teacher_members,
            "runner_sha256": acceptance["runner_sha256"],
            "prompt_sha256": acceptance["prompt_sha256"],
            "source_sha256": acceptance["source_sha256"],
            "image_manifest_sha256": acceptance["image_manifest_sha256"],
            "pixel_set_sha256": acceptance["pixel_set_sha256"],
            "model_contract_sha256": acceptance["model_contract_sha256"],
            "model_contract_self_sha256": acceptance["model_contract_self_sha256"],
            "model_registry_input_identity_sha256": acceptance[
                "model_registry_input_identity_sha256"
            ],
            "model_tree_sha256": acceptance["model_tree_sha256"],
            "processor_sha256": acceptance["processor_sha256"],
            "terminal_metadata_sha256": terminal_file_sha,
            "terminal_metadata_self_sha256": terminal["self_sha256"],
            "terminal_state": terminal["terminal_state"],
            "remote_receipt_sha256": receipt_file_sha,
            "remote_receipt_self_sha256": receipt["self_sha256"],
            "remote_output_ref_sha256": output_ref_sha,
            "runner_output_inventory": inventory,
            "runner_output_inventory_sha256": acceptance[
                "runner_output_inventory_sha256"
            ],
            "verifier_bundle_sha256": expected_verifier_bundle_sha256,
            "verifier_bundle_members": verifier_members,
            "verifier_sha256": expected_verifier_sha256,
            "gate_builder_sha256": expected_gate_builder_sha256,
            "labels_read": 0,
            "sealed_rows": 0,
            "public_used": False,
            "quality_evaluated": False,
            "full_teacher_authorized": True,
            "student_gpu_authorized": False,
            "self_sha256": None,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-acceptance", type=Path, required=True)
    parser.add_argument("--expected-remote-acceptance-sha256", required=True)
    parser.add_argument("--remote-receipt", type=Path, required=True)
    parser.add_argument("--terminal-metadata", type=Path, required=True)
    parser.add_argument("--teacher-code-bundle", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--verifier-bundle", type=Path, required=True)
    parser.add_argument("--expected-verifier-bundle-sha256", required=True)
    parser.add_argument("--expected-verifier-sha256", required=True)
    parser.add_argument("--expected-gate-builder-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        remote_acceptance_path=args.remote_acceptance,
        expected_remote_acceptance_sha256=args.expected_remote_acceptance_sha256,
        remote_receipt_path=args.remote_receipt,
        terminal_metadata_path=args.terminal_metadata,
        teacher_code_bundle_path=args.teacher_code_bundle,
        model_contract_path=args.model_contract,
        verifier_bundle_path=args.verifier_bundle,
        expected_verifier_bundle_sha256=args.expected_verifier_bundle_sha256,
        expected_verifier_sha256=args.expected_verifier_sha256,
        expected_gate_builder_sha256=args.expected_gate_builder_sha256,
        expected_commit=args.expected_commit,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Re-verify actual remote smoke objects and issue an independent exp689 gate."""

from __future__ import annotations

import argparse
import json
import re
import stat
import zipfile
from pathlib import Path
from typing import Any

import run_teacher
import verify_teacher_run
from build_target_audit import (
    ContractError,
    expect_exact_keys,
    load_json,
    require_hex64,
    sha256_bytes,
    sha256_file,
    validate_self_hash,
    with_self_hash,
    write_json,
)

VERIFIER_BUNDLE_MEMBERS = {
    "build_target_audit.py",
    "build_teacher_model_contract.py",
    "build_teacher_smoke_promotion_gate.py",
    "run_teacher.py",
    "verify_teacher_run.py",
}
RESOLVED_METADATA_FIELDS = {
    "schema_version",
    "metadata_source",
    "job_identity_sha256",
    "terminal_state",
    "exit_code",
    "failure_reason",
    "submitted_at_utc",
    "started_at_utc",
    "finished_at_utc",
    "attempt",
    "gpu_count",
    "gpu_model",
    "resolved_preset_sha256",
    "resolved_command_sha256",
    "resolved_inputs",
    "resolved_output_ref",
    "self_sha256",
}
RESOLVED_INPUT_FIELDS = {
    "commit_sha",
    "code_bundle_sha256",
    "code_bundle_members",
    "runner_sha256",
    "teacher_request_sha256",
    "image_manifest_sha256",
    "pixel_set_sha256",
    "model_registry_input_identity_sha256",
    "model_contract_sha256",
    "model_contract_self_sha256",
    "model_tree_sha256",
    "processor_sha256",
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
    executing = {
        "verify_teacher_run.py": sha256_file(Path(verify_teacher_run.__file__)),
        "run_teacher.py": sha256_file(Path(run_teacher.__file__)),
        "build_teacher_smoke_promotion_gate.py": sha256_file(Path(__file__)),
    }
    if any(members[name] != value for name, value in executing.items()):
        raise ContractError("executing verifier/gate/runner differs from verifier bundle")
    if members["verify_teacher_run.py"] != expected_verifier_sha256:
        raise ContractError("verifier bundle verifier-code SHA mismatch")
    if members["build_teacher_smoke_promotion_gate.py"] != expected_gate_builder_sha256:
        raise ContractError("verifier bundle gate-builder SHA mismatch")
    return dict(sorted(members.items()))


def validate_resolved_job_metadata(
    metadata: dict[str, Any],
    *,
    receipt: dict[str, Any],
    remote_acceptance: dict[str, Any],
    expected_preset_sha256: str,
    expected_command_sha256: str,
) -> None:
    expect_exact_keys(metadata, RESOLVED_METADATA_FIELDS, "resolved ML Core metadata")
    validate_self_hash(metadata, "resolved ML Core metadata")
    if metadata["schema_version"] != "exp689_teacher_resolved_job_metadata_v1":
        raise ContractError("resolved ML Core metadata schema mismatch")
    if metadata["metadata_source"] != "mlcore_resolved_job_api_independent":
        raise ContractError("resolved ML Core metadata is not independently sourced")
    if (
        metadata["terminal_state"] != "SUCCESS"
        or metadata["exit_code"] != 0
        or metadata["failure_reason"] is not None
        or metadata["attempt"] != 1
        or metadata["gpu_count"] != 1
        or "H100" not in str(metadata["gpu_model"]).upper()
    ):
        raise ContractError("resolved ML Core metadata is not exact one-H100 SUCCESS")
    for field in ("submitted_at_utc", "started_at_utc", "finished_at_utc"):
        if not isinstance(metadata[field], str) or not TIMESTAMP.fullmatch(metadata[field]):
            raise ContractError("resolved ML Core metadata timestamp mismatch")
        if metadata[field] != receipt[field]:
            raise ContractError("resolved ML Core metadata/receipt timestamp mismatch")
    if metadata["job_identity_sha256"] != receipt["job_identity_sha256"]:
        raise ContractError("resolved ML Core metadata job identity mismatch")
    for value, context in (
        (expected_preset_sha256, "expected resolved preset SHA"),
        (expected_command_sha256, "expected resolved command SHA"),
    ):
        require_hex64(value, context)
    if (
        metadata["resolved_preset_sha256"] != expected_preset_sha256
        or metadata["resolved_command_sha256"] != expected_command_sha256
    ):
        raise ContractError("resolved ML Core preset/command identity mismatch")
    inputs = metadata["resolved_inputs"]
    if not isinstance(inputs, dict):
        raise ContractError("resolved ML Core inputs must be an object")
    expect_exact_keys(inputs, RESOLVED_INPUT_FIELDS, "resolved ML Core inputs")
    expected_inputs = {
        "commit_sha": remote_acceptance["commit_sha"],
        "code_bundle_sha256": remote_acceptance["code_bundle_sha256"],
        "code_bundle_members": remote_acceptance["code_bundle_members"],
        "runner_sha256": remote_acceptance["runner_sha256"],
        "teacher_request_sha256": remote_acceptance["source_sha256"],
        "image_manifest_sha256": remote_acceptance["image_manifest_sha256"],
        "pixel_set_sha256": remote_acceptance["pixel_set_sha256"],
        "model_registry_input_identity_sha256": remote_acceptance[
            "model_registry_input_identity_sha256"
        ],
        "model_contract_sha256": remote_acceptance["model_contract_sha256"],
        "model_contract_self_sha256": remote_acceptance["model_contract_self_sha256"],
        "model_tree_sha256": remote_acceptance["model_tree_sha256"],
        "processor_sha256": remote_acceptance["processor_sha256"],
    }
    for field, expected in expected_inputs.items():
        if inputs[field] != expected:
            raise ContractError(f"resolved ML Core input {field} mismatch")
    if metadata["resolved_output_ref"] != receipt["output_ref"]:
        raise ContractError("resolved ML Core output destination/object binding mismatch")


def build(
    *,
    remote_root: Path,
    smoke_output_dir: Path,
    remote_receipt_path: Path,
    resolved_job_metadata_path: Path,
    teacher_request_path: Path,
    image_manifest_path: Path,
    model_contract_path: Path,
    teacher_code_bundle_path: Path,
    verifier_bundle_path: Path,
    expected_commit: str,
    expected_teacher_bundle_sha256: str,
    expected_runner_sha256: str,
    expected_prompt_sha256: str,
    expected_source_sha256: str,
    expected_image_manifest_sha256: str,
    expected_pixel_set_sha256: str,
    expected_model_contract_sha256: str,
    expected_model_input_identity_sha256: str,
    expected_verifier_bundle_sha256: str,
    expected_verifier_sha256: str,
    expected_gate_builder_sha256: str,
    expected_resolved_job_metadata_sha256: str,
    expected_resolved_preset_sha256: str,
    expected_resolved_command_sha256: str,
    approved_output_prefix: str,
    remote_acceptance_output_path: Path,
    promotion_gate_output_path: Path,
) -> dict[str, Any]:
    if promotion_gate_output_path.exists():
        raise FileExistsError("refusing to overwrite immutable teacher smoke gate")
    if remote_acceptance_output_path.exists():
        raise FileExistsError("refusing to overwrite remote smoke acceptance")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ContractError("expected smoke commit must be exact lowercase Git SHA")
    verifier_members = validate_verifier_bundle(
        verifier_bundle_path,
        expected_sha256=expected_verifier_bundle_sha256,
        expected_verifier_sha256=expected_verifier_sha256,
        expected_gate_builder_sha256=expected_gate_builder_sha256,
    )
    remote_acceptance = verify_teacher_run.verify(
        remote_root=remote_root,
        scope="technical_smoke",
        output_dir=smoke_output_dir,
        receipt_path=remote_receipt_path,
        teacher_request_path=teacher_request_path,
        image_manifest_path=image_manifest_path,
        model_contract_path=model_contract_path,
        code_bundle_path=teacher_code_bundle_path,
        expected_commit=expected_commit,
        expected_bundle_sha256=expected_teacher_bundle_sha256,
        expected_runner_sha256=expected_runner_sha256,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_source_sha256=expected_source_sha256,
        expected_image_manifest_sha256=expected_image_manifest_sha256,
        expected_pixel_set_sha256=expected_pixel_set_sha256,
        expected_accepted_smoke_self_sha256=None,
        expected_accepted_smoke_promotion_gate_self_sha256=None,
        expected_model_contract_sha256=expected_model_contract_sha256,
        expected_model_input_identity_sha256=expected_model_input_identity_sha256,
        approved_output_prefix=approved_output_prefix,
        acceptance_output_path=remote_acceptance_output_path,
    )
    receipt = load_json(remote_receipt_path, "teacher remote receipt replay")
    require_hex64(
        expected_resolved_job_metadata_sha256,
        "expected resolved-job-metadata SHA",
    )
    if sha256_file(resolved_job_metadata_path) != expected_resolved_job_metadata_sha256:
        raise ContractError("resolved ML Core job metadata file SHA mismatch")
    metadata = load_json(resolved_job_metadata_path, "resolved ML Core job metadata")
    validate_resolved_job_metadata(
        metadata,
        receipt=receipt,
        remote_acceptance=remote_acceptance,
        expected_preset_sha256=expected_resolved_preset_sha256,
        expected_command_sha256=expected_resolved_command_sha256,
    )
    gate = with_self_hash(
        {
            "schema_version": "exp689_teacher_smoke_promotion_gate_v2",
            "experiment_id": "689",
            "scope": "technical_smoke_to_full_teacher",
            "status": "accepted",
            "decision": "OPEN_FULL_TEACHER",
            "independent_remote_reverification": True,
            "remote_acceptance_sha256": sha256_file(remote_acceptance_output_path),
            "remote_acceptance_self_sha256": remote_acceptance["self_sha256"],
            "smoke_commit_sha": expected_commit,
            "teacher_code_bundle_sha256": expected_teacher_bundle_sha256,
            "teacher_code_bundle_members": remote_acceptance["code_bundle_members"],
            "runner_sha256": expected_runner_sha256,
            "prompt_sha256": expected_prompt_sha256,
            "source_sha256": expected_source_sha256,
            "image_manifest_sha256": expected_image_manifest_sha256,
            "pixel_set_sha256": expected_pixel_set_sha256,
            "model_contract_sha256": expected_model_contract_sha256,
            "model_contract_self_sha256": remote_acceptance[
                "model_contract_self_sha256"
            ],
            "model_registry_input_identity_sha256": (
                expected_model_input_identity_sha256
            ),
            "model_tree_sha256": remote_acceptance["model_tree_sha256"],
            "processor_sha256": remote_acceptance["processor_sha256"],
            "resolved_job_metadata_sha256": expected_resolved_job_metadata_sha256,
            "resolved_job_metadata_self_sha256": metadata["self_sha256"],
            "resolved_preset_sha256": expected_resolved_preset_sha256,
            "resolved_command_sha256": expected_resolved_command_sha256,
            "terminal_state": metadata["terminal_state"],
            "remote_receipt_sha256": remote_acceptance["remote_receipt_sha256"],
            "remote_receipt_self_sha256": remote_acceptance[
                "remote_receipt_self_sha256"
            ],
            "remote_output_ref_sha256": remote_acceptance["remote_output_ref_sha256"],
            "runner_output_inventory": remote_acceptance["runner_output_inventory"],
            "runner_output_inventory_sha256": remote_acceptance[
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
    promotion_gate_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(promotion_gate_output_path, gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--smoke-output-dir", type=Path, required=True)
    parser.add_argument("--remote-receipt", type=Path, required=True)
    parser.add_argument("--resolved-job-metadata", type=Path, required=True)
    parser.add_argument("--teacher-request", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--teacher-code-bundle", type=Path, required=True)
    parser.add_argument("--verifier-bundle", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-teacher-bundle-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-image-manifest-sha256", required=True)
    parser.add_argument("--expected-pixel-set-sha256", required=True)
    parser.add_argument("--expected-model-contract-sha256", required=True)
    parser.add_argument("--expected-model-input-identity-sha256", required=True)
    parser.add_argument("--expected-verifier-bundle-sha256", required=True)
    parser.add_argument("--expected-verifier-sha256", required=True)
    parser.add_argument("--expected-gate-builder-sha256", required=True)
    parser.add_argument("--expected-resolved-job-metadata-sha256", required=True)
    parser.add_argument("--expected-resolved-preset-sha256", required=True)
    parser.add_argument("--expected-resolved-command-sha256", required=True)
    parser.add_argument("--approved-output-prefix", required=True)
    parser.add_argument("--remote-acceptance-output", type=Path, required=True)
    parser.add_argument("--promotion-gate-output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        remote_root=args.remote_root,
        smoke_output_dir=args.smoke_output_dir,
        remote_receipt_path=args.remote_receipt,
        resolved_job_metadata_path=args.resolved_job_metadata,
        teacher_request_path=args.teacher_request,
        image_manifest_path=args.image_manifest,
        model_contract_path=args.model_contract,
        teacher_code_bundle_path=args.teacher_code_bundle,
        verifier_bundle_path=args.verifier_bundle,
        expected_commit=args.expected_commit,
        expected_teacher_bundle_sha256=args.expected_teacher_bundle_sha256,
        expected_runner_sha256=args.expected_runner_sha256,
        expected_prompt_sha256=args.expected_prompt_sha256,
        expected_source_sha256=args.expected_source_sha256,
        expected_image_manifest_sha256=args.expected_image_manifest_sha256,
        expected_pixel_set_sha256=args.expected_pixel_set_sha256,
        expected_model_contract_sha256=args.expected_model_contract_sha256,
        expected_model_input_identity_sha256=args.expected_model_input_identity_sha256,
        expected_verifier_bundle_sha256=args.expected_verifier_bundle_sha256,
        expected_verifier_sha256=args.expected_verifier_sha256,
        expected_gate_builder_sha256=args.expected_gate_builder_sha256,
        expected_resolved_job_metadata_sha256=(
            args.expected_resolved_job_metadata_sha256
        ),
        expected_resolved_preset_sha256=args.expected_resolved_preset_sha256,
        expected_resolved_command_sha256=args.expected_resolved_command_sha256,
        approved_output_prefix=args.approved_output_prefix,
        remote_acceptance_output_path=args.remote_acceptance_output,
        promotion_gate_output_path=args.promotion_gate_output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

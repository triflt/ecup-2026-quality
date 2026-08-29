"""Fail-closed remote acceptance verifier for exp689 teacher smoke/full outputs."""

from __future__ import annotations

import argparse
import json
import math
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
    load_spec,
    read_jsonl,
    require_hex64,
    require_remote_path,
    sha256_bytes,
    sha256_file,
    validate_self_hash,
    validate_teacher_request,
    validate_teacher_selection,
    with_self_hash,
    write_json,
)
from build_teacher_model_contract import MODEL_ID, MODEL_REVISION

OUTPUT_FILES = ("acceptance.json", "report.json", "targets.jsonl")
BUNDLE_MEMBERS = (
    "build_target_audit.py",
    "frozen_target_audit_spec.json",
    "run_teacher.py",
    "target_audit_schema_v1.json",
)
TECHNICAL_CHECKS = {
    "exact_one_h100",
    "base_model_only",
    "no_cpu_or_disk_offload",
    "cuda_forward_all_rows",
    "pixel_tensor_all_rows",
    "full_and_2x2_visible_all_rows",
    "closed_schema_all_rows",
    "request_binding_all_rows",
    "evidence_binding_all_rows",
    "artifact_write_complete",
}
REPORT_FIELDS = {
    "schema_version",
    "experiment_id",
    "scope",
    "model_id",
    "model_revision",
    "model_tree_sha256",
    "processor_sha256",
    "code_sha256",
    "prompt_sha256",
    "source_sha256",
    "image_manifest_sha256",
    "pixel_set_sha256",
    "input_contract_self_sha256",
    "accepted_smoke_self_sha256",
    "accepted_smoke_promotion_gate_self_sha256",
    "selection_payload_sha256",
    "targets_sha256",
    "rows",
    "decoding",
    "runtime_seconds",
    "peak_cuda_bytes",
    "total_cuda_bytes",
    "cuda_device_name",
    "model_class",
    "packages",
    "packages_sha256",
    "technical_checks",
    "labels_read",
    "sealed_rows",
    "public_used",
    "quality_evaluated",
    "student_gpu_authorized",
    "self_sha256",
}
RUNNER_ACCEPTANCE_FIELDS = {
    "schema_version",
    "experiment_id",
    "scope",
    "status",
    "decision",
    "rows",
    "model_id",
    "model_revision",
    "model_tree_sha256",
    "processor_sha256",
    "code_sha256",
    "prompt_sha256",
    "source_sha256",
    "pixel_set_sha256",
    "selection_payload_sha256",
    "targets_sha256",
    "report_sha256",
    "runtime_seconds",
    "peak_cuda_bytes",
    "technical_checks",
    "labels_read",
    "sealed_rows",
    "public_used",
    "quality_evaluated",
    "full_teacher_authorized",
    "student_gpu_authorized",
    "terminal_job_metadata_bound",
    "approved_remote_output_bound",
    "self_sha256",
}
RECEIPT_FIELDS = {
    "schema_version",
    "experiment_id",
    "scope",
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
    "commit_sha",
    "code_bundle_sha256",
    "code_bundle_members",
    "runner_sha256",
    "teacher_request_sha256",
    "image_manifest_sha256",
    "pixel_set_sha256",
    "model_registry_input",
    "output_ref",
    "runner_runtime_seconds",
    "peak_cuda_bytes",
    "labels_read",
    "sealed_rows",
    "public_used",
    "self_sha256",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def validate_code_bundle(
    path: Path, *, expected_sha256: str, expected_runner_sha256: str
) -> dict[str, str]:
    require_hex64(expected_sha256, "expected code-bundle SHA")
    require_hex64(expected_runner_sha256, "expected runner SHA")
    if sha256_file(path) != expected_sha256:
        raise ContractError("code bundle SHA mismatch")
    try:
        with zipfile.ZipFile(path) as bundle:
            infos = bundle.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ContractError("code bundle contains duplicate members")
            if tuple(sorted(names)) != BUNDLE_MEMBERS:
                raise ContractError("code bundle member whitelist mismatch")
            members: dict[str, str] = {}
            for info in infos:
                member_path = Path(info.filename)
                mode = info.external_attr >> 16
                if (
                    info.is_dir()
                    or member_path.is_absolute()
                    or ".." in member_path.parts
                    or stat.S_ISLNK(mode)
                ):
                    raise ContractError("code bundle contains an unsafe member")
                members[info.filename] = sha256_bytes(bundle.read(info))
            if bundle.testzip() is not None:
                raise ContractError("code bundle integrity check failed")
    except (OSError, zipfile.BadZipFile) as error:
        raise ContractError("code bundle is unreadable or corrupt") from error
    if members["run_teacher.py"] != expected_runner_sha256:
        raise ContractError("code bundle runner SHA mismatch")
    return dict(sorted(members.items()))


def validate_model_contract(contract: dict[str, Any], expected_file_sha256: str) -> None:
    require_hex64(expected_file_sha256, "expected model-contract SHA")
    expect_exact_keys(
        contract,
        {
            "schema_version",
            "model_id",
            "model_revision",
            "model_tree_sha256",
            "model_tree_files",
            "processor_sha256",
            "processor_files",
            "base_only",
            "class_lora_present",
            "self_sha256",
        },
        "teacher model contract",
    )
    validate_self_hash(contract, "teacher model contract")
    if contract["schema_version"] != "exp689_teacher_model_contract_v1":
        raise ContractError("teacher model contract schema mismatch")
    if contract["model_id"] != MODEL_ID or contract["model_revision"] != MODEL_REVISION:
        raise ContractError("teacher model contract identity/revision mismatch")
    if contract["base_only"] is not True or contract["class_lora_present"] is not False:
        raise ContractError("teacher model contract is not exact base-only")
    for field in ("model_tree_sha256", "processor_sha256"):
        require_hex64(contract[field], f"teacher model contract.{field}")
    for field in ("model_tree_files", "processor_files"):
        if not isinstance(contract[field], int) or contract[field] <= 0:
            raise ContractError(f"teacher model contract.{field} must be positive")


def validate_output_inventory(output_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    if not output_dir.is_dir():
        raise ContractError("teacher runner output is not a directory")
    paths = sorted(output_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ContractError("teacher runner output may contain only regular files")
    if tuple(path.name for path in paths) != OUTPUT_FILES:
        raise ContractError("teacher runner output inventory mismatch")
    hashes = {path.name: sha256_file(path) for path in paths}
    sizes = {path.name: path.stat().st_size for path in paths}
    return hashes, sizes


def validate_requests_and_targets(
    request_path: Path, targets_path: Path, *, scope: str
) -> tuple[list[dict[str, Any]], str]:
    requests = read_jsonl(request_path, "teacher request")
    targets = read_jsonl(targets_path, "teacher targets")
    expected_rows = 12 if scope == "technical_smoke" else 300
    if len(requests) != expected_rows or len(targets) != expected_rows:
        raise ContractError(f"{scope}: request and target rows must both equal {expected_rows}")
    spec = load_spec()
    for index, (request, target) in enumerate(zip(requests, targets, strict=True), 1):
        validate_teacher_request(request, index, spec)
        validate_teacher_selection(target, request, spec, index)
    return targets, sha256_bytes(canonical_json_bytes(targets))


def validate_pixel_manifest(
    image_manifest_path: Path, request_path: Path, *, expected_pixel_set_sha256: str
) -> None:
    require_hex64(expected_pixel_set_sha256, "expected pixel-set SHA")
    requests = read_jsonl(request_path, "teacher request")
    manifest = read_jsonl(image_manifest_path, "teacher image manifest")
    if len(manifest) != len(requests):
        raise ContractError("teacher image manifest row count mismatch")
    pixels: list[str] = []
    for index, (row, request) in enumerate(zip(manifest, requests, strict=True), 1):
        if row.get("audit_id") != request["audit_id"]:
            raise ContractError(f"teacher image manifest row {index}: audit binding mismatch")
        for field in (
            "reference",
            "content_sha256",
            "decoded_rgb_sha256",
            "media_type",
            "width",
            "height",
        ):
            if row.get(field) != request["first_image"][field]:
                raise ContractError(
                    f"teacher image manifest row {index}: {field} binding mismatch"
                )
        pixel_sha = require_hex64(row.get("pixel_sha256"), f"image manifest row {index}.pixel")
        if pixel_sha != row["decoded_rgb_sha256"]:
            raise ContractError(f"teacher image manifest row {index}: pixel SHA mismatch")
        pixels.append(pixel_sha)
    if sha256_bytes(canonical_json_bytes(pixels)) != expected_pixel_set_sha256:
        raise ContractError("teacher image manifest pixel-set SHA mismatch")


def _validate_technical_checks(value: Any, context: str) -> None:
    if not isinstance(value, dict) or set(value) != TECHNICAL_CHECKS:
        raise ContractError(f"{context}: exact technical checks required")
    if any(result is not True for result in value.values()):
        raise ContractError(f"{context}: every technical check must pass")


def validate_runner_artifacts(
    *,
    scope: str,
    output_dir: Path,
    request_path: Path,
    image_manifest_path: Path,
    expected_runner_sha256: str,
    expected_prompt_sha256: str,
    expected_source_sha256: str,
    expected_image_manifest_sha256: str,
    expected_pixel_set_sha256: str,
    expected_accepted_smoke_self_sha256: str | None,
    expected_accepted_smoke_promotion_gate_self_sha256: str | None,
    model_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, int]]:
    inventory, sizes = validate_output_inventory(output_dir)
    if sha256_file(request_path) != expected_source_sha256:
        raise ContractError("teacher request/source SHA pin mismatch")
    if sha256_file(image_manifest_path) != expected_image_manifest_sha256:
        raise ContractError("teacher image-manifest SHA pin mismatch")
    validate_pixel_manifest(
        image_manifest_path, request_path, expected_pixel_set_sha256=expected_pixel_set_sha256
    )
    targets, selection_sha = validate_requests_and_targets(
        request_path, output_dir / "targets.jsonl", scope=scope
    )
    report = load_json(output_dir / "report.json", "teacher runner report")
    runner_acceptance = load_json(
        output_dir / "acceptance.json", "teacher runner acceptance"
    )
    expect_exact_keys(report, REPORT_FIELDS, "teacher runner report")
    expect_exact_keys(
        runner_acceptance, RUNNER_ACCEPTANCE_FIELDS, "teacher runner acceptance"
    )
    validate_self_hash(report, "teacher runner report")
    validate_self_hash(runner_acceptance, "teacher runner acceptance")
    expected_rows = 12 if scope == "technical_smoke" else 300
    expected_decision = (
        "TECHNICAL_SMOKE_PASS_NOT_QUALITY"
        if scope == "technical_smoke"
        else "TARGETS_READY_FOR_DUAL_BLIND_REVIEW"
    )
    common = {
        "experiment_id": "689",
        "scope": scope,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_tree_sha256": model_contract["model_tree_sha256"],
        "processor_sha256": model_contract["processor_sha256"],
        "code_sha256": expected_runner_sha256,
        "prompt_sha256": expected_prompt_sha256,
        "source_sha256": expected_source_sha256,
        "pixel_set_sha256": expected_pixel_set_sha256,
        "selection_payload_sha256": selection_sha,
        "targets_sha256": inventory["targets.jsonl"],
        "rows": expected_rows,
        "labels_read": 0,
        "sealed_rows": 0,
        "public_used": False,
        "quality_evaluated": False,
        "student_gpu_authorized": False,
    }
    for document_name, document in (("report", report), ("acceptance", runner_acceptance)):
        for field, expected in common.items():
            if document[field] != expected:
                raise ContractError(f"teacher runner {document_name}: {field} mismatch")
        _validate_technical_checks(
            document["technical_checks"], f"teacher runner {document_name}"
        )
        if (
            not isinstance(document["runtime_seconds"], (int, float))
            or not math.isfinite(document["runtime_seconds"])
            or document["runtime_seconds"] <= 0
            or not isinstance(document["peak_cuda_bytes"], int)
            or document["peak_cuda_bytes"] <= 0
        ):
            raise ContractError(f"teacher runner {document_name}: runtime/peak invalid")
    if scope == "technical_smoke":
        if (
            expected_accepted_smoke_self_sha256 is not None
            or expected_accepted_smoke_promotion_gate_self_sha256 is not None
        ):
            raise ContractError("technical smoke may not consume an accepted smoke gate")
        if (
            report["accepted_smoke_self_sha256"] is not None
            or report["accepted_smoke_promotion_gate_self_sha256"] is not None
        ):
            raise ContractError("technical smoke report pre-claims an accepted smoke gate")
    else:
        require_hex64(
            expected_accepted_smoke_self_sha256,
            "expected accepted-smoke self SHA",
        )
        if report["accepted_smoke_self_sha256"] != expected_accepted_smoke_self_sha256:
            raise ContractError("full teacher report accepted-smoke binding mismatch")
        require_hex64(
            expected_accepted_smoke_promotion_gate_self_sha256,
            "expected accepted-smoke promotion-gate self SHA",
        )
        if (
            report["accepted_smoke_promotion_gate_self_sha256"]
            != expected_accepted_smoke_promotion_gate_self_sha256
        ):
            raise ContractError("full teacher report promotion-gate binding mismatch")
    if report["schema_version"] != "exp689_teacher_run_report_v1":
        raise ContractError("teacher runner report schema mismatch")
    if report["image_manifest_sha256"] != expected_image_manifest_sha256:
        raise ContractError("teacher runner report image-manifest SHA mismatch")
    if report["decoding"] != {
        "do_sample": False,
        "enable_thinking": False,
        "max_new_tokens": 160,
        "repair_attempts": 0,
    }:
        raise ContractError("teacher runner report decoding contract mismatch")
    if report["packages_sha256"] != sha256_bytes(canonical_json_bytes(report["packages"])):
        raise ContractError("teacher runner report package inventory SHA mismatch")
    if report["total_cuda_bytes"] <= 0 or "H100" not in report["cuda_device_name"].upper():
        raise ContractError("teacher runner report did not bind an H100 runtime")
    if runner_acceptance["schema_version"] != "exp689_teacher_run_acceptance_v1":
        raise ContractError("teacher runner acceptance schema mismatch")
    if runner_acceptance["status"] != "accepted" or runner_acceptance["decision"] != expected_decision:
        raise ContractError("teacher runner acceptance status/decision mismatch")
    if runner_acceptance["report_sha256"] != inventory["report.json"]:
        raise ContractError("teacher runner acceptance report SHA mismatch")
    if (
        runner_acceptance["runtime_seconds"] != report["runtime_seconds"]
        or runner_acceptance["peak_cuda_bytes"] != report["peak_cuda_bytes"]
    ):
        raise ContractError("teacher runner report/acceptance runtime or peak mismatch")
    if runner_acceptance["full_teacher_authorized"] is not False:
        raise ContractError("runner-local acceptance may not authorize the full teacher")
    if (
        runner_acceptance["terminal_job_metadata_bound"] is not False
        or runner_acceptance["approved_remote_output_bound"] is not False
    ):
        raise ContractError("runner-local acceptance cannot pre-claim remote provenance")
    if targets != read_jsonl(output_dir / "targets.jsonl", "teacher targets replay"):
        raise ContractError("teacher targets changed during verification")
    return report, runner_acceptance, inventory, sizes


def validate_receipt(
    receipt: dict[str, Any],
    *,
    scope: str,
    expected_commit: str,
    expected_bundle_sha256: str,
    bundle_members: dict[str, str],
    expected_runner_sha256: str,
    expected_source_sha256: str,
    expected_image_manifest_sha256: str,
    expected_pixel_set_sha256: str,
    expected_model_input_identity_sha256: str,
    model_contract: dict[str, Any],
    model_contract_file_sha256: str,
    inventory: dict[str, str],
    sizes: dict[str, int],
    report: dict[str, Any],
    approved_output_prefix: str,
) -> None:
    expect_exact_keys(receipt, RECEIPT_FIELDS, "teacher remote receipt")
    validate_self_hash(receipt, "teacher remote receipt")
    if receipt["schema_version"] != "exp689_teacher_remote_receipt_v1":
        raise ContractError("teacher remote receipt schema mismatch")
    if receipt["experiment_id"] != "689" or receipt["scope"] != scope:
        raise ContractError("teacher remote receipt experiment/scope mismatch")
    require_hex64(receipt["job_identity_sha256"], "teacher remote job identity")
    if (
        receipt["terminal_state"] != "SUCCESS"
        or receipt["exit_code"] != 0
        or receipt["failure_reason"] is not None
        or receipt["attempt"] != 1
    ):
        raise ContractError("teacher remote receipt is not an exact terminal SUCCESS")
    timestamps = [
        receipt["submitted_at_utc"],
        receipt["started_at_utc"],
        receipt["finished_at_utc"],
    ]
    if any(not isinstance(value, str) or not TIMESTAMP.fullmatch(value) for value in timestamps):
        raise ContractError("teacher remote receipt timestamps are invalid")
    if timestamps != sorted(timestamps):
        raise ContractError("teacher remote receipt timestamps are out of order")
    if receipt["gpu_count"] != 1 or "H100" not in receipt["gpu_model"].upper():
        raise ContractError("teacher remote receipt must bind exactly one H100")
    if not HEX40.fullmatch(expected_commit) or receipt["commit_sha"] != expected_commit:
        raise ContractError("teacher remote receipt commit SHA mismatch")
    exact_hashes = {
        "code_bundle_sha256": expected_bundle_sha256,
        "runner_sha256": expected_runner_sha256,
        "teacher_request_sha256": expected_source_sha256,
        "image_manifest_sha256": expected_image_manifest_sha256,
        "pixel_set_sha256": expected_pixel_set_sha256,
    }
    for field, expected in exact_hashes.items():
        if receipt[field] != expected:
            raise ContractError(f"teacher remote receipt {field} mismatch")
    if receipt["code_bundle_members"] != bundle_members:
        raise ContractError("teacher remote receipt code-bundle member binding mismatch")
    registry = receipt["model_registry_input"]
    if not isinstance(registry, dict):
        raise ContractError("teacher remote receipt model-registry input must be an object")
    expect_exact_keys(
        registry,
        {
            "input_identity_sha256",
            "model_id",
            "model_revision",
            "model_contract_sha256",
            "model_contract_self_sha256",
            "model_tree_sha256",
            "processor_sha256",
        },
        "teacher remote model-registry input",
    )
    registry_expected = {
        "input_identity_sha256": expected_model_input_identity_sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_contract_sha256": model_contract_file_sha256,
        "model_contract_self_sha256": model_contract["self_sha256"],
        "model_tree_sha256": model_contract["model_tree_sha256"],
        "processor_sha256": model_contract["processor_sha256"],
    }
    for field, expected in registry_expected.items():
        if registry[field] != expected:
            raise ContractError(f"teacher remote model-registry input {field} mismatch")
    require_hex64(registry["input_identity_sha256"], "model-registry input identity")
    if (
        receipt["runner_runtime_seconds"] != report["runtime_seconds"]
        or receipt["peak_cuda_bytes"] != report["peak_cuda_bytes"]
    ):
        raise ContractError("teacher remote receipt runtime/peak mismatch")
    if (
        receipt["labels_read"] != 0
        or receipt["sealed_rows"] != 0
        or receipt["public_used"] is not False
    ):
        raise ContractError("teacher remote receipt used labels, sealed rows, or Public")
    output_ref = receipt["output_ref"]
    if not isinstance(output_ref, dict):
        raise ContractError("teacher remote output ref must be an object")
    expect_exact_keys(
        output_ref,
        {"schema_version", "s3_prefix", "immutable", "objects"},
        "teacher remote output ref",
    )
    if output_ref["schema_version"] != "exp689_immutable_s3_output_ref_v1":
        raise ContractError("teacher remote output ref schema mismatch")
    if not approved_output_prefix.startswith("s3://") or not approved_output_prefix.endswith("/"):
        raise ContractError("approved S3 output prefix must be an exact s3:// prefix ending in /")
    actual_prefix = output_ref["s3_prefix"]
    if (
        not isinstance(actual_prefix, str)
        or not actual_prefix.startswith(approved_output_prefix)
        or not actual_prefix.endswith("/")
        or "?" in actual_prefix
        or output_ref["immutable"] is not True
    ):
        raise ContractError("teacher remote output ref is not below the approved immutable prefix")
    objects = output_ref["objects"]
    if not isinstance(objects, list) or len(objects) != len(OUTPUT_FILES):
        raise ContractError("teacher remote output ref must bind exactly three objects")
    expected_names = sorted(OUTPUT_FILES)
    if [item.get("name") for item in objects if isinstance(item, dict)] != expected_names:
        raise ContractError("teacher remote output object inventory mismatch")
    for item in objects:
        expect_exact_keys(
            item, {"name", "uri", "version_id", "sha256", "size"}, "S3 output object"
        )
        name = item["name"]
        if item["uri"] != actual_prefix + name or "?" in item["uri"]:
            raise ContractError("teacher remote output object URI mismatch")
        if not isinstance(item["version_id"], str) or not item["version_id"].strip():
            raise ContractError("teacher remote output object lacks immutable version ID")
        if item["sha256"] != inventory[name] or item["size"] != sizes[name]:
            raise ContractError("teacher remote output object content binding mismatch")


def verify(
    *,
    remote_root: Path,
    scope: str,
    output_dir: Path,
    receipt_path: Path,
    teacher_request_path: Path,
    image_manifest_path: Path,
    model_contract_path: Path,
    code_bundle_path: Path,
    expected_commit: str,
    expected_bundle_sha256: str,
    expected_runner_sha256: str,
    expected_prompt_sha256: str,
    expected_source_sha256: str,
    expected_image_manifest_sha256: str,
    expected_pixel_set_sha256: str,
    expected_accepted_smoke_self_sha256: str | None,
    expected_accepted_smoke_promotion_gate_self_sha256: str | None,
    expected_model_contract_sha256: str,
    expected_model_input_identity_sha256: str,
    approved_output_prefix: str,
    acceptance_output_path: Path,
) -> dict[str, Any]:
    if scope not in {"technical_smoke", "full"}:
        raise ContractError("teacher remote scope must be technical_smoke or full")
    remote_root = remote_root.resolve(strict=True)
    required = {
        "teacher runner output": output_dir,
        "teacher remote receipt": receipt_path,
        "teacher request": teacher_request_path,
        "teacher image manifest": image_manifest_path,
        "teacher model contract": model_contract_path,
        "teacher code bundle": code_bundle_path,
    }
    resolved = {
        name: require_remote_path(remote_root, path, context=name, must_exist=True)
        for name, path in required.items()
    }
    acceptance_output_path = require_remote_path(
        remote_root,
        acceptance_output_path,
        context="teacher remote acceptance output",
        must_exist=False,
    )
    if acceptance_output_path.exists():
        raise FileExistsError("refusing to overwrite immutable teacher remote acceptance")
    for value, context in (
        (expected_prompt_sha256, "expected prompt SHA"),
        (expected_source_sha256, "expected source SHA"),
        (expected_image_manifest_sha256, "expected image-manifest SHA"),
        (expected_pixel_set_sha256, "expected pixel-set SHA"),
        (expected_model_input_identity_sha256, "expected model-input identity SHA"),
    ):
        require_hex64(value, context)
    bundle_members = validate_code_bundle(
        resolved["teacher code bundle"],
        expected_sha256=expected_bundle_sha256,
        expected_runner_sha256=expected_runner_sha256,
    )
    model_contract = load_json(resolved["teacher model contract"], "teacher model contract")
    if sha256_file(resolved["teacher model contract"]) != expected_model_contract_sha256:
        raise ContractError("teacher model-contract file SHA mismatch")
    validate_model_contract(model_contract, expected_model_contract_sha256)
    report, _, inventory, sizes = validate_runner_artifacts(
        scope=scope,
        output_dir=resolved["teacher runner output"],
        request_path=resolved["teacher request"],
        image_manifest_path=resolved["teacher image manifest"],
        expected_runner_sha256=expected_runner_sha256,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_source_sha256=expected_source_sha256,
        expected_image_manifest_sha256=expected_image_manifest_sha256,
        expected_pixel_set_sha256=expected_pixel_set_sha256,
        expected_accepted_smoke_self_sha256=expected_accepted_smoke_self_sha256,
        expected_accepted_smoke_promotion_gate_self_sha256=(
            expected_accepted_smoke_promotion_gate_self_sha256
        ),
        model_contract=model_contract,
    )
    receipt = load_json(resolved["teacher remote receipt"], "teacher remote receipt")
    validate_receipt(
        receipt,
        scope=scope,
        expected_commit=expected_commit,
        expected_bundle_sha256=expected_bundle_sha256,
        bundle_members=bundle_members,
        expected_runner_sha256=expected_runner_sha256,
        expected_source_sha256=expected_source_sha256,
        expected_image_manifest_sha256=expected_image_manifest_sha256,
        expected_pixel_set_sha256=expected_pixel_set_sha256,
        expected_model_input_identity_sha256=expected_model_input_identity_sha256,
        model_contract=model_contract,
        model_contract_file_sha256=expected_model_contract_sha256,
        inventory=inventory,
        sizes=sizes,
        report=report,
        approved_output_prefix=approved_output_prefix,
    )
    decision = (
        "OPEN_FULL_TEACHER"
        if scope == "technical_smoke"
        else "TARGETS_READY_FOR_DUAL_BLIND_REVIEW"
    )
    result = with_self_hash(
        {
            "schema_version": "exp689_teacher_remote_acceptance_v1",
            "experiment_id": "689",
            "scope": scope,
            "status": "accepted",
            "decision": decision,
            "technical_only": scope == "technical_smoke",
            "quality_evaluated": False,
            "student_gpu_authorized": False,
            "terminal_job_metadata_bound": True,
            "approved_remote_output_bound": True,
            "full_teacher_technical_gate_open": scope == "technical_smoke",
            "commit_sha": expected_commit,
            "code_bundle_sha256": expected_bundle_sha256,
            "code_bundle_members": bundle_members,
            "runner_sha256": expected_runner_sha256,
            "prompt_sha256": expected_prompt_sha256,
            "source_sha256": expected_source_sha256,
            "image_manifest_sha256": expected_image_manifest_sha256,
            "pixel_set_sha256": expected_pixel_set_sha256,
            "accepted_smoke_self_sha256": expected_accepted_smoke_self_sha256,
            "accepted_smoke_promotion_gate_self_sha256": (
                expected_accepted_smoke_promotion_gate_self_sha256
            ),
            "model_contract_sha256": expected_model_contract_sha256,
            "model_contract_self_sha256": model_contract["self_sha256"],
            "model_registry_input_identity_sha256": expected_model_input_identity_sha256,
            "model_tree_sha256": model_contract["model_tree_sha256"],
            "processor_sha256": model_contract["processor_sha256"],
            "runner_output_inventory": inventory,
            "runner_output_inventory_sha256": sha256_bytes(
                canonical_json_bytes(inventory)
            ),
            "remote_receipt_self_sha256": receipt["self_sha256"],
            "remote_receipt_sha256": sha256_file(resolved["teacher remote receipt"]),
            "remote_output_ref_sha256": sha256_bytes(
                canonical_json_bytes(receipt["output_ref"])
            ),
            "runtime_seconds": report["runtime_seconds"],
            "peak_cuda_bytes": report["peak_cuda_bytes"],
            "labels_read": 0,
            "sealed_rows": 0,
            "public_used": False,
            "jobs_launched_by_verifier": 0,
            "uploads_by_verifier": 0,
            "presets_built_by_verifier": 0,
            "bundles_built_by_verifier": 0,
            "self_sha256": None,
        }
    )
    acceptance_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(acceptance_output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("technical_smoke", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--teacher-request", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--code-bundle", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-image-manifest-sha256", required=True)
    parser.add_argument("--expected-pixel-set-sha256", required=True)
    parser.add_argument("--expected-accepted-smoke-self-sha256")
    parser.add_argument("--expected-accepted-smoke-promotion-gate-self-sha256")
    parser.add_argument("--expected-model-contract-sha256", required=True)
    parser.add_argument("--expected-model-input-identity-sha256", required=True)
    parser.add_argument("--approved-output-prefix", required=True)
    parser.add_argument("--acceptance-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(
        remote_root=args.remote_root,
        scope=args.scope,
        output_dir=args.output_dir,
        receipt_path=args.receipt,
        teacher_request_path=args.teacher_request,
        image_manifest_path=args.image_manifest,
        model_contract_path=args.model_contract,
        code_bundle_path=args.code_bundle,
        expected_commit=args.expected_commit,
        expected_bundle_sha256=args.expected_bundle_sha256,
        expected_runner_sha256=args.expected_runner_sha256,
        expected_prompt_sha256=args.expected_prompt_sha256,
        expected_source_sha256=args.expected_source_sha256,
        expected_image_manifest_sha256=args.expected_image_manifest_sha256,
        expected_pixel_set_sha256=args.expected_pixel_set_sha256,
        expected_accepted_smoke_self_sha256=args.expected_accepted_smoke_self_sha256,
        expected_accepted_smoke_promotion_gate_self_sha256=(
            args.expected_accepted_smoke_promotion_gate_self_sha256
        ),
        expected_model_contract_sha256=args.expected_model_contract_sha256,
        expected_model_input_identity_sha256=args.expected_model_input_identity_sha256,
        approved_output_prefix=args.approved_output_prefix,
        acceptance_output_path=args.acceptance_output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

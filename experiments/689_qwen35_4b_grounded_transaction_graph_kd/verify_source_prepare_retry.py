"""Transport-aware post-terminal verifier for the exp689 PREPARE retry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

EXPERIMENT = "experiments/689_qwen35_4b_grounded_transaction_graph_kd"
RUNNER_PATH = f"{EXPERIMENT}/verify_source_prepare_retry.py"
RETRY_PATHS = {
    f"{EXPERIMENT}/extract_source_archive_transport.py",
    f"{EXPERIMENT}/prepare_source_universe.py",
    f"{EXPERIMENT}/source_prepare_spec_v1.json",
    f"{EXPERIMENT}/verify_source_prepare.py",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
S3_REF = re.compile(r"^s3://[^/?#]+/[^?#]+$")
TRANSPORT_REPORT_FIELDS = {
    "schema_version",
    "archive_id",
    "archive_sha256",
    "archive_size_bytes",
    "member_count",
    "type_counts",
    "apple_metadata_skipped_count",
    "apple_metadata_skipped_names",
    "extracted_regular_files",
    "extracted_directories",
    "symlinks_extracted",
    "hardlinks_extracted",
    "path_traversal_extracted",
    "self_sha256",
}


def _fields(value: str) -> set[str]:
    return set(value.split())


LIVE_GO_FIELDS = _fields(
    """schema_version experiment_id issuer_role decision preset_sha256
    preset_size_bytes gate_file_sha256 gate_self_sha256 contract_file_sha256
    contract_self_sha256 materialization_receipt_file_sha256
    materialization_receipt_self_sha256 materialization_args_file_sha256
    builder_commit builder_sha256 submit_wrapper_sha256 submit_wrapper_test_sha256
    s3_endpoint s3_cli_sha256 s3_requirements_sha256 project_file_sha256
    one_shot_nonce receipt_path_sha256 output_bucket output_prefix
    transport_report_prefix audit_preset_errors audit_preset_warnings
    audit_preset_result_sha256 server_dry_run_passed server_dry_run_receipt_sha256
    empty_prefix_max_age_seconds max_jobs retry_attempt teacher_authorized
    model_authorized review_authorized student_gpu_authorized public_used
    resolved_terminal_metadata_required self_sha256"""
)
SUBMIT_RECEIPT_FIELDS = _fields(
    """schema_version experiment_id state returncode job_id error_class
    live_go_file_sha256 live_go_self_sha256 actual_sent_clean_preset_sha256
    actual_sent_clean_preset_size_bytes gate_file_sha256 gate_self_sha256
    contract_file_sha256 contract_self_sha256 materialization_receipt_file_sha256
    materialization_receipt_self_sha256 materialization_args_file_sha256
    builder_commit builder_sha256 submit_wrapper_sha256 submit_wrapper_test_sha256
    s3_endpoint s3_cli_sha256 s3_requirements_sha256 project_file_sha256
    one_shot_nonce receipt_path_sha256 output_bucket output_prefix
    transport_report_prefix empty_prefix_proof empty_prefix_proof_self_sha256
    max_jobs retry_attempt teacher_authorized model_authorized review_authorized
    student_gpu_authorized public_used resolved_terminal_metadata_required
    audit_preset_result_sha256 server_dry_run_receipt_sha256
    secret_payload_persisted process_output_persisted self_sha256"""
)
MATERIALIZATION_RECEIPT_FIELDS = _fields(
    """schema_version experiment_id scope retry_attempt
    diagnostic_acceptance_file_sha256 diagnostic_acceptance_self_sha256
    transport_retry_gate_file_sha256 transport_retry_gate_self_sha256
    preset_contract_file_sha256 preset_semantic_contract_sha256
    final_preset_sha256 final_preset_size_bytes retry_preset_builder_sha256
    controlled_prepare_retry_authorized max_jobs teacher_authorized
    student_gpu_authorized public_used self_sha256"""
)
RESOLVED_TERMINAL_FIELDS = _fields(
    """schema_version metadata_source experiment_id retry_attempt platform_attempt job_id status
    exit_code finished_at region flavor image gpu_count preset_sha256
    preset_size_bytes resolved_command_sha256 submit_receipt live_go
    materialization_receipt retry_gate retry_contract retry_code resolved_inputs
    resolved_outputs self_sha256"""
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, context: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _self_hash(value: dict[str, Any], context: str) -> str:
    actual = value.get("self_sha256")
    if not isinstance(actual, str) or not HEX64.fullmatch(actual):
        raise ValueError(f"{context} requires an exact self SHA")
    copy = dict(value)
    copy["self_sha256"] = None
    if sha256_bytes(canonical_json_bytes(copy)) != actual:
        raise ValueError(f"{context} canonical self SHA mismatch")
    return actual


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_code_manifest(
    *,
    root: Path,
    manifest_path: Path,
    expected_file_sha256: str,
    expected_self_sha256: str,
    expected_revision: str,
    expected_schema: str,
    expected_paths: set[str],
) -> dict[str, Any]:
    if not HEX40.fullmatch(expected_revision):
        raise ValueError("code manifest revision must be an exact Git SHA")
    if not HEX64.fullmatch(expected_file_sha256) or not HEX64.fullmatch(
        expected_self_sha256
    ):
        raise ValueError("code manifest requires exact SHA-256 bindings")
    if sha256_file(manifest_path) != expected_file_sha256:
        raise ValueError("code manifest file SHA mismatch")
    manifest = _json(manifest_path, "code manifest")
    if set(manifest) != {"schema_version", "builder_revision", "files", "self_sha256"}:
        raise ValueError("code manifest exact schema mismatch")
    if _self_hash(manifest, "code manifest") != expected_self_sha256:
        raise ValueError("code manifest expected self SHA mismatch")
    files = manifest["files"]
    if (
        manifest["schema_version"] != expected_schema
        or manifest["builder_revision"] != expected_revision
        or not isinstance(files, list)
        or [item.get("path") for item in files] != sorted(expected_paths)
    ):
        raise ValueError("code manifest frozen identity mismatch")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_paths:
        raise ValueError("code bundle exact file inventory mismatch")
    for index, item in enumerate(files):
        if set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"code manifest file {index} schema mismatch")
        path = root / item["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item["size_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"code manifest file {index} content mismatch")
    return manifest


def _expected_ref(contract: dict[str, Any], key: str) -> str:
    value = f"s3://{contract['bucket']}{key}"
    if not S3_REF.fullmatch(value):
        raise ValueError("retry contract contains an unsafe S3 reference")
    return value


def _validate_transport_report(
    path: Path, archive_id: str, profile: dict[str, Any]
) -> dict[str, Any]:
    report = _json(path, f"{archive_id} transport report")
    if set(report) != TRANSPORT_REPORT_FIELDS:
        raise ValueError(f"{archive_id} transport report exact schema mismatch")
    self_sha = _self_hash(report, f"{archive_id} transport report")
    skipped = report["apple_metadata_skipped_names"]
    if (
        report["schema_version"]
        != "exp689_source_archive_transport_extraction_v1"
        or report["archive_id"] != archive_id
        or report["archive_sha256"] != profile["sha256"]
        or report["archive_size_bytes"] != profile["size_bytes"]
        or report["member_count"] != profile["member_count"]
        or report["type_counts"] != profile["type_counts"]
        or report["apple_metadata_skipped_count"]
        != profile["apple_metadata_count"]
        or not isinstance(skipped, list)
        or len(skipped) != profile["apple_metadata_count"]
        or len(skipped) != len(set(skipped))
        or report["extracted_regular_files"]
        != profile["type_counts"].get("regular", 0)
        - profile["apple_metadata_count"]
        or report["extracted_directories"]
        != profile["type_counts"].get("directory", 0)
        or any(
            report[field] != 0
            for field in (
                "symlinks_extracted",
                "hardlinks_extracted",
                "path_traversal_extracted",
            )
        )
    ):
        raise ValueError(f"{archive_id} transport report frozen result mismatch")
    return {
        "archive_id": archive_id,
        "archive_sha256": report["archive_sha256"],
        "report_file_sha256": sha256_file(path),
        "report_self_sha256": self_sha,
        "report_size_bytes": path.stat().st_size,
        "apple_metadata_skipped_count": report["apple_metadata_skipped_count"],
        "apple_metadata_skipped_names_sha256": sha256_bytes(
            canonical_json_bytes(skipped)
        ),
    }


def _read_exact_document(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_self_sha256: str,
    expected_fields: set[str],
    context: str,
) -> dict[str, Any]:
    if sha256_file(path) != expected_file_sha256:
        raise ValueError(f"{context} file SHA mismatch")
    value = _json(path, context)
    if set(value) != expected_fields:
        raise ValueError(f"{context} exact schema mismatch")
    if _self_hash(value, context) != expected_self_sha256:
        raise ValueError(f"{context} expected self SHA mismatch")
    return value


def _false_authorizations(value: dict[str, Any]) -> bool:
    return all(
        value[field] is False
        for field in (
            "teacher_authorized",
            "model_authorized",
            "review_authorized",
            "student_gpu_authorized",
            "public_used",
        )
    )


def _validate_live_go(
    args: argparse.Namespace, contract: dict[str, Any]
) -> dict[str, Any]:
    value = _read_exact_document(
        args.live_go,
        expected_file_sha256=args.live_go_sha256,
        expected_self_sha256=args.live_go_self_sha256,
        expected_fields=LIVE_GO_FIELDS,
        context="independent live GO",
    )
    sha_fields = [field for field in LIVE_GO_FIELDS if field.endswith("_sha256")]
    if (
        any(not HEX64.fullmatch(value[field]) for field in sha_fields)
        or not HEX64.fullmatch(value["one_shot_nonce"])
    ):
        raise ValueError("independent live GO contains a non-exact SHA")
    if (
        value["schema_version"] != "exp689_source_prepare_retry_live_go_v1"
        or value["experiment_id"] != "689"
        or value["issuer_role"] != "independent_integrator"
        or value["decision"] != "OPEN_EXACT_ONE_CPU_SOURCE_PREPARE_RETRY"
        or value["gate_file_sha256"] != args.retry_gate_sha256
        or value["gate_self_sha256"] != args.retry_gate_self_sha256
        or value["contract_file_sha256"] != args.retry_contract_sha256
        or value["contract_self_sha256"] != args.retry_contract_self_sha256
        or value["materialization_receipt_file_sha256"]
        != args.materialization_receipt_sha256
        or value["materialization_receipt_self_sha256"]
        != args.materialization_receipt_self_sha256
        or value["builder_commit"] != args.retry_revision
        or value["builder_sha256"] != args.retry_preset_builder_sha256
        or value["output_bucket"] != contract["bucket"]
        or value["output_prefix"] != contract["outputs"]["source_prepare"]
        or value["transport_report_prefix"]
        != contract["outputs"]["transport_reports"]
        or value["audit_preset_errors"] != 0
        or value["audit_preset_warnings"] != 1
        or value["server_dry_run_passed"] is not True
        or isinstance(value["empty_prefix_max_age_seconds"], bool)
        or value["empty_prefix_max_age_seconds"] not in range(1, 121)
        or value["max_jobs"] != 1
        or value["retry_attempt"] != 1
        or value["resolved_terminal_metadata_required"] is not True
        or not _false_authorizations(value)
        or not isinstance(value["preset_size_bytes"], int)
        or isinstance(value["preset_size_bytes"], bool)
        or value["preset_size_bytes"] <= 0
    ):
        raise ValueError("independent live GO frozen authorization mismatch")
    return value


def _validate_materialization_receipt(
    args: argparse.Namespace, live_go: dict[str, Any]
) -> dict[str, Any]:
    value = _read_exact_document(
        args.materialization_receipt,
        expected_file_sha256=args.materialization_receipt_sha256,
        expected_self_sha256=args.materialization_receipt_self_sha256,
        expected_fields=MATERIALIZATION_RECEIPT_FIELDS,
        context="materialization receipt",
    )
    if (
        value["schema_version"]
        != "exp689_source_prepare_retry_materialization_v1"
        or value["experiment_id"] != "689"
        or value["scope"] != "source_prepare_transport_retry_only"
        or value["retry_attempt"] != 1
        or value["diagnostic_acceptance_file_sha256"]
        != args.diagnostic_acceptance_sha256
        or value["diagnostic_acceptance_self_sha256"]
        != args.diagnostic_acceptance_self_sha256
        or value["transport_retry_gate_file_sha256"] != args.retry_gate_sha256
        or value["transport_retry_gate_self_sha256"]
        != args.retry_gate_self_sha256
        or value["preset_contract_file_sha256"] != args.retry_contract_sha256
        or value["preset_semantic_contract_sha256"]
        != args.retry_contract_self_sha256
        or value["final_preset_sha256"] != live_go["preset_sha256"]
        or value["final_preset_size_bytes"] != live_go["preset_size_bytes"]
        or value["retry_preset_builder_sha256"]
        != args.retry_preset_builder_sha256
        or value["controlled_prepare_retry_authorized"] is not True
        or value["max_jobs"] != 1
        or value["teacher_authorized"] is not False
        or value["student_gpu_authorized"] is not False
        or value["public_used"] is not False
    ):
        raise ValueError("materialization receipt frozen lineage mismatch")
    return value


def _validate_submit_receipt(
    args: argparse.Namespace,
    live_go: dict[str, Any],
    materialization: dict[str, Any],
) -> dict[str, Any]:
    value = _read_exact_document(
        args.submit_receipt,
        expected_file_sha256=args.submit_receipt_sha256,
        expected_self_sha256=args.submit_receipt_self_sha256,
        expected_fields=SUBMIT_RECEIPT_FIELDS,
        context="one-shot submit receipt",
    )
    proof = value["empty_prefix_proof"]
    if not isinstance(proof, dict):
        raise TypeError("one-shot submit receipt empty-prefix proof missing")
    proof_self = _self_hash(proof, "empty-prefix proof")
    expected_prefixes = [
        {"prefix": live_go["output_prefix"], "object_count": 0},
        {"prefix": live_go["transport_report_prefix"], "object_count": 0},
    ]
    if (
        set(proof)
        != {
            "schema_version",
            "checked_at_utc",
            "bucket",
            "prefixes",
            "method",
            "max_age_seconds",
            "self_sha256",
        }
        or not isinstance(proof["checked_at_utc"], str)
        or not proof["checked_at_utc"].strip()
        or proof.get("schema_version")
        != "exp689_source_prepare_retry_empty_prefix_proof_v1"
        or proof.get("bucket") != live_go["output_bucket"]
        or proof.get("prefixes") != expected_prefixes
        or proof.get("method") != "s3_list_limit_1"
        or proof.get("max_age_seconds") != live_go["empty_prefix_max_age_seconds"]
        or proof_self != value["empty_prefix_proof_self_sha256"]
    ):
        raise ValueError("one-shot submit receipt empty-prefix proof mismatch")
    if (
        value["schema_version"] != "exp689_source_prepare_retry_submit_receipt_v1"
        or value["experiment_id"] != "689"
        or value["state"] != "SUBMITTED"
        or value["returncode"] != 0
        or not isinstance(value["job_id"], str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", value["job_id"])
        or value["error_class"] is not None
        or value["live_go_file_sha256"] != args.live_go_sha256
        or value["live_go_self_sha256"] != args.live_go_self_sha256
        or value["actual_sent_clean_preset_sha256"] != live_go["preset_sha256"]
        or value["actual_sent_clean_preset_size_bytes"]
        != live_go["preset_size_bytes"]
        or value["gate_file_sha256"] != args.retry_gate_sha256
        or value["gate_self_sha256"] != args.retry_gate_self_sha256
        or value["contract_file_sha256"] != args.retry_contract_sha256
        or value["contract_self_sha256"] != args.retry_contract_self_sha256
        or value["materialization_receipt_file_sha256"]
        != args.materialization_receipt_sha256
        or value["materialization_receipt_self_sha256"]
        != materialization["self_sha256"]
        or value["materialization_args_file_sha256"]
        != live_go["materialization_args_file_sha256"]
        or value["builder_commit"] != live_go["builder_commit"]
        or value["builder_sha256"] != live_go["builder_sha256"]
        or value["submit_wrapper_sha256"] != live_go["submit_wrapper_sha256"]
        or value["submit_wrapper_test_sha256"]
        != live_go["submit_wrapper_test_sha256"]
        or value["s3_endpoint"] != live_go["s3_endpoint"]
        or value["s3_cli_sha256"] != live_go["s3_cli_sha256"]
        or value["s3_requirements_sha256"] != live_go["s3_requirements_sha256"]
        or value["project_file_sha256"] != live_go["project_file_sha256"]
        or value["one_shot_nonce"] != live_go["one_shot_nonce"]
        or value["receipt_path_sha256"] != live_go["receipt_path_sha256"]
        or value["output_bucket"] != live_go["output_bucket"]
        or value["output_prefix"] != live_go["output_prefix"]
        or value["transport_report_prefix"]
        != live_go["transport_report_prefix"]
        or value["max_jobs"] != 1
        or value["retry_attempt"] != 1
        or value["resolved_terminal_metadata_required"] is not True
        or value["audit_preset_result_sha256"]
        != live_go["audit_preset_result_sha256"]
        or value["server_dry_run_receipt_sha256"]
        != live_go["server_dry_run_receipt_sha256"]
        or value["secret_payload_persisted"] is not False
        or value["process_output_persisted"] is not False
        or not _false_authorizations(value)
    ):
        raise ValueError("one-shot submit receipt frozen lineage mismatch")
    return value


def _object_binding(reference: str, file_sha256: str, self_sha256: str) -> dict[str, str]:
    return {
        "reference": reference,
        "file_sha256": file_sha256,
        "self_sha256": self_sha256,
    }


def _validate_terminal_attempts(value: dict[str, Any]) -> None:
    platform_attempt = value["platform_attempt"]
    if (
        value["retry_attempt"] != 1
        or not isinstance(platform_attempt, int)
        or isinstance(platform_attempt, bool)
        or platform_attempt != 1
    ):
        raise ValueError("resolved terminal retry/platform attempt mismatch")


def _validate_resolved_terminal_metadata(
    args: argparse.Namespace,
    *,
    contract: dict[str, Any],
    live_go: dict[str, Any],
    materialization: dict[str, Any],
    submit_receipt: dict[str, Any],
    terminal_report_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    value = _read_exact_document(
        args.resolved_terminal_metadata,
        expected_file_sha256=args.resolved_terminal_metadata_sha256,
        expected_self_sha256=args.resolved_terminal_metadata_self_sha256,
        expected_fields=RESOLVED_TERMINAL_FIELDS,
        context="independently exported resolved terminal metadata",
    )
    _validate_terminal_attempts(value)
    expected_inputs = {
        "bundle": {
            "reference": args.retry_bundle_ref,
            "sha256": args.retry_bundle_sha256,
        },
        "manifest": _object_binding(
            args.retry_manifest_ref,
            args.retry_manifest_sha256,
            args.retry_manifest_self_sha256,
        ),
        "source_f03": {
            "reference": args.source_f03_ref,
            "sha256": args.source_f03_sha256,
        },
        "source_f124": {
            "reference": args.source_f124_ref,
            "sha256": args.source_f124_sha256,
        },
        "exclusion_670": {
            "reference": args.exclusion_670_ref,
            "sha256": args.exclusion_670_sha256,
        },
        "exclusion_672": {
            "reference": args.exclusion_672_ref,
            "sha256": args.exclusion_672_sha256,
        },
        "diagnostic_acceptance": _object_binding(
            args.diagnostic_acceptance_ref,
            args.diagnostic_acceptance_sha256,
            args.diagnostic_acceptance_self_sha256,
        ),
        "retry_contract": _object_binding(
            args.retry_contract_ref,
            args.retry_contract_sha256,
            args.retry_contract_self_sha256,
        ),
        "retry_gate": _object_binding(
            args.retry_gate_ref,
            args.retry_gate_sha256,
            args.retry_gate_self_sha256,
        ),
    }
    resolved_outputs = value["resolved_outputs"]
    if not isinstance(resolved_outputs, dict):
        raise TypeError("resolved terminal outputs must be an object")
    prepared_output = resolved_outputs.get("prepared")
    if (
        not isinstance(prepared_output, dict)
        or set(prepared_output) != {"reference", "inventory_sha256", "inventory"}
        or prepared_output["reference"] != args.prepare_output_ref
        or not isinstance(prepared_output["inventory"], list)
        or not isinstance(prepared_output["inventory_sha256"], str)
        or not HEX64.fullmatch(prepared_output["inventory_sha256"])
        or prepared_output["inventory_sha256"]
        != sha256_bytes(canonical_json_bytes(prepared_output["inventory"]))
    ):
        raise ValueError("resolved prepared-output inventory mismatch")
    expected_outputs = {
        "prepared": prepared_output,
        "transport_reports": {
            "prefix": _expected_ref(
                contract, contract["outputs"]["transport_reports"]
            ),
            "inventory": terminal_report_inventory,
        },
    }
    expected_retry_code = {
        "builder_commit": args.retry_revision,
        "bundle": {
            "reference": args.retry_bundle_ref,
            "sha256": args.retry_bundle_sha256,
        },
        "manifest": {
            "reference": args.retry_manifest_ref,
            "file_sha256": args.retry_manifest_sha256,
            "self_sha256": args.retry_manifest_self_sha256,
        },
        "extractor_sha256": args.extractor_sha256,
    }
    if (
        value["schema_version"]
        != "exp689_source_prepare_retry_resolved_terminal_metadata_v1"
        or value["metadata_source"] != "remote_compute_resolved_job_api_independent"
        or value["experiment_id"] != "689"
        or value["job_id"] != submit_receipt["job_id"]
        or value["status"] != "SUCCESS"
        or value["exit_code"] != 0
        or not isinstance(value["finished_at"], str)
        or not value["finished_at"].strip()
        or value["region"] != contract["job"]["region"]
        or value["flavor"] != contract["job"]["flavor"]
        or value["image"] != contract["job"]["image"]
        or value["gpu_count"] != 0
        or value["gpu_count"] != contract["job"]["gpu_count"]
        or value["preset_sha256"] != live_go["preset_sha256"]
        or value["preset_sha256"] != materialization["final_preset_sha256"]
        or value["preset_size_bytes"] != live_go["preset_size_bytes"]
        or value["preset_size_bytes"] != materialization["final_preset_size_bytes"]
        or value["resolved_command_sha256"] != args.expected_resolved_command_sha256
        or value["submit_receipt"]
        != _object_binding(
            args.submit_receipt_ref,
            args.submit_receipt_sha256,
            args.submit_receipt_self_sha256,
        )
        or value["live_go"]
        != _object_binding(
            args.live_go_ref, args.live_go_sha256, args.live_go_self_sha256
        )
        or value["materialization_receipt"]
        != _object_binding(
            args.materialization_receipt_ref,
            args.materialization_receipt_sha256,
            args.materialization_receipt_self_sha256,
        )
        or value["retry_gate"] != expected_inputs["retry_gate"]
        or value["retry_contract"] != expected_inputs["retry_contract"]
        or value["retry_code"] != expected_retry_code
        or value["resolved_inputs"] != expected_inputs
        or value["resolved_outputs"] != expected_outputs
    ):
        raise ValueError("resolved terminal metadata frozen lineage mismatch")
    return value


def verify(args: argparse.Namespace) -> dict[str, Any]:
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
        raise ValueError("transport-aware verifier requires exact SHA-256 bindings")
    if not HEX40.fullmatch(args.verifier_revision) or not HEX40.fullmatch(
        args.retry_revision
    ):
        raise ValueError("transport-aware verifier requires exact Git revisions")
    for path in (
        args.verifier_manifest,
        args.retry_manifest,
        args.source_f03,
        args.source_f124,
        args.exclusion_670,
        args.exclusion_672,
        args.diagnostic_acceptance,
        args.retry_contract,
        args.retry_gate,
        args.submit_receipt,
        args.live_go,
        args.materialization_receipt,
        args.resolved_terminal_metadata,
        args.terminal_transport_f03,
        args.terminal_transport_f124,
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError("verifier inputs must be regular non-symlink files")
    for path in (
        args.base_acceptance,
        args.transport_report_dir,
        args.runtime_root,
        args.acceptance,
    ):
        if path.exists():
            raise FileExistsError("refusing to overwrite verifier output")

    verifier_manifest = _validate_code_manifest(
        root=args.verifier_bundle_root,
        manifest_path=args.verifier_manifest,
        expected_file_sha256=args.verifier_manifest_sha256,
        expected_self_sha256=args.verifier_manifest_self_sha256,
        expected_revision=args.verifier_revision,
        expected_schema="exp689_source_prepare_retry_verifier_bundle_manifest_v1",
        expected_paths={RUNNER_PATH},
    )
    runner = args.verifier_bundle_root / RUNNER_PATH
    if (
        sha256_file(runner) != args.verifier_runner_sha256
        or verifier_manifest["files"][0]["sha256"] != args.verifier_runner_sha256
    ):
        raise ValueError("transport-aware verifier runner SHA mismatch")
    retry_manifest = _validate_code_manifest(
        root=args.retry_bundle_root,
        manifest_path=args.retry_manifest,
        expected_file_sha256=args.retry_manifest_sha256,
        expected_self_sha256=args.retry_manifest_self_sha256,
        expected_revision=args.retry_revision,
        expected_schema="exp689_source_prepare_bundle_manifest_v2_transport",
        expected_paths=RETRY_PATHS,
    )
    extractor_path = args.retry_bundle_root / f"{EXPERIMENT}/extract_source_archive_transport.py"
    if (
        sha256_file(extractor_path) != args.extractor_sha256
        or next(
            item["sha256"]
            for item in retry_manifest["files"]
            if item["path"].endswith("extract_source_archive_transport.py")
        )
        != args.extractor_sha256
    ):
        raise ValueError("gate-authorized extractor SHA mismatch")

    transport = _load_module("exp689_retry_transport", extractor_path)
    source_verifier = _load_module(
        "exp689_retry_source_verifier",
        args.retry_bundle_root / f"{EXPERIMENT}/verify_source_prepare.py",
    )
    contract = transport.validate_retry_preset_contract(
        args.retry_contract,
        expected_file_sha256=args.retry_contract_sha256,
        expected_self_sha256=args.retry_contract_self_sha256,
    )
    diagnostic = transport.validate_diagnostic_acceptance(
        args.diagnostic_acceptance, args.diagnostic_acceptance_sha256
    )
    if diagnostic["self_sha256"] != args.diagnostic_acceptance_self_sha256:
        raise ValueError("diagnostic acceptance expected self SHA mismatch")
    transport.validate_transport_retry_gate(
        args.retry_gate,
        expected_file_sha256=args.retry_gate_sha256,
        expected_self_sha256=args.retry_gate_self_sha256,
        diagnostic_acceptance=diagnostic,
        diagnostic_acceptance_file_sha256=args.diagnostic_acceptance_sha256,
        expected_verifier_terminal_metadata_sha256=(
            args.diagnostic_verifier_terminal_metadata_sha256
        ),
        expected_retry_code_commit=args.retry_revision,
        expected_retry_code_bundle_sha256=args.retry_bundle_sha256,
        expected_retry_bundle_manifest_file_sha256=args.retry_manifest_sha256,
        expected_retry_bundle_manifest_self_sha256=args.retry_manifest_self_sha256,
        expected_retry_preset_builder_sha256=args.retry_preset_builder_sha256,
        retry_preset_contract=contract,
        retry_preset_contract_file_sha256=args.retry_contract_sha256,
        expected_retry_preset_contract_sha256=args.retry_contract_self_sha256,
        source_prepare_spec_path=(
            args.retry_bundle_root / f"{EXPERIMENT}/source_prepare_spec_v1.json"
        ),
        expected_output_prefix=args.prepare_output_prefix,
        expected_transport_report_prefix=args.prepare_transport_report_prefix,
    )

    expected_refs = {
        "retry_bundle_ref": _expected_ref(contract, contract["inputs"]["bundle"]["key"]),
        "retry_manifest_ref": _expected_ref(
            contract, contract["inputs"]["manifest"]["key"]
        ),
        "source_f03_ref": _expected_ref(
            contract, contract["inputs"]["source_f03"]["key"]
        ),
        "source_f124_ref": _expected_ref(
            contract, contract["inputs"]["source_f124"]["key"]
        ),
        "exclusion_670_ref": _expected_ref(
            contract, contract["inputs"]["exclusion_670"]["key"]
        ),
        "exclusion_672_ref": _expected_ref(
            contract, contract["inputs"]["exclusion_672"]["key"]
        ),
        "diagnostic_acceptance_ref": _expected_ref(
            contract, contract["inputs"]["diagnostic_acceptance"]["key"]
        ),
        "retry_contract_ref": _expected_ref(
            contract, contract["inputs"]["preset_contract_key"]
        ),
        "retry_gate_ref": _expected_ref(
            contract, contract["inputs"]["transport_retry_gate_key"]
        ),
    }
    if any(getattr(args, field) != expected for field, expected in expected_refs.items()):
        raise ValueError("S3 input reference differs from gate-authorized retry contract")
    approved_output_ref = _expected_ref(contract, contract["outputs"]["source_prepare"])
    if (
        args.prepare_output_ref != approved_output_ref
        or args.prepare_output_prefix != contract["outputs"]["source_prepare"]
        or args.prepare_transport_report_prefix
        != contract["outputs"]["transport_reports"]
    ):
        raise ValueError("PREPARE output identity differs from retry contract")
    if (
        args.source_f03_sha256 != transport.PROFILES["source_f03"]["sha256"]
        or args.source_f124_sha256 != transport.PROFILES["source_f124"]["sha256"]
        or args.exclusion_670_sha256
        != transport.EXCLUSION_BINDINGS["exp670_audit_csv_sha256"]
        or args.exclusion_672_sha256
        != transport.EXCLUSION_BINDINGS["exp672_private_manifest_sha256"]
    ):
        raise ValueError("frozen archive/exclusion SHA mismatch")

    live_go = _validate_live_go(args, contract)
    materialization = _validate_materialization_receipt(args, live_go)
    submit_receipt = _validate_submit_receipt(args, live_go, materialization)
    terminal_report_paths = {
        "source_f03": args.terminal_transport_f03,
        "source_f124": args.terminal_transport_f124,
    }
    terminal_report_refs = {
        "source_f03": args.terminal_transport_f03_ref,
        "source_f124": args.terminal_transport_f124_ref,
    }
    expected_terminal_report_refs = {
        archive_id: _expected_ref(
            contract,
            contract["outputs"]["transport_reports"].rstrip("/")
            + f"/{archive_id}_extraction.json",
        )
        for archive_id in ("source_f03", "source_f124")
    }
    if terminal_report_refs != expected_terminal_report_refs:
        raise ValueError("terminal transport report object reference mismatch")
    terminal_report_inventory: list[dict[str, Any]] = []
    for archive_id, file_sha, self_sha in (
        (
            "source_f03",
            args.terminal_transport_f03_sha256,
            args.terminal_transport_f03_self_sha256,
        ),
        (
            "source_f124",
            args.terminal_transport_f124_sha256,
            args.terminal_transport_f124_self_sha256,
        ),
    ):
        binding = _validate_transport_report(
            terminal_report_paths[archive_id],
            archive_id,
            transport.PROFILES[archive_id],
        )
        if (
            binding["report_file_sha256"] != file_sha
            or binding["report_self_sha256"] != self_sha
        ):
            raise ValueError("terminal transport report expected hash mismatch")
        terminal_report_inventory.append(
            {
                "archive_id": archive_id,
                "reference": terminal_report_refs[archive_id],
                "file_sha256": file_sha,
                "self_sha256": self_sha,
                "size_bytes": terminal_report_paths[archive_id].stat().st_size,
            }
        )
    resolved_terminal = _validate_resolved_terminal_metadata(
        args,
        contract=contract,
        live_go=live_go,
        materialization=materialization,
        submit_receipt=submit_receipt,
        terminal_report_inventory=terminal_report_inventory,
    )

    args.transport_report_dir.mkdir(parents=True, exist_ok=False)
    args.runtime_root.mkdir(parents=True, exist_ok=False)
    transport_reports: dict[str, dict[str, Any]] = {}
    runtime_destinations = {
        "source_f03": args.runtime_root / "source_f03",
        "source_f124": args.runtime_root / "source_f124",
    }
    archive_paths = {
        "source_f03": args.source_f03,
        "source_f124": args.source_f124,
    }
    for archive_id in ("source_f03", "source_f124"):
        report_path = args.transport_report_dir / f"{archive_id}_extraction.json"
        transport.extract(
            archive_path=archive_paths[archive_id],
            destination=runtime_destinations[archive_id],
            report_path=report_path,
            archive_id=archive_id,
        )
        transport_reports[archive_id] = _validate_transport_report(
            report_path, archive_id, transport.PROFILES[archive_id]
        )
        if report_path.read_bytes() != terminal_report_paths[archive_id].read_bytes():
            raise ValueError(
                f"{archive_id} regenerated transport report differs from terminal object"
            )

    base_terminal = {
        "schema_version": "exp689_source_prepare_terminal_v1",
        "job_id": resolved_terminal["job_id"],
        "status": resolved_terminal["status"],
        "finished_at": resolved_terminal["finished_at"],
        "output_ref": args.prepare_output_ref,
        "self_sha256": None,
    }
    base_terminal["self_sha256"] = sha256_bytes(canonical_json_bytes(base_terminal))
    base_terminal_path = args.runtime_root / "source_prepare_terminal.json"
    base_terminal_path.write_text(
        json.dumps(base_terminal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    base_terminal_file_sha = sha256_file(base_terminal_path)
    args.base_acceptance.parent.mkdir(parents=True, exist_ok=True)
    base = source_verifier.verify(
        prepare_dir=args.prepare_dir,
        runtime_dirs=[
            runtime_destinations["source_f03"]
            / "experiments/641_qwen35_4b_class_only_lora/.local/runtime/fold0",
            runtime_destinations["source_f124"] / "runtime/fold1",
            runtime_destinations["source_f124"] / "runtime/fold2",
            runtime_destinations["source_f03"]
            / "experiments/641_qwen35_4b_class_only_lora/.local/runtime/fold3",
            runtime_destinations["source_f124"] / "runtime/fold4",
        ],
        runtime_archives=[args.source_f03, args.source_f124],
        runtime_archive_refs=[args.source_f03_ref, args.source_f124_ref],
        exclusion_670_path=args.exclusion_670,
        exclusion_672_path=args.exclusion_672,
        exclusion_670_sha256=args.exclusion_670_sha256,
        exclusion_672_sha256=args.exclusion_672_sha256,
        bundle_root=args.retry_bundle_root,
        bundle_manifest_path=args.retry_manifest,
        bundle_manifest_sha256=args.retry_manifest_sha256,
        builder_revision=args.retry_revision,
        spec_path=(args.retry_bundle_root / f"{EXPERIMENT}/source_prepare_spec_v1.json"),
        terminal_metadata_path=base_terminal_path,
        terminal_metadata_sha256=base_terminal_file_sha,
        approved_s3_output_ref=args.prepare_output_ref,
        acceptance_path=args.base_acceptance,
    )
    expected_runtime_archives = [
        {
            "archive_id": "folds_0_3",
            "folds": [0, 3],
            "reference": args.source_f03_ref,
            "sha256": args.source_f03_sha256,
            "size_bytes": args.source_f03.stat().st_size,
        },
        {
            "archive_id": "folds_1_2_4",
            "folds": [1, 2, 4],
            "reference": args.source_f124_ref,
            "sha256": args.source_f124_sha256,
            "size_bytes": args.source_f124.stat().st_size,
        },
    ]
    if (
        base.get("schema_version") != "exp689_source_prepare_acceptance_v1"
        or base.get("decision") != "ACCEPT"
        or base.get("terminal_metadata_sha256") != base_terminal_file_sha
        or base.get("approved_s3_output_ref") != args.prepare_output_ref
        or base.get("runtime_archives") != expected_runtime_archives
        or resolved_terminal["resolved_outputs"]["prepared"]
        != {
            "reference": args.prepare_output_ref,
            "inventory_sha256": base["output_inventory_sha256"],
            "inventory": base["output_inventory"],
        }
    ):
        raise ValueError("base source PREPARE acceptance identity mismatch")
    base_self = _self_hash(base, "base source PREPARE acceptance")
    if sha256_file(args.base_acceptance) != sha256_bytes(
        (json.dumps(base, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ):
        raise ValueError("base source PREPARE acceptance file serialization mismatch")
    acceptance = {
        "schema_version": "exp689_source_prepare_retry_acceptance_v1",
        "decision": "ACCEPT_TRANSPORT_AWARE_RETRY",
        "verifier_revision": args.verifier_revision,
        "verifier_runner_sha256": args.verifier_runner_sha256,
        "verifier_bundle_sha256": args.verifier_bundle_sha256,
        "verifier_manifest_file_sha256": args.verifier_manifest_sha256,
        "verifier_manifest_self_sha256": args.verifier_manifest_self_sha256,
        "retry_revision": args.retry_revision,
        "retry_bundle_sha256": args.retry_bundle_sha256,
        "retry_manifest_file_sha256": args.retry_manifest_sha256,
        "retry_manifest_self_sha256": args.retry_manifest_self_sha256,
        "extractor_sha256": args.extractor_sha256,
        "diagnostic_acceptance_ref": args.diagnostic_acceptance_ref,
        "diagnostic_acceptance_file_sha256": args.diagnostic_acceptance_sha256,
        "diagnostic_acceptance_self_sha256": diagnostic["self_sha256"],
        "retry_contract_ref": args.retry_contract_ref,
        "retry_contract_file_sha256": args.retry_contract_sha256,
        "retry_contract_self_sha256": contract["self_sha256"],
        "retry_gate_ref": args.retry_gate_ref,
        "retry_gate_file_sha256": args.retry_gate_sha256,
        "retry_gate_self_sha256": args.retry_gate_self_sha256,
        "original_runtime_archives": base["runtime_archives"],
        "transport_reports": transport_reports,
        "terminal_transport_report_inventory": terminal_report_inventory,
        "source_prepare_acceptance_file_sha256": sha256_file(args.base_acceptance),
        "source_prepare_acceptance_self_sha256": base_self,
        "output_inventory_sha256": base["output_inventory_sha256"],
        "output_inventory": base["output_inventory"],
        "submit_receipt_ref": args.submit_receipt_ref,
        "submit_receipt_file_sha256": args.submit_receipt_sha256,
        "submit_receipt_self_sha256": args.submit_receipt_self_sha256,
        "live_go_ref": args.live_go_ref,
        "live_go_file_sha256": args.live_go_sha256,
        "live_go_self_sha256": args.live_go_self_sha256,
        "materialization_receipt_ref": args.materialization_receipt_ref,
        "materialization_receipt_file_sha256": args.materialization_receipt_sha256,
        "materialization_receipt_self_sha256": (
            args.materialization_receipt_self_sha256
        ),
        "resolved_terminal_metadata_ref": args.resolved_terminal_metadata_ref,
        "resolved_terminal_metadata_file_sha256": (
            args.resolved_terminal_metadata_sha256
        ),
        "resolved_terminal_metadata_self_sha256": (
            args.resolved_terminal_metadata_self_sha256
        ),
        "resolved_command_sha256": resolved_terminal["resolved_command_sha256"],
        "terminal_job_id": base["terminal_job_id"],
        "terminal_status": base["terminal_status"],
        "terminal_finished_at": base["terminal_finished_at"],
        "approved_s3_output_ref": args.prepare_output_ref,
        "max_jobs": 1,
        "retry_attempt": 1,
        "platform_attempt": resolved_terminal["platform_attempt"],
        "teacher_authorized": False,
        "model_authorized": False,
        "review_authorized": False,
        "student_gpu_authorized": False,
        "public_used": False,
        "sealed_rows": 0,
        "public_rows": 0,
        "self_sha256": None,
    }
    acceptance["self_sha256"] = sha256_bytes(canonical_json_bytes(acceptance))
    args.acceptance.parent.mkdir(parents=True, exist_ok=True)
    args.acceptance.write_text(
        json.dumps(
            acceptance,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return acceptance


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    for name in ("verifier", "retry"):
        value.add_argument(f"--{name}-bundle-root", type=Path, required=True)
        value.add_argument(f"--{name}-bundle-sha256", required=True)
        value.add_argument(f"--{name}-manifest", type=Path, required=True)
        value.add_argument(f"--{name}-manifest-sha256", required=True)
        value.add_argument(f"--{name}-manifest-self-sha256", required=True)
        value.add_argument(f"--{name}-revision", required=True)
    value.add_argument("--verifier-runner-sha256", required=True)
    value.add_argument("--extractor-sha256", required=True)
    for name in (
        "source-f03",
        "source-f124",
        "exclusion-670",
        "exclusion-672",
    ):
        value.add_argument(f"--{name}", type=Path, required=True)
        value.add_argument(f"--{name}-sha256", required=True)
        value.add_argument(f"--{name}-ref", required=True)
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
        value.add_argument(f"--{name}", type=Path, required=True)
        value.add_argument(f"--{name}-sha256", required=True)
        value.add_argument(f"--{name}-self-sha256", required=True)
        value.add_argument(f"--{name}-ref", required=True)
    value.add_argument(
        "--diagnostic-verifier-terminal-metadata-sha256", required=True
    )
    value.add_argument("--retry-preset-builder-sha256", required=True)
    value.add_argument("--expected-resolved-command-sha256", required=True)
    value.add_argument("--retry-bundle-ref", required=True)
    value.add_argument("--retry-manifest-ref", required=True)
    value.add_argument("--prepare-output-prefix", required=True)
    value.add_argument("--prepare-transport-report-prefix", required=True)
    value.add_argument("--prepare-output-ref", required=True)
    value.add_argument("--prepare-dir", type=Path, required=True)
    value.add_argument("--runtime-root", type=Path, required=True)
    value.add_argument("--transport-report-dir", type=Path, required=True)
    value.add_argument("--base-acceptance", type=Path, required=True)
    value.add_argument("--acceptance", type=Path, required=True)
    return value


def main() -> None:
    result = verify(parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Independently verify the remote exp689 source-archive diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
S3_URI = re.compile(r"^s3://[^/\s]+/[^\s]+$")

AUDIT_CODE_COMMIT = "076c009eaee106caba428c699627f864e302d292"
AUDIT_CODE_SHA256 = "20a55c6e1268881c2281cba8d53efad26b5ad58fdc527d2bbddaae7f5477fc75"
AUDIT_CODE_SIZE_BYTES = 5_313
AUDIT_PRESET_SHA256 = "622370e98f66d1378a5e299d9d10401fd01b0d31f26e5191f3bea36321e59ddb"

ALLOWED_TYPES = {
    "regular",
    "directory",
    "symlink",
    "hardlink",
    "fifo",
    "character_device",
    "block_device",
    "other",
}
ALLOWED_REASONS = {
    "absolute_path",
    "path_traversal",
    "empty_normalized_name",
    "duplicate_normalized_name",
    "apple_metadata",
    "non_regular_or_directory",
}
REPORT_FIELDS = {
    "schema_version",
    "execution_scope",
    "archives",
    "labels_read",
    "sealed_rows_read",
    "public_used",
    "jobs_launched_by_auditor",
    "self_sha256",
}
ARCHIVE_FIELDS = {
    "archive_id",
    "sha256",
    "size_bytes",
    "member_count",
    "type_counts",
    "unsafe_member_count",
    "unsafe_members",
    "file_contents_extracted",
}
UNSAFE_FIELDS = {"name", "normalized_name", "member_type", "reasons"}
RECEIPT_FIELDS = {
    "schema_version",
    "experiment_id",
    "scope",
    "dry_run",
    "returncode",
    "state",
    "job_id",
    "error_class",
    "secret_payload_persisted",
    "process_output_persisted",
}
METADATA_FIELDS = {
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
    "resolved_preset_sha256",
    "resolved_command_sha256",
    "resolved_code_commit",
    "resolved_entrypoint_sha256",
    "resolved_inputs",
    "resolved_output_ref",
    "submit_receipt_sha256",
    "self_sha256",
}
INPUT_FIELDS = {"uri", "sha256", "size_bytes"}
OUTPUT_REF_FIELDS = {
    "schema_version",
    "s3_prefix",
    "immutable_unique_prefix",
    "objects",
}
OUTPUT_OBJECT_FIELDS = {
    "name",
    "uri",
    "sha256",
    "size_bytes",
    "etag",
    "version_id",
}

EXPECTED_ARCHIVES = {
    "source_f03": {
        "sha256": "e371c03a3fc893d990d38874e07200a5ac136b8c72f43109aa7f567761943ccd",
        "size_bytes": 15_852_324,
    },
    "source_f124": {
        "sha256": "1dfb9bf01a9567286051ee76a79fc4b41c2af360e5761b7a4663090745c5474d",
        "size_bytes": 15_000_958,
    },
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def self_hash(value: dict[str, Any], context: str) -> str:
    actual = value.get("self_sha256")
    if not isinstance(actual, str) or not HEX64.fullmatch(actual):
        raise ValueError(f"{context}: invalid self SHA")
    copy = dict(value)
    copy["self_sha256"] = None
    expected = hashlib.sha256(canonical_json_bytes(copy)).hexdigest()
    if actual != expected:
        raise ValueError(f"{context}: self SHA mismatch")
    return actual


def exact_keys(value: dict[str, Any], fields: set[str], context: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{context}: exact fields required")


def load_json(path: Path, context: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def require_hex64(value: str, context: str) -> None:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{context} must be an exact lowercase SHA-256")


def validate_receipt(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    require_hex64(expected_sha256, "submit receipt SHA")
    if sha256_file(path) != expected_sha256:
        raise ValueError("submit receipt file SHA mismatch")
    receipt = load_json(path, "submit receipt")
    exact_keys(receipt, RECEIPT_FIELDS, "submit receipt")
    if (
        receipt["schema_version"] != 1
        or receipt["experiment_id"] != 689
        or receipt["scope"] != "source_archive_header_audit_cpu_only"
        or receipt["dry_run"] is not False
        or receipt["returncode"] != 0
        or receipt["state"] != "SUBMITTED"
        or receipt["error_class"] is not None
        or receipt["secret_payload_persisted"] is not False
        or receipt["process_output_persisted"] is not False
        or not isinstance(receipt["job_id"], str)
        or not SAFE_JOB_ID.fullmatch(receipt["job_id"])
    ):
        raise ValueError("submit receipt scope/safety mismatch")
    identity = hashlib.sha256(receipt["job_id"].encode("utf-8")).hexdigest()
    return receipt, identity


def validate_input(
    value: Any,
    *,
    expected_uri: str,
    expected_sha256: str,
    expected_size_bytes: int,
    context: str,
) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    exact_keys(value, INPUT_FIELDS, context)
    if value != {
        "uri": expected_uri,
        "sha256": expected_sha256,
        "size_bytes": expected_size_bytes,
    }:
        raise ValueError(f"{context} exact immutable binding mismatch")


def validate_metadata(
    path: Path,
    *,
    expected_sha256: str,
    receipt_sha256: str,
    receipt_job_identity_sha256: str,
    expected_command_sha256: str,
    expected_code_uri: str,
    expected_f03_uri: str,
    expected_f124_uri: str,
    expected_output_prefix: str,
    report_path: Path,
    expected_report_sha256: str,
    expected_report_size_bytes: int,
) -> dict[str, Any]:
    require_hex64(expected_sha256, "resolved metadata SHA")
    require_hex64(expected_command_sha256, "resolved command SHA")
    if sha256_file(path) != expected_sha256:
        raise ValueError("resolved ML Core metadata file SHA mismatch")
    metadata = load_json(path, "resolved ML Core metadata")
    exact_keys(metadata, METADATA_FIELDS, "resolved ML Core metadata")
    self_hash(metadata, "resolved ML Core metadata")
    if (
        metadata["schema_version"] != "exp689_source_archive_resolved_job_metadata_v1"
        or metadata["metadata_source"] != "mlcore_resolved_job_api_independent"
        or metadata["job_identity_sha256"] != receipt_job_identity_sha256
        or metadata["terminal_state"] != "SUCCESS"
        or metadata["exit_code"] != 0
        or metadata["failure_reason"] is not None
        or metadata["attempt"] != 1
        or metadata["gpu_count"] != 0
        or metadata["resolved_preset_sha256"] != AUDIT_PRESET_SHA256
        or metadata["resolved_command_sha256"] != expected_command_sha256
        or metadata["resolved_code_commit"] != AUDIT_CODE_COMMIT
        or metadata["resolved_entrypoint_sha256"] != AUDIT_CODE_SHA256
        or metadata["submit_receipt_sha256"] != receipt_sha256
    ):
        raise ValueError("resolved ML Core terminal/job identity mismatch")
    for field in ("submitted_at_utc", "started_at_utc", "finished_at_utc"):
        if not isinstance(metadata[field], str) or not TIMESTAMP.fullmatch(metadata[field]):
            raise ValueError("resolved ML Core timestamp mismatch")
    inputs = metadata["resolved_inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {
        "audit_code",
        "source_f03",
        "source_f124",
    }:
        raise ValueError("resolved ML Core exact input set mismatch")
    validate_input(
        inputs["audit_code"],
        expected_uri=expected_code_uri,
        expected_sha256=AUDIT_CODE_SHA256,
        expected_size_bytes=AUDIT_CODE_SIZE_BYTES,
        context="resolved audit-code input",
    )
    validate_input(
        inputs["source_f03"],
        expected_uri=expected_f03_uri,
        expected_sha256=EXPECTED_ARCHIVES["source_f03"]["sha256"],
        expected_size_bytes=EXPECTED_ARCHIVES["source_f03"]["size_bytes"],
        context="resolved f03 input",
    )
    validate_input(
        inputs["source_f124"],
        expected_uri=expected_f124_uri,
        expected_sha256=EXPECTED_ARCHIVES["source_f124"]["sha256"],
        expected_size_bytes=EXPECTED_ARCHIVES["source_f124"]["size_bytes"],
        context="resolved f124 input",
    )
    output_ref = metadata["resolved_output_ref"]
    if not isinstance(output_ref, dict):
        raise TypeError("resolved output ref must be an object")
    exact_keys(output_ref, OUTPUT_REF_FIELDS, "resolved output ref")
    if (
        output_ref["schema_version"] != "exp689_immutable_s3_output_ref_v1"
        or output_ref["s3_prefix"] != expected_output_prefix
        or output_ref["immutable_unique_prefix"] is not True
    ):
        raise ValueError("resolved output prefix mismatch")
    objects = output_ref["objects"]
    if not isinstance(objects, list) or len(objects) != 1:
        raise ValueError("resolved output ref requires exactly one object")
    output_object = objects[0]
    if not isinstance(output_object, dict):
        raise TypeError("resolved output object must be an object")
    exact_keys(output_object, OUTPUT_OBJECT_FIELDS, "resolved output object")
    expected_uri = expected_output_prefix.rstrip("/") + "/archive_header_audit.json"
    if (
        output_object["name"] != "archive_header_audit.json"
        or output_object["uri"] != expected_uri
        or output_object["sha256"] != expected_report_sha256
        or output_object["size_bytes"] != expected_report_size_bytes
        or not isinstance(output_object["etag"], str)
        or not output_object["etag"]
        or output_object["version_id"] not in (None, "")
        and not isinstance(output_object["version_id"], str)
    ):
        raise ValueError("resolved output object exact binding mismatch")
    if sha256_file(report_path) != expected_report_sha256:
        raise ValueError("mounted archive-audit report SHA mismatch")
    if report_path.stat().st_size != expected_report_size_bytes:
        raise ValueError("mounted archive-audit report size mismatch")
    return metadata


def validate_report(report_path: Path) -> tuple[dict[str, Any], str, list[str]]:
    report = load_json(report_path, "archive-audit report")
    exact_keys(report, REPORT_FIELDS, "archive-audit report")
    report_self = self_hash(report, "archive-audit report")
    if (
        report["schema_version"] != "exp689_source_archive_header_audit_v1"
        or report["execution_scope"] != "remote_cpu_header_only"
        or report["labels_read"] != 0
        or report["sealed_rows_read"] != 0
        or report["public_used"] is not False
        or report["jobs_launched_by_auditor"] != 0
    ):
        raise ValueError("archive-audit report scope/safety mismatch")
    archives = report["archives"]
    if not isinstance(archives, list) or len(archives) != 2:
        raise ValueError("archive-audit report requires exact two archives")
    by_id: dict[str, dict[str, Any]] = {}
    all_reasons: set[str] = set()
    for archive in archives:
        if not isinstance(archive, dict):
            raise TypeError("archive-audit archive entry must be an object")
        exact_keys(archive, ARCHIVE_FIELDS, "archive-audit archive")
        archive_id = archive["archive_id"]
        if archive_id not in EXPECTED_ARCHIVES or archive_id in by_id:
            raise ValueError("archive-audit archive identity mismatch")
        expected = EXPECTED_ARCHIVES[archive_id]
        if (
            archive["sha256"] != expected["sha256"]
            or archive["size_bytes"] != expected["size_bytes"]
        ):
            raise ValueError("archive-audit frozen object binding mismatch")
        if archive["file_contents_extracted"] is not False:
            raise ValueError("archive-audit claims payload extraction")
        counts = archive["type_counts"]
        if (
            not isinstance(counts, dict)
            or not set(counts).issubset(ALLOWED_TYPES)
            or any(not isinstance(count, int) or count < 0 for count in counts.values())
            or sum(counts.values()) != archive["member_count"]
        ):
            raise ValueError("archive-audit member-type counts mismatch")
        unsafe = archive["unsafe_members"]
        if not isinstance(unsafe, list) or archive["unsafe_member_count"] != len(unsafe):
            raise ValueError("archive-audit unsafe-member count mismatch")
        for item in unsafe:
            if not isinstance(item, dict):
                raise TypeError("archive-audit unsafe member must be an object")
            exact_keys(item, UNSAFE_FIELDS, "archive-audit unsafe member")
            if (
                not isinstance(item["name"], str)
                or not isinstance(item["normalized_name"], str)
                or item["member_type"] not in ALLOWED_TYPES
                or not isinstance(item["reasons"], list)
                or not item["reasons"]
                or not set(item["reasons"]).issubset(ALLOWED_REASONS)
            ):
                raise ValueError("archive-audit unsafe-member schema mismatch")
            all_reasons.update(item["reasons"])
        by_id[archive_id] = archive
    if set(by_id) != set(EXPECTED_ARCHIVES):
        raise ValueError("archive-audit exact archive set mismatch")
    return report, report_self, sorted(all_reasons)


def verify(
    *,
    report_path: Path,
    expected_report_sha256: str,
    expected_report_size_bytes: int,
    submit_receipt_path: Path,
    expected_submit_receipt_sha256: str,
    resolved_metadata_path: Path,
    expected_resolved_metadata_sha256: str,
    expected_command_sha256: str,
    expected_code_uri: str,
    expected_f03_uri: str,
    expected_f124_uri: str,
    approved_output_prefix: str,
    expected_verifier_sha256: str,
    verifier_commit: str,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite archive-audit acceptance")
    require_hex64(expected_report_sha256, "report SHA")
    require_hex64(expected_verifier_sha256, "verifier SHA")
    if not HEX40.fullmatch(verifier_commit):
        raise ValueError("verifier commit must be an exact lowercase Git SHA")
    for uri, context in (
        (expected_code_uri, "audit-code input URI"),
        (expected_f03_uri, "f03 input URI"),
        (expected_f124_uri, "f124 input URI"),
        (approved_output_prefix.rstrip("/") + "/sentinel", "output prefix"),
    ):
        if not S3_URI.fullmatch(uri):
            raise ValueError(f"{context} must be an exact S3 URI")
    if sha256_file(Path(__file__)) != expected_verifier_sha256:
        raise ValueError("executing archive verifier SHA mismatch")
    _, job_identity = validate_receipt(
        submit_receipt_path, expected_submit_receipt_sha256
    )
    metadata = validate_metadata(
        resolved_metadata_path,
        expected_sha256=expected_resolved_metadata_sha256,
        receipt_sha256=expected_submit_receipt_sha256,
        receipt_job_identity_sha256=job_identity,
        expected_command_sha256=expected_command_sha256,
        expected_code_uri=expected_code_uri,
        expected_f03_uri=expected_f03_uri,
        expected_f124_uri=expected_f124_uri,
        expected_output_prefix=approved_output_prefix,
        report_path=report_path,
        expected_report_sha256=expected_report_sha256,
        expected_report_size_bytes=expected_report_size_bytes,
    )
    _, report_self, unsafe_reasons = validate_report(report_path)
    if not unsafe_reasons:
        decision = "NO_UNSAFE_MEMBERS_EXTRACTOR_MISMATCH"
    elif unsafe_reasons == ["apple_metadata"]:
        decision = "TRANSPORT_CAUSE_APPLE_METADATA_ONLY"
    else:
        decision = "REQUIRE_NEW_VERIFIED_CLEAN_SOURCE_OBJECT"
    acceptance = {
        "schema_version": "exp689_source_archive_audit_acceptance_v2",
        "status": "accepted_diagnostic_only",
        "decision": decision,
        "independent_remote_provenance_verified": True,
        "report_sha256": expected_report_sha256,
        "report_size_bytes": expected_report_size_bytes,
        "report_self_sha256": report_self,
        "diagnostic_code_commit": AUDIT_CODE_COMMIT,
        "diagnostic_code_sha256": AUDIT_CODE_SHA256,
        "diagnostic_preset_sha256": AUDIT_PRESET_SHA256,
        "diagnostic_command_sha256": expected_command_sha256,
        "submit_receipt_sha256": expected_submit_receipt_sha256,
        "resolved_metadata_sha256": expected_resolved_metadata_sha256,
        "resolved_metadata_self_sha256": metadata["self_sha256"],
        "verifier_commit": verifier_commit,
        "verifier_sha256": expected_verifier_sha256,
        "archive_bindings": EXPECTED_ARCHIVES,
        "unsafe_reason_set": unsafe_reasons,
        "prepare_retry_authorized": False,
        "teacher_authorized": False,
        "student_gpu_authorized": False,
        "self_sha256": None,
    }
    acceptance["self_sha256"] = hashlib.sha256(
        canonical_json_bytes(acceptance)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--expected-report-size-bytes", type=int, required=True)
    parser.add_argument("--submit-receipt", type=Path, required=True)
    parser.add_argument("--expected-submit-receipt-sha256", required=True)
    parser.add_argument("--resolved-metadata", type=Path, required=True)
    parser.add_argument("--expected-resolved-metadata-sha256", required=True)
    parser.add_argument("--expected-command-sha256", required=True)
    parser.add_argument("--expected-code-uri", required=True)
    parser.add_argument("--expected-f03-uri", required=True)
    parser.add_argument("--expected-f124-uri", required=True)
    parser.add_argument("--approved-output-prefix", required=True)
    parser.add_argument("--expected-verifier-sha256", required=True)
    parser.add_argument("--verifier-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        report_path=args.report,
        expected_report_sha256=args.expected_report_sha256,
        expected_report_size_bytes=args.expected_report_size_bytes,
        submit_receipt_path=args.submit_receipt,
        expected_submit_receipt_sha256=args.expected_submit_receipt_sha256,
        resolved_metadata_path=args.resolved_metadata,
        expected_resolved_metadata_sha256=args.expected_resolved_metadata_sha256,
        expected_command_sha256=args.expected_command_sha256,
        expected_code_uri=args.expected_code_uri,
        expected_f03_uri=args.expected_f03_uri,
        expected_f124_uri=args.expected_f124_uri,
        approved_output_prefix=args.approved_output_prefix,
        expected_verifier_sha256=args.expected_verifier_sha256,
        verifier_commit=args.verifier_commit,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

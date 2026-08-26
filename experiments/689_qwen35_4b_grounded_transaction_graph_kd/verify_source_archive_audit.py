"""Verify the compact remote exp689 source-archive header audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
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


def verify(
    *,
    report_path: Path,
    expected_report_sha256: str,
    expected_code_commit: str,
    expected_code_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite archive-audit acceptance")
    if not HEX40.fullmatch(expected_code_commit):
        raise ValueError("code commit must be exact lowercase Git SHA")
    for value in (expected_report_sha256, expected_code_sha256):
        if not HEX64.fullmatch(value):
            raise ValueError("report/code SHA must be exact lowercase SHA-256")
    if sha256_file(report_path) != expected_report_sha256:
        raise ValueError("archive-audit report file SHA mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("archive-audit report must be an object")
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
    expected_archives = {
        "source_f03": (
            "e371c03a3fc893d990d38874e07200a5ac136b8c72f43109aa7f567761943ccd",
            15_852_324,
        ),
        "source_f124": (
            "1dfb9bf01a9567286051ee76a79fc4b41c2af360e5761b7a4663090745c5474d",
            15_000_958,
        ),
    }
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
        if archive_id not in expected_archives or archive_id in by_id:
            raise ValueError("archive-audit archive identity mismatch")
        expected_sha, expected_size = expected_archives[archive_id]
        if archive["sha256"] != expected_sha or archive["size_bytes"] != expected_size:
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
        if (
            not isinstance(unsafe, list)
            or archive["unsafe_member_count"] != len(unsafe)
        ):
            raise ValueError("archive-audit unsafe-member count mismatch")
        for item in unsafe:
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
    if set(by_id) != set(expected_archives):
        raise ValueError("archive-audit exact archive set mismatch")
    if not all_reasons:
        decision = "NO_UNSAFE_MEMBERS_EXTRACTOR_MISMATCH"
    elif all_reasons == {"apple_metadata"}:
        decision = "TRANSPORT_CAUSE_APPLE_METADATA_ONLY"
    else:
        decision = "REQUIRE_NEW_VERIFIED_CLEAN_SOURCE_OBJECT"
    acceptance = {
        "schema_version": "exp689_source_archive_audit_acceptance_v1",
        "status": "accepted_diagnostic_only",
        "decision": decision,
        "report_sha256": expected_report_sha256,
        "report_self_sha256": report_self,
        "code_commit": expected_code_commit,
        "code_sha256": expected_code_sha256,
        "archive_bindings": {
            key: {"sha256": value[0], "size_bytes": value[1]}
            for key, value in sorted(expected_archives.items())
        },
        "unsafe_reason_set": sorted(all_reasons),
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
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-code-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        report_path=args.report,
        expected_report_sha256=args.expected_report_sha256,
        expected_code_commit=args.expected_code_commit,
        expected_code_sha256=args.expected_code_sha256,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

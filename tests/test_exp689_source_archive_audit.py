from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "689_qwen35_4b_grounded_transaction_graph_kd"
)
sys.path.insert(0, str(EXPERIMENT))

import audit_source_archives as audit
import verify_source_archive_audit as verifier


def make_tar(path: Path, *, unsafe: bool) -> None:
    with tarfile.open(path, "w:gz") as archive:
        payload = b"runtime-contract"
        info = tarfile.TarInfo("runtime/fold0/runtime_audit.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        if unsafe:
            link = tarfile.TarInfo("runtime/fold0/latest")
            link.type = tarfile.SYMTYPE
            link.linkname = "runtime_audit.json"
            archive.addfile(link)


def test_header_audit_reports_link_without_extracting_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safe = tmp_path / "safe.tar.gz"
    unsafe = tmp_path / "unsafe.tar.gz"
    make_tar(safe, unsafe=False)
    make_tar(unsafe, unsafe=True)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("payload extraction is forbidden")

    monkeypatch.setattr(tarfile.TarFile, "extract", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extractall", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extractfile", forbidden)
    output = tmp_path / "audit.json"
    result = audit.run(
        source_f03=unsafe,
        source_f03_sha256=audit.sha256_file(unsafe),
        source_f03_size=unsafe.stat().st_size,
        source_f124=safe,
        source_f124_sha256=audit.sha256_file(safe),
        source_f124_size=safe.stat().st_size,
        output=output,
    )
    assert result["archives"][0]["unsafe_member_count"] == 1
    assert result["archives"][0]["unsafe_members"] == [
        {
            "name": "runtime/fold0/latest",
            "normalized_name": "runtime/fold0/latest",
            "member_type": "symlink",
            "reasons": ["non_regular_or_directory"],
        }
    ]
    assert result["archives"][1]["unsafe_member_count"] == 0
    assert all(
        item["file_contents_extracted"] is False for item in result["archives"]
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    expected_self = persisted.pop("self_sha256")
    persisted["self_sha256"] = None
    assert audit.hashlib.sha256(audit.canonical_json_bytes(persisted)).hexdigest() == expected_self


def test_header_audit_requires_exact_size_and_sha(tmp_path: Path) -> None:
    path = tmp_path / "source.tar.gz"
    make_tar(path, unsafe=False)
    with pytest.raises(ValueError, match="size mismatch"):
        audit.audit_archive(path, audit.sha256_file(path), path.stat().st_size + 1, "source")
    with pytest.raises(ValueError, match="SHA mismatch"):
        audit.audit_archive(path, "0" * 64, path.stat().st_size, "source")


def make_report() -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": "exp689_source_archive_header_audit_v1",
        "execution_scope": "remote_cpu_header_only",
        "archives": [
            {
                "archive_id": archive_id,
                "sha256": sha,
                "size_bytes": size,
                "member_count": 1,
                "type_counts": {"regular": 1},
                "unsafe_member_count": 0,
                "unsafe_members": [],
                "file_contents_extracted": False,
            }
            for archive_id, sha, size in (
                (
                    "source_f03",
                    "e371c03a3fc893d990d38874e07200a5ac136b8c72f43109aa7f567761943ccd",
                    15_852_324,
                ),
                (
                    "source_f124",
                    "1dfb9bf01a9567286051ee76a79fc4b41c2af360e5761b7a4663090745c5474d",
                    15_000_958,
                ),
            )
        ],
        "labels_read": 0,
        "sealed_rows_read": 0,
        "public_used": False,
        "jobs_launched_by_auditor": 0,
        "self_sha256": None,
    }
    report["self_sha256"] = audit.hashlib.sha256(
        audit.canonical_json_bytes(report)
    ).hexdigest()
    return report


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def with_self_hash(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["self_sha256"] = None
    result["self_sha256"] = hashlib.sha256(
        verifier.canonical_json_bytes(result)
    ).hexdigest()
    return result


def make_verification_packet(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    report = make_report()
    report_path = tmp_path / "archive_header_audit.json"
    write_json(report_path, report)
    report_sha = verifier.sha256_file(report_path)
    receipt = {
        "schema_version": 1,
        "experiment_id": 689,
        "scope": "source_archive_header_audit_cpu_only",
        "dry_run": False,
        "returncode": 0,
        "state": "SUBMITTED",
        "job_id": "private-diagnostic-id",
        "error_class": None,
        "secret_payload_persisted": False,
        "process_output_persisted": False,
    }
    receipt_path = tmp_path / "submit_receipt.json"
    write_json(receipt_path, receipt)
    receipt_sha = verifier.sha256_file(receipt_path)
    command_sha = "e" * 64
    code_uri = "s3://approved/exp689/audit_source_archives.py"
    f03_uri = "s3://approved/exp689/source-f03.tar.gz"
    f124_uri = "s3://approved/exp689/source-f124.tar.gz"
    output_prefix = "s3://approved/exp689/archive-audit/run1"
    metadata = with_self_hash(
        {
            "schema_version": "exp689_source_archive_resolved_job_metadata_v1",
            "metadata_source": "remote_compute_resolved_job_api_independent",
            "job_identity_sha256": hashlib.sha256(
                receipt["job_id"].encode("utf-8")
            ).hexdigest(),
            "terminal_state": "SUCCESS",
            "exit_code": 0,
            "failure_reason": None,
            "submitted_at_utc": "2026-08-26T10:00:00Z",
            "started_at_utc": "2026-08-26T10:00:01Z",
            "finished_at_utc": "2026-08-26T10:05:00Z",
            "attempt": 1,
            "gpu_count": 0,
            "resolved_preset_sha256": verifier.AUDIT_PRESET_SHA256,
            "resolved_command_sha256": command_sha,
            "resolved_code_commit": verifier.AUDIT_CODE_COMMIT,
            "resolved_entrypoint_sha256": verifier.AUDIT_CODE_SHA256,
            "resolved_inputs": {
                "audit_code": {
                    "uri": code_uri,
                    "sha256": verifier.AUDIT_CODE_SHA256,
                    "size_bytes": verifier.AUDIT_CODE_SIZE_BYTES,
                },
                "source_f03": {
                    "uri": f03_uri,
                    **verifier.EXPECTED_ARCHIVES["source_f03"],
                },
                "source_f124": {
                    "uri": f124_uri,
                    **verifier.EXPECTED_ARCHIVES["source_f124"],
                },
            },
            "resolved_output_ref": {
                "schema_version": "exp689_immutable_s3_output_ref_v1",
                "s3_prefix": output_prefix,
                "immutable_unique_prefix": True,
                "objects": [
                    {
                        "name": "archive_header_audit.json",
                        "uri": output_prefix + "/archive_header_audit.json",
                        "sha256": report_sha,
                        "size_bytes": report_path.stat().st_size,
                        "etag": "single-part-etag",
                        "version_id": None,
                    }
                ],
            },
            "submit_receipt_sha256": receipt_sha,
            "self_sha256": None,
        }
    )
    metadata_path = tmp_path / "resolved_metadata.json"
    write_json(metadata_path, metadata)
    kwargs: dict[str, object] = {
        "report_path": report_path,
        "expected_report_sha256": report_sha,
        "expected_report_size_bytes": report_path.stat().st_size,
        "submit_receipt_path": receipt_path,
        "expected_submit_receipt_sha256": receipt_sha,
        "resolved_metadata_path": metadata_path,
        "expected_resolved_metadata_sha256": verifier.sha256_file(metadata_path),
        "expected_command_sha256": command_sha,
        "expected_code_uri": code_uri,
        "expected_f03_uri": f03_uri,
        "expected_f124_uri": f124_uri,
        "approved_output_prefix": output_prefix,
        "expected_verifier_sha256": verifier.sha256_file(
            Path(verifier.__file__)
        ),
        "verifier_commit": "c" * 40,
        "output_path": tmp_path / "acceptance.json",
    }
    return kwargs, metadata


def test_compact_verifier_binds_independent_terminal_provenance(
    tmp_path: Path,
) -> None:
    kwargs, _ = make_verification_packet(tmp_path)
    result = verifier.verify(**kwargs)
    assert result["decision"] == "NO_UNSAFE_MEMBERS_EXTRACTOR_MISMATCH"
    assert result["independent_remote_provenance_verified"] is True
    assert result["prepare_retry_authorized"] is False


def test_compact_verifier_rejects_successful_job_input_substitution(
    tmp_path: Path,
) -> None:
    kwargs, metadata = make_verification_packet(tmp_path)
    metadata["resolved_inputs"]["source_f03"]["uri"] = "s3://approved/other.tar.gz"
    metadata = with_self_hash(metadata)
    write_json(kwargs["resolved_metadata_path"], metadata)
    kwargs["expected_resolved_metadata_sha256"] = verifier.sha256_file(
        kwargs["resolved_metadata_path"]
    )
    with pytest.raises(ValueError, match="f03 input.*binding mismatch"):
        verifier.verify(**kwargs)


def test_compact_verifier_rejects_frozen_report_substitution(tmp_path: Path) -> None:
    kwargs, metadata = make_verification_packet(tmp_path)
    report = json.loads(kwargs["report_path"].read_text(encoding="utf-8"))
    report["archives"][0]["size_bytes"] += 1
    report = with_self_hash(report)
    write_json(kwargs["report_path"], report)
    report_sha = verifier.sha256_file(kwargs["report_path"])
    kwargs["expected_report_sha256"] = report_sha
    kwargs["expected_report_size_bytes"] = kwargs["report_path"].stat().st_size
    output_object = metadata["resolved_output_ref"]["objects"][0]
    output_object["sha256"] = report_sha
    output_object["size_bytes"] = kwargs["report_path"].stat().st_size
    metadata = with_self_hash(metadata)
    write_json(kwargs["resolved_metadata_path"], metadata)
    kwargs["expected_resolved_metadata_sha256"] = verifier.sha256_file(
        kwargs["resolved_metadata_path"]
    )
    with pytest.raises(ValueError, match="frozen object binding mismatch"):
        verifier.verify(**kwargs)


def test_compact_verifier_rejects_submit_receipt_substitution(tmp_path: Path) -> None:
    kwargs, _ = make_verification_packet(tmp_path)
    receipt = json.loads(kwargs["submit_receipt_path"].read_text(encoding="utf-8"))
    receipt["job_id"] = "different-successful-job"
    write_json(kwargs["submit_receipt_path"], receipt)
    kwargs["expected_submit_receipt_sha256"] = verifier.sha256_file(
        kwargs["submit_receipt_path"]
    )
    with pytest.raises(ValueError, match="terminal/job identity mismatch"):
        verifier.verify(**kwargs)

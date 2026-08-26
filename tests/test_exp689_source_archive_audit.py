from __future__ import annotations

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


def test_compact_verifier_rejects_frozen_object_substitution(tmp_path: Path) -> None:
    report = {
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
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = verifier.verify(
        report_path=report_path,
        expected_report_sha256=verifier.sha256_file(report_path),
        expected_code_commit="a" * 40,
        expected_code_sha256="b" * 64,
        output_path=tmp_path / "acceptance.json",
    )
    assert result["decision"] == "NO_UNSAFE_MEMBERS_EXTRACTOR_MISMATCH"
    assert result["prepare_retry_authorized"] is False
    report["archives"][0]["size_bytes"] += 1
    report["self_sha256"] = None
    report["self_sha256"] = audit.hashlib.sha256(
        audit.canonical_json_bytes(report)
    ).hexdigest()
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen object binding mismatch"):
        verifier.verify(
            report_path=forged,
            expected_report_sha256=verifier.sha256_file(forged),
            expected_code_commit="a" * 40,
            expected_code_sha256="b" * 64,
            output_path=tmp_path / "forbidden.json",
        )

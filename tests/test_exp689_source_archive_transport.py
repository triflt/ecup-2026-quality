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

import extract_source_archive_transport as transport


def add_regular(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def profile(path: Path, *, members: int, apple: int, types: dict[str, int]) -> dict[str, object]:
    return {
        "sha256": transport.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "member_count": members,
        "type_counts": types,
        "apple_metadata_count": apple,
    }


def test_exact_appledouble_members_are_skipped_not_extracted(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        add_regular(archive, "runtime/fold0/runtime_audit.json", b"accepted")
        add_regular(archive, "runtime/fold0/._runtime_audit.json", b"appledouble")
    destination = tmp_path / "out"
    result = transport.extract(
        archive_path=archive_path,
        destination=destination,
        report_path=tmp_path / "report.json",
        archive_id="test",
        profile=profile(archive_path, members=2, apple=1, types={"regular": 2}),
    )
    assert (destination / "runtime/fold0/runtime_audit.json").read_bytes() == b"accepted"
    assert not (destination / "runtime/fold0/._runtime_audit.json").exists()
    assert result["apple_metadata_skipped_count"] == 1
    assert result["symlinks_extracted"] == 0


@pytest.mark.parametrize("kind", ["symlink", "traversal", "duplicate"])
def test_any_nonapple_unsafe_member_remains_fail_closed(
    tmp_path: Path, kind: str
) -> None:
    archive_path = tmp_path / f"{kind}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        if kind == "symlink":
            info = tarfile.TarInfo("runtime/latest")
            info.type = tarfile.SYMTYPE
            info.linkname = "fold0"
            archive.addfile(info)
            types = {"forbidden": 1}
        elif kind == "traversal":
            add_regular(archive, "../escape.json", b"forbidden")
            types = {"regular": 1}
        else:
            add_regular(archive, "runtime/a.json", b"one")
            add_regular(archive, "runtime/a.json", b"two")
            types = {"regular": 2}
    expected = profile(
        archive_path,
        members=1 if kind != "duplicate" else 2,
        apple=0,
        types=types,
    )
    with pytest.raises(ValueError, match="non-regular|unsafe tar path|duplicate"):
        transport.extract(
            archive_path=archive_path,
            destination=tmp_path / "out",
            report_path=tmp_path / "report.json",
            archive_id="test",
            profile=expected,
        )


def test_profile_mismatch_does_not_weaken_header_gate(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        add_regular(archive, "runtime/a.json", b"safe")
    frozen = profile(archive_path, members=2, apple=0, types={"regular": 1})
    with pytest.raises(ValueError, match="member count mismatch"):
        transport.extract(
            archive_path=archive_path,
            destination=tmp_path / "out",
            report_path=tmp_path / "report.json",
            archive_id="test",
            profile=frozen,
        )


def make_diagnostic_acceptance(path: Path) -> str:
    value = {
        "schema_version": "exp689_source_archive_audit_acceptance_v2",
        "status": "accepted_diagnostic_only",
        "decision": "TRANSPORT_CAUSE_APPLE_METADATA_ONLY",
        "independent_remote_provenance_verified": True,
        "report_sha256": "48e58fe07447544d8ebe1b3013ee6d003914864837b5a87a4c24ba5e6625cc67",
        "report_size_bytes": 21_769,
        "report_self_sha256": "b" * 64,
        "diagnostic_code_commit": "076c009eaee106caba428c699627f864e302d292",
        "diagnostic_code_sha256": "20a55c6e1268881c2281cba8d53efad26b5ad58fdc527d2bbddaae7f5477fc75",
        "diagnostic_preset_sha256": "622370e98f66d1378a5e299d9d10401fd01b0d31f26e5191f3bea36321e59ddb",
        "diagnostic_command_sha256": "5e36767eb7b7744347bf643f14603ffdfc12344bcd09652dbf1ac12b05cf494b",
        "submit_receipt_sha256": "620c88baac32ac5edd8f8e630d861eda91c3f95a7286c50b3bcb5afba047436b",
        "resolved_metadata_sha256": "75eb921999b1256dfa3192617b38f7394a09274f165698d5287fde8f97722140",
        "resolved_metadata_self_sha256": "ef162d3abaf62637611d60df02c027851608138a32a3edb803f4d55768a188f0",
        "verifier_commit": "14d215ed4890903be10712dc18e717fc515d1bac",
        "verifier_sha256": "a8b125dfd232c3cb13afb18ceb6f3789e15958cb0d0b189fdffadee97590b41b",
        "archive_bindings": {
            key: {"sha256": item["sha256"], "size_bytes": item["size_bytes"]}
            for key, item in transport.PROFILES.items()
        },
        "unsafe_reason_set": ["apple_metadata"],
        "prepare_retry_authorized": False,
        "teacher_authorized": False,
        "student_gpu_authorized": False,
        "self_sha256": None,
    }
    value["self_sha256"] = hashlib.sha256(
        transport.canonical_json_bytes(value)
    ).hexdigest()
    path.write_text(json.dumps(value), encoding="utf-8")
    return transport.sha256_file(path)


def test_transport_requires_exact_accepted_diagnostic_lineage(tmp_path: Path) -> None:
    path = tmp_path / "acceptance.json"
    file_sha = make_diagnostic_acceptance(path)
    result = transport.validate_diagnostic_acceptance(path, file_sha)
    assert result["decision"] == "TRANSPORT_CAUSE_APPLE_METADATA_ONLY"
    result["unsafe_reason_set"] = []
    result["self_sha256"] = None
    result["self_sha256"] = hashlib.sha256(
        transport.canonical_json_bytes(result)
    ).hexdigest()
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="lineage/decision mismatch"):
        transport.validate_diagnostic_acceptance(path, transport.sha256_file(path))

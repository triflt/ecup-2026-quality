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

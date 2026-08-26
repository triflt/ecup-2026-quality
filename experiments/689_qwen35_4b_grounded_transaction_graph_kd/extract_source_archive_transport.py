"""Safely extract frozen exp689 source archives after the accepted header audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

PROFILES: dict[str, dict[str, Any]] = {
    "source_f03": {
        "sha256": "e371c03a3fc893d990d38874e07200a5ac136b8c72f43109aa7f567761943ccd",
        "size_bytes": 15_852_324,
        "member_count": 138,
        "type_counts": {"directory": 21, "regular": 117},
        "apple_metadata_count": 69,
    },
    "source_f124": {
        "sha256": "1dfb9bf01a9567286051ee76a79fc4b41c2af360e5761b7a4663090745c5474d",
        "size_bytes": 15_000_958,
        "member_count": 10,
        "type_counts": {"regular": 10},
        "apple_metadata_count": 0,
    },
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ACCEPTANCE_FIELDS = {
    "schema_version",
    "status",
    "decision",
    "independent_remote_provenance_verified",
    "report_sha256",
    "report_size_bytes",
    "report_self_sha256",
    "diagnostic_code_commit",
    "diagnostic_code_sha256",
    "diagnostic_preset_sha256",
    "diagnostic_command_sha256",
    "submit_receipt_sha256",
    "resolved_metadata_sha256",
    "resolved_metadata_self_sha256",
    "verifier_commit",
    "verifier_sha256",
    "archive_bindings",
    "unsafe_reason_set",
    "prepare_retry_authorized",
    "teacher_authorized",
    "student_gpu_authorized",
    "self_sha256",
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


def validate_diagnostic_acceptance(
    path: Path, expected_file_sha256: str
) -> dict[str, Any]:
    if not HEX64.fullmatch(expected_file_sha256):
        raise ValueError("diagnostic acceptance requires an exact file SHA")
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("diagnostic acceptance file SHA mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != ACCEPTANCE_FIELDS:
        raise ValueError("diagnostic acceptance exact schema mismatch")
    copy = dict(value)
    actual_self = copy["self_sha256"]
    copy["self_sha256"] = None
    if (
        not isinstance(actual_self, str)
        or not HEX64.fullmatch(actual_self)
        or hashlib.sha256(canonical_json_bytes(copy)).hexdigest() != actual_self
    ):
        raise ValueError("diagnostic acceptance self-hash mismatch")
    expected_bindings = {
        archive_id: {
            "sha256": profile["sha256"],
            "size_bytes": profile["size_bytes"],
        }
        for archive_id, profile in PROFILES.items()
    }
    if (
        value["schema_version"] != "exp689_source_archive_audit_acceptance_v2"
        or value["status"] != "accepted_diagnostic_only"
        or value["decision"] != "TRANSPORT_CAUSE_APPLE_METADATA_ONLY"
        or value["independent_remote_provenance_verified"] is not True
        or value["report_sha256"]
        != "48e58fe07447544d8ebe1b3013ee6d003914864837b5a87a4c24ba5e6625cc67"
        or value["report_size_bytes"] != 21_769
        or value["diagnostic_code_commit"]
        != "076c009eaee106caba428c699627f864e302d292"
        or value["diagnostic_code_sha256"]
        != "20a55c6e1268881c2281cba8d53efad26b5ad58fdc527d2bbddaae7f5477fc75"
        or value["diagnostic_preset_sha256"]
        != "622370e98f66d1378a5e299d9d10401fd01b0d31f26e5191f3bea36321e59ddb"
        or value["diagnostic_command_sha256"]
        != "5e36767eb7b7744347bf643f14603ffdfc12344bcd09652dbf1ac12b05cf494b"
        or value["submit_receipt_sha256"]
        != "620c88baac32ac5edd8f8e630d861eda91c3f95a7286c50b3bcb5afba047436b"
        or value["resolved_metadata_sha256"]
        != "75eb921999b1256dfa3192617b38f7394a09274f165698d5287fde8f97722140"
        or value["resolved_metadata_self_sha256"]
        != "ef162d3abaf62637611d60df02c027851608138a32a3edb803f4d55768a188f0"
        or value["verifier_commit"]
        != "14d215ed4890903be10712dc18e717fc515d1bac"
        or value["verifier_sha256"]
        != "a8b125dfd232c3cb13afb18ceb6f3789e15958cb0d0b189fdffadee97590b41b"
        or value["archive_bindings"] != expected_bindings
        or value["unsafe_reason_set"] != ["apple_metadata"]
        or value["prepare_retry_authorized"] is not False
        or value["teacher_authorized"] is not False
        or value["student_gpu_authorized"] is not False
        or not isinstance(value["report_self_sha256"], str)
        or not HEX64.fullmatch(value["report_self_sha256"])
    ):
        raise ValueError("diagnostic acceptance frozen lineage/decision mismatch")
    return value


def kind(member: tarfile.TarInfo) -> str:
    if member.isfile():
        return "regular"
    if member.isdir():
        return "directory"
    return "forbidden"


def is_apple_metadata(path: PurePosixPath) -> bool:
    return any(
        part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
        for part in path.parts
    )


def validate_headers(
    members: list[tarfile.TarInfo], profile: dict[str, Any]
) -> tuple[list[tuple[tarfile.TarInfo, str]], list[str]]:
    normalized = [
        PurePosixPath(member.name).as_posix().removeprefix("./")
        for member in members
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate normalized tar member")
    if len(members) != profile["member_count"]:
        raise ValueError("frozen tar member count mismatch")
    type_counts = Counter(kind(member) for member in members)
    if dict(sorted(type_counts.items())) != profile["type_counts"]:
        raise ValueError("frozen tar member-type profile mismatch")
    safe: list[tuple[tarfile.TarInfo, str]] = []
    skipped: list[str] = []
    for member, name in zip(members, normalized, strict=True):
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not name:
            raise ValueError("unsafe tar path")
        member_kind = kind(member)
        apple = is_apple_metadata(path)
        if apple:
            if member_kind != "regular":
                raise ValueError("AppleDouble member is not a regular file")
            skipped.append(name)
            continue
        if member_kind not in {"regular", "directory"}:
            raise ValueError("non-regular tar member is forbidden")
        safe.append((member, name))
    if len(skipped) != profile["apple_metadata_count"]:
        raise ValueError("frozen AppleDouble member count mismatch")
    return safe, skipped


def extract(
    *,
    archive_path: Path,
    destination: Path,
    report_path: Path,
    archive_id: str,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if destination.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite extraction output")
    frozen = PROFILES[archive_id] if profile is None else profile
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ValueError("source archive must be a regular file")
    if archive_path.stat().st_size != frozen["size_bytes"]:
        raise ValueError("frozen source archive size mismatch")
    actual_sha = sha256_file(archive_path)
    if actual_sha != frozen["sha256"]:
        raise ValueError("frozen source archive SHA mismatch")
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        safe, skipped = validate_headers(members, frozen)
        destination.mkdir(parents=True, exist_ok=False)
        extracted_files = 0
        extracted_directories = 0
        for member, name in safe:
            target = destination.joinpath(*PurePosixPath(name).parts)
            if member.isdir():
                if target.exists() and not target.is_dir():
                    raise ValueError("directory collides with extracted file")
                target.mkdir(parents=True, exist_ok=True)
                extracted_directories += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError("regular member collides with existing path")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("regular member payload is unavailable")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1 << 20)
            extracted_files += 1
    report = {
        "schema_version": "exp689_source_archive_transport_extraction_v1",
        "archive_id": archive_id,
        "archive_sha256": actual_sha,
        "archive_size_bytes": archive_path.stat().st_size,
        "member_count": frozen["member_count"],
        "type_counts": frozen["type_counts"],
        "apple_metadata_skipped_count": len(skipped),
        "apple_metadata_skipped_names": skipped,
        "extracted_regular_files": extracted_files,
        "extracted_directories": extracted_directories,
        "symlinks_extracted": 0,
        "hardlinks_extracted": 0,
        "path_traversal_extracted": 0,
        "self_sha256": None,
    }
    report["self_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-id", choices=sorted(PROFILES), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = extract(
        archive_path=args.archive,
        destination=args.destination,
        report_path=args.report,
        archive_id=args.archive_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

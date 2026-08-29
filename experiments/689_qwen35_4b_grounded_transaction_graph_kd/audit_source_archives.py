"""Inspect exp689 source tar headers remotely without extracting file contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


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


def member_type(member: tarfile.TarInfo) -> str:
    if member.isfile():
        return "regular"
    if member.isdir():
        return "directory"
    if member.issym():
        return "symlink"
    if member.islnk():
        return "hardlink"
    if member.isfifo():
        return "fifo"
    if member.ischr():
        return "character_device"
    if member.isblk():
        return "block_device"
    return "other"


def audit_archive(
    path: Path, expected_sha256: str, expected_size: int, archive_id: str
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{archive_id}: archive must be a regular file")
    if path.stat().st_size != expected_size:
        raise ValueError(f"{archive_id}: exact archive size mismatch")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise ValueError(f"{archive_id}: exact archive SHA mismatch")
    with tarfile.open(path) as archive:
        members = archive.getmembers()
    normalized = [
        PurePosixPath(member.name).as_posix().removeprefix("./") for member in members
    ]
    duplicates = {name for name, count in Counter(normalized).items() if count > 1}
    unsafe: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    for member, name in zip(members, normalized, strict=True):
        path_value = PurePosixPath(member.name)
        kind = member_type(member)
        type_counts[kind] += 1
        reasons: list[str] = []
        if path_value.is_absolute():
            reasons.append("absolute_path")
        if ".." in path_value.parts:
            reasons.append("path_traversal")
        if not name:
            reasons.append("empty_normalized_name")
        if name in duplicates:
            reasons.append("duplicate_normalized_name")
        if any(
            part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
            for part in path_value.parts
        ):
            reasons.append("apple_metadata")
        if kind not in {"regular", "directory"}:
            reasons.append("non_regular_or_directory")
        if reasons:
            unsafe.append(
                {
                    "name": member.name,
                    "normalized_name": name,
                    "member_type": kind,
                    "reasons": sorted(set(reasons)),
                }
            )
    return {
        "archive_id": archive_id,
        "sha256": actual_sha,
        "size_bytes": path.stat().st_size,
        "member_count": len(members),
        "type_counts": dict(sorted(type_counts.items())),
        "unsafe_member_count": len(unsafe),
        "unsafe_members": unsafe,
        "file_contents_extracted": False,
    }


def run(
    *,
    source_f03: Path,
    source_f03_sha256: str,
    source_f03_size: int,
    source_f124: Path,
    source_f124_sha256: str,
    source_f124_size: int,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable archive audit")
    report = {
        "schema_version": "exp689_source_archive_header_audit_v1",
        "execution_scope": "remote_cpu_header_only",
        "archives": [
            audit_archive(
                source_f03, source_f03_sha256, source_f03_size, "source_f03"
            ),
            audit_archive(
                source_f124, source_f124_sha256, source_f124_size, "source_f124"
            ),
        ],
        "labels_read": 0,
        "sealed_rows_read": 0,
        "public_used": False,
        "jobs_launched_by_auditor": 0,
        "self_sha256": None,
    }
    report["self_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-f03", type=Path, required=True)
    parser.add_argument("--source-f03-sha256", required=True)
    parser.add_argument("--source-f03-size", type=int, required=True)
    parser.add_argument("--source-f124", type=Path, required=True)
    parser.add_argument("--source-f124-sha256", required=True)
    parser.add_argument("--source-f124-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

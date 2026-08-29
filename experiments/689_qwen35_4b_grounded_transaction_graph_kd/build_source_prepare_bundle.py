"""Build the exact minimal exp689 source-PREPARE bundle from a frozen Git revision."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any

EXPERIMENT = "experiments/689_qwen35_4b_grounded_transaction_graph_kd"
FILES = (
    f"{EXPERIMENT}/prepare_source_universe.py",
    f"{EXPERIMENT}/source_prepare_spec_v1.json",
    f"{EXPERIMENT}/verify_source_prepare.py",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob(repo: Path, revision: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{relative}"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"missing frozen Git blob: {relative}")
    return completed.stdout


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def build(
    repo: Path,
    revision: str,
    output_dir: Path,
    *,
    files: tuple[str, ...] = FILES,
    manifest_schema: str = "exp689_source_prepare_bundle_manifest_v1",
) -> dict[str, Any]:
    if not HEX40.fullmatch(revision):
        raise ValueError("revision must be an exact lowercase Git commit")
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite bundle output")
    repo = repo.resolve(strict=True)
    output_dir.mkdir(parents=True)
    payload_root = output_dir / "payload"
    payload_root.mkdir()

    entries: list[dict[str, Any]] = []
    blobs: dict[str, bytes] = {}
    for relative in files:
        blob = git_blob(repo, revision, relative)
        blobs[relative] = blob
        destination = payload_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
        entries.append(
            {"path": relative, "sha256": sha256_bytes(blob), "size_bytes": len(blob)}
        )
    manifest = {
        "schema_version": manifest_schema,
        "builder_revision": revision,
        "files": entries,
        "self_sha256": None,
    }
    manifest["self_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    manifest_path = output_dir / "bundle_manifest.json"
    write_json(manifest_path, manifest)

    archive_path = output_dir / "source_prepare_bundle.tar.gz"
    compressed = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as gzip_stream,
        tarfile.open(fileobj=gzip_stream, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for relative in files:
            blob = blobs[relative]
            info = tarfile.TarInfo(relative)
            info.size = len(blob)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(blob))
    archive_path.write_bytes(compressed.getvalue())
    report = {
        "schema_version": "exp689_source_prepare_bundle_report_v1",
        "builder_revision": revision,
        "archive_name": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "bundle_manifest_self_sha256": manifest["self_sha256"],
        "files": entries,
        "teacher_code_members": 0,
        "student_code_members": 0,
        "jobs_launched": 0,
        "uploads": 0,
    }
    write_json(output_dir / "bundle_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

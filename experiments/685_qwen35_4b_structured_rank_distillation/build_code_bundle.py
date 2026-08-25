from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from verify_code_bundle import MANIFEST_NAME, canonical_sha256


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def normalized_info(source: tarfile.TarInfo) -> tarfile.TarInfo:
    info = tarfile.TarInfo(PurePosixPath(source.name).as_posix().removeprefix("./"))
    info.size = source.size
    info.mode = source.mode
    info.type = source.type
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def build(repo: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite code bundle")
    if git(repo, "status", "--porcelain"):
        raise ValueError("code bundle requires a clean worktree")
    revision = git(repo, "rev-parse", "HEAD")
    with tempfile.TemporaryDirectory(prefix="exp685_code_bundle_") as directory:
        source_tar = Path(directory) / "source.tar"
        subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", "-o", str(source_tar), "HEAD"],
            check=True,
        )
        members: list[tuple[tarfile.TarInfo, bytes | None]] = []
        files: dict[str, dict[str, Any]] = {}
        directories: set[str] = set()
        with tarfile.open(source_tar) as source:
            names: set[str] = set()
            for member in source.getmembers():
                info = normalized_info(member)
                path = PurePosixPath(info.name)
                if (
                    not info.name
                    or path.is_absolute()
                    or ".." in path.parts
                    or any(
                        part == "__MACOSX"
                        or part == ".DS_Store"
                        or part.startswith("._")
                        for part in path.parts
                    )
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(f"unsafe git-archive member: {member.name}")
                if info.name in names:
                    raise ValueError("duplicate git-archive member")
                names.add(info.name)
                payload: bytes | None = None
                if member.isfile():
                    extracted = source.extractfile(member)
                    if extracted is None:
                        raise ValueError("git-archive file is unreadable")
                    payload = extracted.read()
                    files[info.name] = {
                        "size": len(payload),
                        "sha256": sha256_bytes(payload),
                    }
                else:
                    directories.add(info.name.rstrip("/"))
                members.append((info, payload))

        manifest = {
            "schema_version": 1,
            "experiment_id": "685",
            "git_revision": revision,
            "files": files,
            "directories": sorted(directories),
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        manifest_payload = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        manifest_info = tarfile.TarInfo(MANIFEST_NAME)
        manifest_info.size = len(manifest_payload)
        manifest_info.mode = 0o644
        manifest_info.mtime = 0
        output.parent.mkdir(parents=True, exist_ok=True)
        with (
            output.open("xb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as destination,
        ):
            for info, payload in members:
                destination.addfile(
                    info,
                    io.BytesIO(payload) if payload is not None else None,
                )
            destination.addfile(manifest_info, io.BytesIO(manifest_payload))
    return {
        "git_revision": revision,
        "files": len(files),
        "directories": len(directories),
        "manifest_sha256": manifest["manifest_sha256"],
        "bundle_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bundle_size": output.stat().st_size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.output), indent=2, sort_keys=True))

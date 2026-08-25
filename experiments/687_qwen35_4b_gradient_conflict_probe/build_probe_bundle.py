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

from verify_probe_code_bundle import MANIFEST_NAME, canonical_sha256

SOURCE_PATH = "experiments/687_qwen35_4b_gradient_conflict_probe"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def build(repo: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite probe bundle")
    if git(repo, "status", "--porcelain"):
        raise ValueError("probe bundle requires a clean worktree")
    revision = git(repo, "rev-parse", "HEAD")
    with tempfile.TemporaryDirectory(prefix="exp687_bundle_") as directory:
        source_tar = Path(directory) / "source.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                "--format=tar",
                "-o",
                str(source_tar),
                "HEAD",
                SOURCE_PATH,
            ],
            check=True,
        )
        members: list[tuple[tarfile.TarInfo, bytes | None]] = []
        files: dict[str, dict[str, Any]] = {}
        directories: list[str] = []
        with tarfile.open(source_tar) as source:
            names: set[str] = set()
            for member in source.getmembers():
                name = PurePosixPath(member.name).as_posix().removeprefix("./")
                path = PurePosixPath(name)
                if (
                    not name
                    or path.is_absolute()
                    or ".." in path.parts
                    or not name.startswith(f"{SOURCE_PATH}/")
                    or any(
                        part == "__MACOSX"
                        or part == ".DS_Store"
                        or part.startswith("._")
                        for part in path.parts
                    )
                    or not (member.isfile() or member.isdir())
                    or name in names
                ):
                    raise ValueError(f"unsafe probe-bundle member: {member.name}")
                names.add(name)
                info = tarfile.TarInfo(name)
                info.size = member.size
                info.mode = member.mode
                info.type = member.type
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                payload: bytes | None = None
                if member.isfile():
                    stream = source.extractfile(member)
                    if stream is None:
                        raise ValueError("probe-bundle file is unreadable")
                    payload = stream.read()
                    files[name] = {
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                else:
                    directories.append(name.rstrip("/"))
                members.append((info, payload))

        manifest = {
            "schema_version": 1,
            "experiment_id": "687",
            "git_revision": revision,
            "source_path": SOURCE_PATH,
            "files": files,
            "directories": sorted(directories),
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
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
                destination.addfile(info, io.BytesIO(payload) if payload is not None else None)
            destination.addfile(manifest_info, io.BytesIO(manifest_payload))
    return {
        "git_revision": revision,
        "manifest_sha256": manifest["manifest_sha256"],
        "bundle_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bundle_size": output.stat().st_size,
        "files": len(files),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2, sort_keys=True))

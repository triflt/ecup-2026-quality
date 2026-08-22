from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    root = output_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("scoped runtime archive is empty")
        for member in members:
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsupported archive member: {member.name}")
        archive.extractall(root, members=members, filter="data")


def prepare_runtime(*, url: str, expected_sha256: str, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    with tempfile.TemporaryDirectory(prefix="scoped-runtime-") as temporary:
        archive_path = Path(temporary) / "runtime.tar.gz"
        with urllib.request.urlopen(url, timeout=300) as response, archive_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        actual_sha256 = sha256(archive_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "scoped runtime checksum mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        safe_extract(archive_path, output_dir)
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError("scoped runtime extraction produced no files")
    return {
        "archive_sha256": actual_sha256,
        "runtime_files": len(files),
        "runtime_bytes": sum(path.stat().st_size for path in files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-env", default="SCOPED_RUNTIME_URL")
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--bootstrap-runtime-dir", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    args, entrypoint_args = parser.parse_known_args()
    url = os.environ.get(args.url_env, "").strip()
    if not url:
        raise ValueError(f"missing runtime URL in {args.url_env}")
    audit = prepare_runtime(
        url=url,
        expected_sha256=args.archive_sha256,
        output_dir=args.bootstrap_runtime_dir,
    )
    print(audit, flush=True)
    entrypoint_args = list(entrypoint_args)
    if entrypoint_args[:1] == ["--"]:
        entrypoint_args = entrypoint_args[1:]
    subprocess.run(
        [sys.executable, str(args.entrypoint), *entrypoint_args],
        check=True,
        env=os.environ.copy(),
    )


if __name__ == "__main__":
    main()

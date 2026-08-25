from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "CODE_BUNDLE_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify(
    root: Path,
    *,
    expected_revision: str,
    archive: Path | None = None,
    expected_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    if (archive is None) != (expected_bundle_sha256 is None):
        raise ValueError("archive and expected bundle SHA must be provided together")
    if archive is not None and sha256_file(archive) != expected_bundle_sha256:
        raise ValueError("code-bundle archive SHA mismatch")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError("code-bundle manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = dict(manifest)
    digest = body.pop("manifest_sha256", None)
    if digest != canonical_sha256(body):
        raise ValueError("code-bundle manifest self-hash mismatch")
    if manifest.get("git_revision") != expected_revision:
        raise ValueError("code-bundle revision differs from the frozen preset")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("code-bundle file manifest is empty")
    expected_directories = manifest.get("directories")
    if not isinstance(expected_directories, list) or any(
        not isinstance(value, str) or not value for value in expected_directories
    ):
        raise ValueError("code-bundle directory manifest is invalid")

    observed: dict[str, dict[str, Any]] = {}
    observed_directories: list[str] = []
    resolved_root = root.resolve()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError("code bundle contains a symlink")
        if any(
            part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
            for part in relative.parts
        ):
            raise ValueError("code bundle contains transport metadata")
        if not path.resolve().is_relative_to(resolved_root):
            raise ValueError("code-bundle member escapes its root")
        if path.is_file() and relative.as_posix() != MANIFEST_NAME:
            observed[relative.as_posix()] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        elif path.is_dir():
            observed_directories.append(relative.as_posix())
        elif not path.is_file() and not path.is_dir():
            raise ValueError("code bundle contains a special filesystem entry")
    if observed != expected_files:
        raise ValueError("code-bundle member set or payload hash mismatch")
    if observed_directories != expected_directories:
        raise ValueError("code-bundle directory set mismatch")
    result = {
        "schema_version": 1,
        "experiment_id": "685",
        "git_revision": expected_revision,
        "bundle_sha256": expected_bundle_sha256,
        "manifest_sha256": digest,
        "files": len(observed),
        "directories": len(observed_directories),
        "decision": "ACCEPT_CODE_BUNDLE",
    }
    result["acceptance_sha256"] = canonical_sha256(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.root,
        expected_revision=args.expected_revision,
        archive=args.archive,
        expected_bundle_sha256=args.expected_bundle_sha256,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        raise FileExistsError("refusing to overwrite code-bundle acceptance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")

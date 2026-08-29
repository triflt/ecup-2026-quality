from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "EXP687_CODE_BUNDLE_MANIFEST.json"
SOURCE_PATH = "experiments/687_qwen35_4b_gradient_conflict_probe"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(
    root: Path,
    archive: Path,
    *,
    expected_revision: str,
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    if sha256_file(archive) != expected_bundle_sha256:
        raise ValueError("probe-bundle archive SHA mismatch")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError("probe-bundle manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = dict(manifest)
    digest = body.pop("manifest_sha256", None)
    if digest != canonical_sha256(body):
        raise ValueError("probe-bundle manifest self-hash mismatch")
    if (
        manifest.get("experiment_id") != "687"
        or manifest.get("git_revision") != expected_revision
        or manifest.get("source_path") != SOURCE_PATH
    ):
        raise ValueError("probe-bundle provenance mismatch")
    expected_files = manifest.get("files")
    expected_directories = manifest.get("directories")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("probe-bundle file manifest is empty")
    if not isinstance(expected_directories, list):
        raise TypeError("probe-bundle directory manifest is invalid")

    observed: dict[str, dict[str, Any]] = {}
    observed_directories: list[str] = []
    resolved_root = root.resolve()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(resolved_root):
            raise ValueError("probe-bundle member escapes isolated root")
        if any(
            part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
            for part in Path(relative).parts
        ):
            raise ValueError("probe bundle contains transport metadata")
        if path.is_file() and relative != MANIFEST_NAME:
            if not relative.startswith(f"{SOURCE_PATH}/"):
                raise ValueError("probe bundle contains a file outside its experiment")
            observed[relative] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        elif path.is_dir():
            if not relative.startswith("experiments"):
                raise ValueError("probe bundle contains an unexpected directory")
            observed_directories.append(relative)
        elif not path.is_file():
            raise ValueError("probe bundle contains a special filesystem entry")
    if observed != expected_files or observed_directories != expected_directories:
        raise ValueError("probe-bundle members differ from manifest")
    result = {
        "schema_version": 1,
        "experiment_id": "687",
        "git_revision": expected_revision,
        "bundle_sha256": expected_bundle_sha256,
        "manifest_sha256": digest,
        "files": len(observed),
        "decision": "ACCEPT_PROBE_CODE_BUNDLE",
    }
    result["acceptance_sha256"] = canonical_sha256(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(
        args.root,
        args.archive,
        expected_revision=args.expected_revision,
        expected_bundle_sha256=args.expected_bundle_sha256,
    )
    if args.output.exists():
        raise FileExistsError("refusing to overwrite probe-code acceptance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(value), flush=True)

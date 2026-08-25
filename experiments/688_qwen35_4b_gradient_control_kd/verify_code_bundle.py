from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "EXP688_CODE_BUNDLE_MANIFEST.json"
SCOPE = "paired_technical_smoke"
SOURCE_PATHS = [
    "experiments/688_qwen35_4b_gradient_control_kd/train_gradient_control.py",
    "experiments/688_qwen35_4b_gradient_control_kd/verify_training_artifact.py",
    "experiments/688_qwen35_4b_gradient_control_kd/verify_paired_smoke.py",
    "experiments/688_qwen35_4b_gradient_control_kd/verify_code_bundle.py",
]
EXPECTED_DIRECTORIES = [
    "experiments",
    "experiments/688_qwen35_4b_gradient_control_kd",
]


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
    if not archive.is_file() or archive.is_symlink():
        raise ValueError("exp688 code archive must be a regular non-symlink file")
    if sha256_file(archive) != expected_bundle_sha256:
        raise ValueError("exp688 code-bundle archive SHA mismatch")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("exp688 code-bundle manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = dict(manifest)
    manifest_sha = body.pop("manifest_sha256", None)
    if manifest_sha != canonical_sha256(body):
        raise ValueError("exp688 code-bundle manifest self-hash mismatch")
    expected_identity = {
        "schema_version": 1,
        "experiment_id": "688",
        "parent_experiment_id": "686",
        "selector_experiment_id": "687",
        "scope": SCOPE,
        "git_revision": expected_revision,
        "source_paths": SOURCE_PATHS,
    }
    if any(manifest.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("exp688 code-bundle identity or whitelist mismatch")
    expected_files = manifest.get("files")
    expected_directories = manifest.get("directories")
    if not isinstance(expected_files, dict) or set(expected_files) != set(SOURCE_PATHS):
        raise ValueError("exp688 code-bundle file manifest differs from exact whitelist")
    if expected_directories != EXPECTED_DIRECTORIES:
        raise ValueError("exp688 code-bundle directory whitelist mismatch")

    observed: dict[str, dict[str, Any]] = {}
    observed_directories: list[str] = []
    resolved_root = root.resolve()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(resolved_root):
            raise ValueError("exp688 code-bundle member escapes isolated root")
        if any(
            part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
            for part in Path(relative).parts
        ):
            raise ValueError("exp688 code bundle contains transport metadata")
        if path.is_file() and relative != MANIFEST_NAME:
            if relative not in SOURCE_PATHS:
                raise ValueError("exp688 code bundle contains an unknown file")
            observed[relative] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        elif path.is_dir():
            observed_directories.append(relative)
        elif not path.is_file():
            raise ValueError("exp688 code bundle contains a special filesystem entry")
    if observed != expected_files or observed_directories != expected_directories:
        raise ValueError("exp688 code-bundle members differ from manifest")
    result = {
        "schema_version": 1,
        "experiment_id": "688",
        "parent_experiment_id": "686",
        "selector_experiment_id": "687",
        "scope": SCOPE,
        "git_revision": expected_revision,
        "bundle_sha256": expected_bundle_sha256,
        "manifest_sha256": manifest_sha,
        "files": len(observed),
        "source_paths": SOURCE_PATHS,
        "decision": "ACCEPT_EXP688_CODE_BUNDLE",
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
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        raise FileExistsError("refusing to overwrite exp688 code acceptance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")

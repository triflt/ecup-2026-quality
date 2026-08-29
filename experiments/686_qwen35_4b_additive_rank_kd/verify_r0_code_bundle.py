from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "R0_CODE_BUNDLE_MANIFEST.json"
R0_SOURCE_PATHS = [
    "experiments/686_qwen35_4b_additive_rank_kd/build_pair_runtime.py",
    "experiments/686_qwen35_4b_additive_rank_kd/verify_pair_runtime.py",
    "experiments/686_qwen35_4b_additive_rank_kd/verify_r0_code_bundle.py",
    "experiments/680_qwen35_4b_flammable_only_hard_bce/build_runtime.py",
    "experiments/681_qwen35_4b_flammable_logit_distillation/build_runtime.py",
    "experiments/662_qwen36_27b_outer_train_scoring/results/full_target_set_acceptance.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
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
        raise ValueError("R0 code-bundle archive SHA mismatch")

    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("R0 code-bundle manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = dict(manifest)
    digest = body.pop("manifest_sha256", None)
    if digest != canonical_sha256(body):
        raise ValueError("R0 code-bundle manifest self-hash mismatch")
    expected_header = {
        "schema_version": 1,
        "experiment_id": "686",
        "scope": "r0_prepare",
        "git_revision": expected_revision,
        "source_paths": R0_SOURCE_PATHS,
    }
    if any(manifest.get(key) != value for key, value in expected_header.items()):
        raise ValueError("R0 code-bundle frozen header mismatch")

    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or sorted(expected_files) != sorted(
        R0_SOURCE_PATHS
    ):
        raise ValueError("R0 code-bundle file whitelist mismatch")
    expected_directories = manifest.get("directories")
    if not isinstance(expected_directories, list):
        raise TypeError("R0 code-bundle directory manifest is invalid")

    observed: dict[str, dict[str, Any]] = {}
    observed_directories: list[str] = []
    resolved_root = root.resolve()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError("R0 code bundle contains a symlink")
        if any(
            part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
            for part in relative.parts
        ):
            raise ValueError("R0 code bundle contains transport metadata")
        if not path.resolve().is_relative_to(resolved_root):
            raise ValueError("R0 code-bundle member escapes its root")
        if path.is_file() and relative.as_posix() != MANIFEST_NAME:
            observed[relative.as_posix()] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        elif path.is_dir():
            observed_directories.append(relative.as_posix())
        elif not path.is_file():
            raise ValueError("R0 code bundle contains a special filesystem entry")
    if observed != expected_files:
        raise ValueError("R0 code-bundle member set or payload hash mismatch")
    if observed_directories != expected_directories:
        raise ValueError("R0 code-bundle directory set mismatch")

    result = {
        "schema_version": 1,
        "experiment_id": "686",
        "scope": "r0_prepare",
        "git_revision": expected_revision,
        "bundle_sha256": expected_bundle_sha256,
        "manifest_sha256": digest,
        "files": len(observed),
        "directories": len(observed_directories),
        "source_paths": R0_SOURCE_PATHS,
        "decision": "ACCEPT_R0_CODE_BUNDLE",
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
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        raise FileExistsError("refusing to overwrite R0 code-bundle acceptance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")

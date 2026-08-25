from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "CODE_BUNDLE_MANIFEST.json"
TRAINING_SOURCE_PATHS = [
    "experiments/686_qwen35_4b_additive_rank_kd",
    "experiments/645_qwen_scale_2x3_gate/grid_contract.py",
    "experiments/645_qwen_scale_2x3_gate/train_lora.py",
]
EVALUATION_ONLY_SOURCE_PATHS = [
    "experiments/680_qwen35_4b_flammable_only_hard_bce/evaluate.py",
    "experiments/635_span_head_full140_integration/evaluate.py",
    "experiments/635_span_head_full140_integration/frozen_spec.json",
    "experiments/140_dual_lora_fusion/submission/run.py",
    "experiments/632_span_head_seed_repeat/run_fold.py",
    "experiments/632_span_head_seed_repeat/evaluate_screen.py",
    "experiments/632_span_head_seed_repeat/evaluate_full.py",
    "experiments/623_semantic_v3_multitask_span_head/frozen_spec.json",
    "experiments/623_semantic_v3_multitask_span_head/renderer.py",
    "experiments/623_semantic_v3_multitask_span_head/protocol.py",
    "experiments/662_qwen36_27b_outer_train_scoring/results/full_target_set_acceptance.json",
    "validation/semantic_family_v3/folds.csv",
]


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
    expected_scope: str,
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
    if manifest.get("experiment_id") != "686" or manifest.get("scope") not in {
        "training",
        "evaluation",
    }:
        raise ValueError("code-bundle experiment or scope mismatch")
    if manifest["scope"] != expected_scope:
        raise ValueError("code-bundle scope differs from the consuming stage")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("code-bundle file manifest is empty")
    source_paths = manifest.get("source_paths")
    if not isinstance(source_paths, list) or any(
        not isinstance(value, str) or not value for value in source_paths
    ):
        raise ValueError("code-bundle source-path whitelist is invalid")
    expected_source_paths = (
        TRAINING_SOURCE_PATHS
        if manifest["scope"] == "training"
        else TRAINING_SOURCE_PATHS + EVALUATION_ONLY_SOURCE_PATHS
    )
    if source_paths != expected_source_paths:
        raise ValueError("code-bundle source-path whitelist differs from frozen scope")
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
    if manifest["scope"] == "training" and any(
        relative == "validation/semantic_family_v3/folds.csv"
        or relative.startswith(
            (
                "experiments/140_dual_lora_fusion/submission/",
                "experiments/635_span_head_full140_integration/",
                "experiments/632_span_head_seed_repeat/",
                "experiments/623_semantic_v3_multitask_span_head/",
            )
        )
        for relative in observed
    ):
        raise ValueError("training code bundle contains evaluation labels or replay code")
    if observed_directories != expected_directories:
        raise ValueError("code-bundle directory set mismatch")
    result = {
        "schema_version": 1,
        "experiment_id": "686",
        "scope": manifest.get("scope"),
        "git_revision": expected_revision,
        "bundle_sha256": expected_bundle_sha256,
        "manifest_sha256": digest,
        "files": len(observed),
        "directories": len(observed_directories),
        "source_paths": source_paths,
        "decision": "ACCEPT_CODE_BUNDLE",
    }
    result["acceptance_sha256"] = canonical_sha256(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-scope", choices=("training", "evaluation"), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.root,
        expected_revision=args.expected_revision,
        expected_scope=args.expected_scope,
        archive=args.archive,
        expected_bundle_sha256=args.expected_bundle_sha256,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        raise FileExistsError("refusing to overwrite code-bundle acceptance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")

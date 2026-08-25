from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "685"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode())


def parse_fold_map(values: list[str], label: str) -> dict[int, str]:
    output: dict[int, str] = {}
    for value in values:
        fold_text, separator, payload = value.partition("=")
        if not separator or not fold_text.isdigit() or not payload:
            raise ValueError(f"{label} must use FOLD=VALUE")
        fold = int(fold_text)
        if fold in output:
            raise ValueError(f"duplicate {label} fold")
        output[fold] = payload
    if not output:
        raise ValueError(f"at least one {label} is required")
    return output


def safe_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"legacy artifact input is not a directory: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("legacy artifact contains a symlink")
        if path.is_file():
            files.append(path)
    if not files:
        raise ValueError("legacy artifact input is empty")
    return files


def teacher_score_payload(
    files: list[Path], expected_archive_sha256: str
) -> tuple[str, str, bytes]:
    inner = [path for path in files if path.name == "teacher_outer_train_scores.zip"]
    if len(inner) != 1:
        raise ValueError("legacy artifact must contain one canonical inner archive")
    archive_path = inner[0]
    archive_sha = sha256_file(archive_path)
    if archive_sha != expected_archive_sha256:
        raise ValueError("legacy canonical inner archive SHA mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("legacy teacher archive is corrupt")
        names = archive.namelist()
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("legacy teacher archive contains an unsafe member")
        score_names = [name for name in names if Path(name).name == "teacher_scores.jsonl"]
        if len(score_names) != 1:
            raise ValueError("canonical inner archive must contain one teacher-score payload")
        score_name = score_names[0]
        return archive_path.name, archive_sha, archive.read(score_name)


def bridge(
    fold_inputs: dict[int, Path],
    expected_archive_sha256: dict[int, str],
    expected_score_sha256: dict[int, str],
    output: Path,
) -> dict[str, Any]:
    if not set(fold_inputs) == set(expected_archive_sha256) == set(expected_score_sha256):
        raise ValueError("input and expected-hash folds differ")
    if output.exists():
        raise FileExistsError("refusing to overwrite bridge output")
    verified: dict[int, dict[str, Any]] = {}
    for fold in sorted(fold_inputs):
        root = fold_inputs[fold]
        files = safe_files(root)
        score_source, archive_sha, score_payload = teacher_score_payload(
            files, expected_archive_sha256[fold]
        )
        score_sha = sha256_bytes(score_payload)
        if score_sha != expected_score_sha256[fold]:
            raise ValueError(f"fold {fold} teacher-score SHA mismatch")
        verified[fold] = {
            "teacher_score_source": score_source,
            "teacher_inner_archive_sha256": archive_sha,
            "teacher_scores_sha256": score_sha,
            "source_files": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in files
            ],
        }
    output.mkdir(parents=True)
    for fold in sorted(fold_inputs):
        shutil.copytree(fold_inputs[fold], output / f"fold{fold}")
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": "LEGACY_662_ARTIFACT_TO_APPROVED_S3",
        "folds": sorted(fold_inputs),
        "verified": {str(fold): verified[fold] for fold in sorted(verified)},
        "labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "decision": "ACCEPT_REMOTE_BRIDGE",
    }
    report["bridge_sha256"] = canonical_sha256(report)
    (output / "bridge_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-input", action="append", required=True)
    parser.add_argument("--expected-archive-sha", action="append", required=True)
    parser.add_argument("--expected-score-sha", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_values = parse_fold_map(args.fold_input, "fold input")
    archive_values = parse_fold_map(args.expected_archive_sha, "expected archive SHA")
    sha_values = parse_fold_map(args.expected_score_sha, "expected score SHA")
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in [*archive_values.values(), *sha_values.values()]
    ):
        raise ValueError("expected score SHA must be 64 lowercase hex characters")
    result = bridge(
        {fold: Path(path) for fold, path in input_values.items()},
        archive_values,
        sha_values,
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

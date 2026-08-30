from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODES = ("gold_control", "hardneg_candidate", "rank_candidate")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path, tier: str) -> dict:
    artifacts = {}
    for mode in MODES:
        archive = root / f"{mode}-submit.zip"
        report_path = root / f"{mode}-package.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": "exp716_submission_package_v1",
            "selected_mode": mode,
            "output_sha256": sha256(archive),
            "output_bytes": archive.stat().st_size,
            "teacher_in_submission": False,
            "base_model_in_submission": False,
            "under_5_gib": True,
            "zip_integrity": "PASS",
        }
        mismatch = {
            key: {"expected": value, "actual": report.get(key)}
            for key, value in expected.items()
            if report.get(key) != value
        }
        if mismatch:
            raise ValueError(f"{tier} {mode} package mismatch: {mismatch}")
        artifacts[mode] = {
            "submission": str(archive),
            "submission_sha256": expected["output_sha256"],
            "submission_bytes": expected["output_bytes"],
            "package_report_sha256": sha256(report_path),
        }
    return {
        "schema_version": "exp704_three_arm_manifest_v1",
        "experiment_id": "704",
        "tier": tier,
        "arms": artifacts,
        "teacher_in_submission": False,
        "base_model_in_submission": False,
        "under_5_gib": True,
        "decision": "THREE_ARM_SUBMISSIONS_READY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tier", choices=("provisional_fold3", "full_data"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable manifest")
    result = build_manifest(args.root, args.tier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

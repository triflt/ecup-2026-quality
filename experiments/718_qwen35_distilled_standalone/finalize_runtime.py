from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bound_smoke(path: Path, *, rows: int, submission_sha256: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "exp718_runtime_smoke_v1",
        "experiment_id": "718",
        "architecture": "qwen35_only",
        "rows": rows,
        "submission_sha256": submission_sha256,
        "return_code": 0,
        "cuda_visible_devices": "0",
        "output_schema_valid": True,
        "unique_ids_valid": True,
        "decision": "RUNTIME_SMOKE_PASS",
    }
    mismatch = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatch:
        raise ValueError(f"runtime smoke contract mismatch for {path}: {mismatch}")
    if int(report.get("peak_gpu_memory_mib", 0)) <= 0:
        raise ValueError(f"runtime smoke lacks measured GPU memory: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--package-report", type=Path, required=True)
    parser.add_argument("--smoke-3", type=Path, required=True)
    parser.add_argument("--smoke-600", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite runtime acceptance")
    submission_sha = sha256(args.submission)
    package = json.loads(args.package_report.read_text(encoding="utf-8"))
    expected_package = {
        "schema_version": "exp718_standalone_package_v1",
        "experiment_id": "718",
        "architecture": "qwen35_only",
        "extra_model_stages": [],
        "output_sha256": submission_sha,
        "under_5_gib": True,
        "teacher_in_submission": False,
        "base_model_in_submission": False,
        "zip_integrity": "PASS",
    }
    mismatch = {
        key: {"expected": value, "actual": package.get(key)}
        for key, value in expected_package.items()
        if package.get(key) != value
    }
    if mismatch:
        raise ValueError(f"package contract mismatch: {mismatch}")
    smoke3 = load_bound_smoke(args.smoke_3, rows=3, submission_sha256=submission_sha)
    smoke600 = load_bound_smoke(args.smoke_600, rows=600, submission_sha256=submission_sha)
    required_projection = {
        "projected_public_minutes_1600",
        "projected_private_minutes_3800",
    }
    if not required_projection.issubset(smoke600):
        raise ValueError("600-row smoke lacks runtime projections")
    result = {
        "schema_version": "exp718_runtime_acceptance_v1",
        "experiment_id": "718",
        "architecture": "qwen35_only",
        "decision": "DEPLOYABLE_RUNTIME_PASS",
        "submission_sha256": submission_sha,
        "submission_bytes": args.submission.stat().st_size,
        "package_report_sha256": sha256(args.package_report),
        "smoke_3_report_sha256": sha256(args.smoke_3),
        "smoke_600_report_sha256": sha256(args.smoke_600),
        "single_h100_visible": True,
        "peak_gpu_memory_mib": max(
            int(smoke3["peak_gpu_memory_mib"]), int(smoke600["peak_gpu_memory_mib"])
        ),
        "measured_rows": 600,
        "measured_seconds": float(smoke600["elapsed_seconds"]),
        "projected_public_minutes_1600": float(
            smoke600["projected_public_minutes_1600"]
        ),
        "projected_private_minutes_3800": float(
            smoke600["projected_private_minutes_3800"]
        ),
        "teacher_in_submission": False,
        "model_mounts": ["Qwen/Qwen3.5-4B"],
        "under_5_gib": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

KEY_FIELDS = ("global_index", "id", "category", "fold", "occurrence_index")
OUTPUT_FIELDS = {*KEY_FIELDS, "score"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def verify(archive_path: Path, runtime_dir: Path) -> dict[str, Any]:
    runtime = json.loads((runtime_dir / "runtime_audit.json").read_text(encoding="utf-8"))
    runtime_payload = dict(runtime)
    runtime_digest = runtime_payload.pop("contract_sha256", None)
    if runtime_digest != canonical_sha256(runtime_payload):
        raise ValueError("runtime self-hash mismatch")
    input_rows = [
        json.loads(line)
        for line in (runtime_dir / "score_input.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if sha256_bytes((runtime_dir / "score_input.jsonl").read_bytes()) != runtime["score_input_sha256"]:
        raise ValueError("runtime input SHA mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP member: {bad}")
        names = archive.namelist()
        if set(names) != {"report.json", "teacher_scores.jsonl"}:
            raise ValueError("artifact schema mismatch")
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("unsafe ZIP member")
        report = json.loads(archive.read("report.json"))
        score_payload = archive.read("teacher_scores.jsonl")
    report_payload = dict(report)
    report_digest = report_payload.pop("report_contract_sha256", None)
    if report_digest != canonical_sha256(report_payload):
        raise ValueError("report self-hash mismatch")
    expected_report = {
        "experiment_id": "662",
        "source_experiment_id": "654",
        "outer_fold": runtime["outer_fold"],
        "mode": runtime["mode"],
        "rows": runtime["rows"],
        "score_semantics": "raw_last_token_logit_1_minus_logit_0",
        "source_train_sha256": runtime["source_train_sha256"],
        "runtime_contract_sha256": runtime_digest,
        "teacher_adapter_manifest_sha256": runtime["teacher_adapter_manifest_sha256"],
        "ordered_occurrence_key_sha256": runtime["ordered_occurrence_key_sha256"],
        "validation_rows_read": 0,
        "validation_labels_read": 0,
        "labels_read": 0,
        "sealed_rows": 0,
        "public_used": False,
        "cpu_or_disk_offload": False,
        "decision": "READY_FOR_FAIL_CLOSED_ACCEPTANCE",
    }
    if any(report.get(key) != value for key, value in expected_report.items()):
        raise ValueError("report contract mismatch")
    scores = [json.loads(line) for line in score_payload.decode().splitlines()]
    if len(scores) != len(input_rows) or len(scores) != int(runtime["rows"]):
        raise ValueError("score row count mismatch")
    if any(set(row) != OUTPUT_FIELDS for row in scores):
        raise ValueError("score schema mismatch")
    input_keys = [[row[key] for key in KEY_FIELDS] for row in input_rows]
    score_keys = [[row[key] for key in KEY_FIELDS] for row in scores]
    if score_keys != input_keys:
        raise ValueError("score rows differ from immutable occurrence order")
    if canonical_sha256(score_keys) != runtime["ordered_occurrence_key_sha256"]:
        raise ValueError("occurrence key SHA mismatch")
    if any(not math.isfinite(float(row["score"])) for row in scores):
        raise ValueError("non-finite teacher score")
    if report.get("teacher_scores_sha256") != sha256_bytes(score_payload):
        raise ValueError("teacher score SHA mismatch")
    return {
        "schema_version": 1,
        "experiment_id": "662",
        "outer_fold": runtime["outer_fold"],
        "mode": runtime["mode"],
        "rows": len(scores),
        "finite_scores": len(scores),
        "archive_sha256": sha256_bytes(archive_path.read_bytes()),
        "teacher_scores_sha256": sha256_bytes(score_payload),
        "runtime_contract_sha256": runtime_digest,
        "report_contract_sha256": report_digest,
        "exact_ordered_occurrence_binding": True,
        "validation_rows_read": 0,
        "validation_labels_read": 0,
        "decision": "ACCEPT_ARTIFACT",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--archive", type=Path, required=True)
    result.add_argument("--runtime-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(verify(args.archive, args.runtime_dir), indent=2, sort_keys=True))

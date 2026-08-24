from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(gate_path: Path, report_path: Path, runtime_dir: Path) -> dict[str, Any]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    expected = (
        gate.get("experiment_id") == "680"
        and gate.get("decision") == "OPEN_SINGLE_REFIT"
        and gate.get("training_lane_open") is True
        and gate.get("allowed_jobs") == 1
        and gate.get("model_id") == "Qwen/Qwen3.5-4B"
        and gate.get("changed_factor") == "remove_bad_training_occurrences"
        and gate.get("learning_rate") == 0.0002
        and gate.get("expected_train_occurrences") == 2590
        and gate.get("expected_optimizer_steps") == 162
        and gate.get("threshold") == 0.0
        and gate.get("threshold_tuned") is False
        and gate.get("public_used") is False
        and gate.get("sealed_rows") == 0
        and gate.get("uses_27b") is False
    )
    if not expected:
        raise ValueError("experiment-680 refit gate is closed or inconsistent")
    if sha256_file(report_path) != gate.get("full_report_sha256"):
        raise ValueError("full report checksum differs from frozen refit gate")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not (
        report.get("experiment_id") == "680"
        and report.get("stage") == "full"
        and report.get("passed") is True
        and report.get("decision") == "ACCEPT_FOR_REFIT"
        and report.get("public_used") is False
        and report.get("sealed_rows") == 0
        and all(report.get("gates", {}).values())
    ):
        raise ValueError("full report does not authorize refit")
    audit_path = runtime_dir / "runtime_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("contract_sha256") != gate.get("runtime_contract_sha256"):
        raise ValueError("full runtime contract mismatch")
    for name in ("train.jsonl", "validation.jsonl"):
        if sha256_file(runtime_dir / name) != gate["runtime_payload_sha256"][name]:
            raise ValueError(f"full runtime payload mismatch: {name}")
    if (
        audit.get("experiment_id") != "680"
        or audit.get("train_occurrences") != 2590
        or audit.get("validation_rows") != 0
        or audit.get("bad_train_occurrences") != 0
        or audit.get("uses_27b") is not False
        or audit.get("decision") != "GO_SINGLE_REFIT"
    ):
        raise ValueError("full runtime semantic contract mismatch")
    return gate


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--gate", type=Path, required=True)
    result.add_argument("--full-report", type=Path, required=True)
    result.add_argument("--runtime-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            verify(args.gate, args.full_report, args.runtime_dir),
            indent=2,
            sort_keys=True,
        )
    )

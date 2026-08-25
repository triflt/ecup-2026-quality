from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "686_qwen35_4b_additive_rank_kd"
for path in (HERE, PARENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_pair_runtime import canonical_sha256
from gradient_metrics import CHECKPOINT_STEPS, DIAGNOSTIC_EFFECTIVE_BATCHES, pcgrad_gate


def verify(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    body = dict(report)
    digest = body.pop("report_sha256", None)
    if digest != canonical_sha256(body):
        raise ValueError("probe report self-hash mismatch")
    expected = {
        "experiment_id": "687",
        "parent_experiment_id": "686",
        "outer_fold": 3,
        "objective": "outer_validation_label_free_gradient_conflict_probe",
        "inline_technical_preflight": True,
        "outer_validation_rows_read": 0,
        "outer_validation_labels_read": 0,
        "outer_quality_metrics_computed": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "submission_artifact": False,
        "pairs": 5440,
        "train_rows": 2280,
        "optimizer_steps_executed": 680,
        "rank_loss_weight": 0.5,
        "hard_loss_weight": 1.0,
    }
    mismatch = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatch:
        raise ValueError(f"probe report contract mismatch: {mismatch}")
    rows = report.get("measurements")
    if not isinstance(rows, list) or len(rows) != len(CHECKPOINT_STEPS) * DIAGNOSTIC_EFFECTIVE_BATCHES:
        raise ValueError("probe measurement coverage mismatch")
    if any(
        not math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key not in {"checkpoint_step", "batch_index", "conflict"}
    ):
        raise ValueError("non-finite probe measurement")
    by_checkpoint = {
        step: [row for row in rows if int(row["checkpoint_step"]) == step]
        for step in CHECKPOINT_STEPS
    }
    recomputed = pcgrad_gate(by_checkpoint)
    if recomputed != report.get("pcgrad_gate") or recomputed["decision"] != report.get("decision"):
        raise ValueError("probe gate does not reproduce")
    for field in (
        "pair_runtime_contract_sha256",
        "pair_runtime_acceptance_sha256",
        "source_runtime_contract_sha256",
        "parent_code_bundle_sha256",
        "parent_code_revision",
        "parent_code_acceptance_sha256",
        "probe_code_bundle_sha256",
        "probe_code_revision",
        "vendor_zip_sha256",
        "vendor_bridge_sha256",
        "model_revision",
        "model_tree_sha256",
        "initial_trainable_state_sha256",
        "final_trainable_state_sha256",
        "diagnostic_pair_indices_sha256",
        "ordered_pair_index_sha256",
    ):
        value = report.get(field)
        if not isinstance(value, str) or len(value) not in {40, 64}:
            raise ValueError(f"invalid provenance field: {field}")
    acceptance: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "687",
        "decision": "ACCEPT_GRADIENT_CONFLICT_PROBE",
        "scientific_decision": report["decision"],
        "report_sha256": digest,
        "measurements": len(rows),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "diagnostic_effective_batches": DIAGNOSTIC_EFFECTIVE_BATCHES,
        "outer_validation_rows_read": 0,
        "outer_validation_labels_read": 0,
        "outer_quality_metrics_computed": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "submission_artifact": False,
    }
    acceptance["acceptance_sha256"] = canonical_sha256(acceptance)
    return acceptance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite probe acceptance")
    value = verify(args.report)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value), flush=True)


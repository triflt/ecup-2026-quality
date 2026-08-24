from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_lora as control
from train_fold import (
    EXPERIMENT_ID,
    CONTROL_EXPERIMENT_ID,
    FLAMMABLE,
    SOFT_LOSS_WEIGHT,
    TEMPERATURE,
    distillation_primary_loss,
)


EXPECTED_TRAIN_OCCURRENCES = 2590
EXPECTED_OPTIMIZER_STEPS = 162


def load_full_runtime(
    runtime_dir: Path, spec_id: str, fold: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if spec_id != CONTROL_EXPERIMENT_ID or fold != 0:
        raise ValueError("full refit uses the frozen 641 cell with synthetic fold 0")
    train_path = runtime_dir / "train.jsonl"
    validation_path = runtime_dir / "validation.jsonl"
    audit_path = runtime_dir / "runtime_audit.json"
    if not train_path.is_file() or not validation_path.is_file() or not audit_path.is_file():
        raise ValueError("full runtime file missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payload = dict(audit)
    digest = payload.pop("contract_sha256", None)
    if digest != control.canonical_sha256(payload):
        raise ValueError("full runtime audit self-hash mismatch")
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "scope": "single_post_validation_flammable_distillation_full_competition_train_refit",
        "changed_factor": "hard_bce_to_fixed_hard_plus_teacher_soft_bce",
        "train_occurrences": EXPECTED_TRAIN_OCCURRENCES,
        "validation_rows": 0,
        "bad_train_occurrences": 0,
        "flammable_train_occurrences": EXPECTED_TRAIN_OCCURRENCES,
        "temperature": TEMPERATURE,
        "soft_loss_weight": SOFT_LOSS_WEIGHT,
        "teacher_target_scope": (
            "development_rowwise_oof_plus_label_free_sealed_teacher0"
        ),
        "uses_27b_at_inference": False,
        "public_used": False,
        "decision": "GO_SINGLE_REFIT",
    }
    mismatch = {
        key: {"expected": value, "actual": audit.get(key)}
        for key, value in expected.items()
        if audit.get(key) != value
    }
    if mismatch:
        raise ValueError(f"full runtime audit mismatch: {mismatch}")
    if audit.get("output_sha256") != {
        "train.jsonl": control.sha256_file(train_path),
        "validation.jsonl": control.sha256_file(validation_path),
    }:
        raise ValueError("full runtime payload checksum mismatch")
    train = control.read_jsonl(train_path)
    validation = control.read_jsonl(validation_path)
    if len(train) != EXPECTED_TRAIN_OCCURRENCES or validation:
        raise ValueError("full runtime row count mismatch")
    if any(
        int(row["fold"]) != -1
        or row["category"] != FLAMMABLE
        or "label" not in row
        or "teacher_score" not in row
        for row in train
    ):
        raise ValueError("full runtime train schema/category mismatch")
    return train, validation, audit


def rewrite_contract(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("experiment_id") != CONTROL_EXPERIMENT_ID:
        raise ValueError("unexpected control report")
    if report.get("technical_smoke") is not False:
        raise ValueError("full refit cannot be a technical smoke")
    if report.get("train_occurrences") != EXPECTED_TRAIN_OCCURRENCES:
        raise ValueError("full refit occurrence count drifted")
    if report.get("validation_rows") != 0 or report.get("optimizer_steps_executed") != EXPECTED_OPTIMIZER_STEPS:
        raise ValueError("full refit update/validation contract drifted")
    source_hash = report.get("contract_sha256")
    result = dict(report)
    result.pop("contract_sha256")
    result.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "control_experiment_id": CONTROL_EXPERIMENT_ID,
            "full_refit": True,
            "changed_factor": "hard_bce_to_fixed_hard_plus_teacher_soft_bce",
            "temperature": TEMPERATURE,
            "soft_loss_weight": SOFT_LOSS_WEIGHT,
            "source_control_contract_sha256": source_hash,
            "category_scope": FLAMMABLE,
            "teacher_target_scope": (
                "development_rowwise_oof_plus_label_free_sealed_teacher0"
            ),
            "submission_base_model": "Qwen/Qwen3.5-4B",
            "uses_27b_at_training": False,
            "uses_27b_at_training_targets": True,
            "uses_27b_at_inference": False,
            "decision": "GO_PACKAGE_AFTER_FULL_CV",
        }
    )
    result["contract_sha256"] = control.canonical_sha256(result)
    return result


def run(args: Any) -> dict[str, Any]:
    original_loader = control.load_runtime
    original_loss = control.primary_loss
    control.load_runtime = load_full_runtime
    control.primary_loss = distillation_primary_loss
    try:
        parent = control.run(CONTROL_EXPERIMENT_ID, args)
    finally:
        control.load_runtime = original_loader
        control.primary_loss = original_loss
    result = rewrite_contract(parent)
    (args.output_dir / "output_contract.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    arguments = control.parser_for(CONTROL_EXPERIMENT_ID).parse_args()
    if arguments.fold != 0:
        raise ValueError("full refit requires synthetic --fold 0")
    print(json.dumps(run(arguments), ensure_ascii=False, indent=2, sort_keys=True))

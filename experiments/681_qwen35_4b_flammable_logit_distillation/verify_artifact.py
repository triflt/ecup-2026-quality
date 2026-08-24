from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "681"
CONTROL_EXPERIMENT_ID = "641"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
FLAMMABLE = "Легковоспламеняющиеся"
EXPECTED_TRAIN_OCCURRENCES = 2280
EXPECTED_VALIDATION_ROWS = {0: 943, 1: 943, 2: 944, 3: 943, 4: 943}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def verify_self_hash(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    digest = payload.pop(field, None)
    if digest != canonical_sha256(payload):
        raise ValueError(f"{field} self-hash mismatch")
    return str(digest)


def verify(
    archive_path: Path,
    *,
    fold: int,
    runtime_dir: Path,
    technical_smoke: bool,
) -> dict[str, Any]:
    if fold not in EXPECTED_VALIDATION_ROWS:
        raise ValueError("fold must be 0..4")
    runtime_audit_path = runtime_dir / "runtime_audit.json"
    runtime_validation = runtime_dir / "validation.jsonl"
    runtime = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
    runtime_contract = verify_self_hash(runtime, "contract_sha256")
    runtime_payload = runtime_validation.read_bytes()
    if runtime.get("output_sha256", {}).get("validation.jsonl") != sha256_bytes(
        runtime_payload
    ):
        raise ValueError("runtime validation checksum mismatch")
    expected_runtime = {
        "experiment_id": CONTROL_EXPERIMENT_ID,
        "distillation_experiment_id": EXPERIMENT_ID,
        "objective": "class_only",
        "outer_fold": fold,
        "train_occurrences": EXPECTED_TRAIN_OCCURRENCES,
        "validation_rows": EXPECTED_VALIDATION_ROWS[fold],
        "changed_factor": "hard_bce_to_fixed_hard_plus_teacher_soft_bce",
        "teacher_experiment_id": "662",
        "teacher_outer_fold": fold,
        "teacher_target_scope": "outer_train_in_sample_outer_validation_unread",
        "temperature": 2.0,
        "soft_loss_weight": 0.5,
        "ordinary_oof_merge_used": False,
        "outer_validation_teacher_overlap": 0,
        "public_used": False,
        "decision": "GO",
    }
    runtime_mismatch = {
        key: {"expected": expected, "actual": runtime.get(key)}
        for key, expected in expected_runtime.items()
        if runtime.get(key) != expected
    }
    if runtime_mismatch:
        raise ValueError(f"runtime contract mismatch: {runtime_mismatch}")
    teacher_manifest = runtime.get("teacher_manifest", {})
    if not (
        set(teacher_manifest)
        == {
            "archive_sha256",
            "report_contract_sha256",
            "runtime_contract_sha256",
            "teacher_scores_sha256",
            "ordered_occurrence_key_sha256",
            "teacher_adapter_manifest_sha256",
        }
        and all(
            isinstance(value, str) and len(value) == 64
            for value in teacher_manifest.values()
        )
    ):
        raise ValueError("teacher manifest is incomplete")

    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP member: {bad}")
        names = archive.namelist()
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("unsafe ZIP member")
        required = {
            "output_contract.json",
            "predictions.jsonl",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
        }
        if not required <= set(names):
            raise ValueError("required artifact member missing")
        contract = json.loads(archive.read("output_contract.json"))
        predictions_payload = archive.read("predictions.jsonl")
        predictions = [json.loads(line) for line in predictions_payload.decode().splitlines()]
        adapter_config = json.loads(archive.read("adapter/adapter_config.json"))

    verify_self_hash(contract, "contract_sha256")
    expected_train = 8 if technical_smoke else EXPECTED_TRAIN_OCCURRENCES
    expected_validation = 2 if technical_smoke else EXPECTED_VALIDATION_ROWS[fold]
    expected_updates = 1 if technical_smoke else 143
    expected_contract = {
        "experiment_id": EXPERIMENT_ID,
        "control_experiment_id": CONTROL_EXPERIMENT_ID,
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": MODEL_REVISION,
        "objective": "class_only",
        "changed_factor": "hard_bce_to_fixed_hard_plus_teacher_soft_bce",
        "category_scope": FLAMMABLE,
        "temperature": 2.0,
        "soft_loss_weight": 0.5,
        "teacher_target_semantics": "raw_last_token_logit_1_minus_logit_0",
        "teacher_target_scope": "outer_train_in_sample_outer_validation_unread",
        "ordinary_oof_merge_used": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
        "uses_27b_at_training": False,
        "uses_27b_at_training_targets": True,
        "uses_27b_at_inference": False,
        "outer_fold": fold,
        "train_occurrences": expected_train,
        "validation_rows": expected_validation,
        "optimizer_steps_executed": expected_updates,
        "runtime_micro_batch_size": 2,
        "runtime_gradient_accumulation": 8,
        "effective_batch_size": 16,
        "runtime_contract_sha256": runtime_contract,
        "technical_smoke": technical_smoke,
        "threshold": 0.0,
        "threshold_tuned": False,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "decision": "TECHNICAL_SMOKE_ONLY" if technical_smoke else "GO_EVALUATE",
    }
    mismatch = {
        key: {"expected": expected, "actual": contract.get(key)}
        for key, expected in expected_contract.items()
        if contract.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"output contract mismatch: {mismatch}")
    if contract.get("artifacts", {}).get("predictions.jsonl") != sha256_bytes(
        predictions_payload
    ):
        raise ValueError("prediction checksum mismatch")
    if len(predictions) != expected_validation:
        raise ValueError("prediction row count mismatch")
    required_schema = {
        "category",
        "concept",
        "fold",
        "format_valid",
        "generated_verdict",
        "global_index",
        "grounded",
        "grounding_source",
        "id",
        "image_index",
        "model_id",
        "model_revision",
        "objective",
        "prediction",
        "preprocessing_version",
        "prompt_version",
        "quote",
        "raw_generation",
        "region_index",
        "score",
        "target_order",
    }
    if any(set(row) != required_schema for row in predictions):
        raise ValueError("prediction schema mismatch")
    if any(int(row["fold"]) != fold or row["category"] != FLAMMABLE for row in predictions):
        raise ValueError("prediction fold/category mismatch")
    if any(not math.isfinite(float(row["score"])) for row in predictions):
        raise ValueError("non-finite prediction score")
    if any(int(row["prediction"]) != int(float(row["score"]) >= 0.0) for row in predictions):
        raise ValueError("prediction differs from frozen zero threshold")

    runtime_rows = [json.loads(line) for line in runtime_payload.decode().splitlines()]
    runtime_rows = runtime_rows[:expected_validation]
    forbidden = {"label", "target", "gold", "answer", "sealed", "public"}
    if len(runtime_rows) != expected_validation or any(
        forbidden.intersection(row) for row in runtime_rows
    ):
        raise ValueError("runtime validation scope or supervision mismatch")
    keys = ("global_index", "id", "fold", "category")
    exact_runtime_binding = all(
        tuple(prediction[key] for key in keys) == tuple(runtime_row[key] for key in keys)
        for prediction, runtime_row in zip(predictions, runtime_rows, strict=True)
    )
    if not exact_runtime_binding:
        raise ValueError("predictions differ from frozen runtime validation")
    if adapter_config.get("base_model_name_or_path") not in {
        "Qwen/Qwen3.5-4B",
        "/hf_models",
    }:
        raise ValueError("adapter base model is not the deployable 4B model")
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "outer_fold": fold,
        "technical_smoke": technical_smoke,
        "rows": len(predictions),
        "archive_sha256": sha256_bytes(archive_path.read_bytes()),
        "predictions_sha256": sha256_bytes(predictions_payload),
        "runtime_validation_sha256": sha256_bytes(runtime_payload),
        "runtime_contract_sha256": runtime_contract,
        "exact_runtime_binding": exact_runtime_binding,
        "deployable_4b_only": True,
        "uses_27b_at_inference": False,
        "decision": "PASS",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--archive", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    result.add_argument("--runtime-dir", type=Path, required=True)
    result.add_argument("--technical-smoke", action="store_true")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            verify(
                args.archive,
                fold=args.fold,
                runtime_dir=args.runtime_dir,
                technical_smoke=args.technical_smoke,
            ),
            indent=2,
            sort_keys=True,
        )
    )

from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from pathlib import Path

from contract import (
    CONTROL_EXPERIMENT_ID,
    EFFECTIVE_BATCH_SIZE,
    EXPECTED_TARGET_COUNTS,
    EXPECTED_TARGET_TOTAL,
    EXPECTED_TRAIN_OCCURRENCES,
    EXPECTED_VALIDATION_ROWS,
    EXPERIMENT_ID,
    GEMMA_SOFT_IMAGE_TOKENS,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    MICRO_BATCH_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    SCREEN_FOLDS,
    canonical_sha256,
    verify_self_hash,
)


def verify(archive_path: Path, runtime_dir: Path, fold: int, technical_smoke: bool) -> dict:
    runtime = json.loads((runtime_dir / "runtime_audit.json").read_text(encoding="utf-8"))
    runtime_contract = verify_self_hash(runtime)
    # Read the delivery exactly once so CRC, members and the recorded archive
    # digest are all bound to the same immutable byte payload.  Besides being
    # stricter, this avoids macOS provenance guards that can deny a second
    # open of a freshly downloaded large archive.
    archive_payload = archive_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
        if archive.testzip() is not None:
            raise ValueError("corrupt ZIP")
        names = archive.namelist()
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("unsafe ZIP member")
        required = {"output_contract.json", "predictions.jsonl", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"}
        if not required.issubset(names):
            raise ValueError("artifact member missing")
        contract = json.loads(archive.read("output_contract.json"))
        predictions_payload = archive.read("predictions.jsonl")
        predictions = [json.loads(line) for line in predictions_payload.decode().splitlines()]
        adapter_config = json.loads(archive.read("adapter/adapter_config.json"))
    verify_self_hash(contract)
    expected_rows = 2 if technical_smoke else EXPECTED_VALIDATION_ROWS[fold]
    expected_train = 2 if technical_smoke else EXPECTED_TRAIN_OCCURRENCES
    expected_updates = 1 if technical_smoke else 306
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "control_experiment_id": CONTROL_EXPERIMENT_ID,
        "outer_fold": fold,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "objective": "class_only",
        "changed_factor": "qwen35_4b_to_gemma4_e4b",
        "runtime_contract_sha256": runtime_contract,
        "train_occurrences": expected_train,
        "validation_rows": expected_rows,
        "technical_smoke": technical_smoke,
        "runtime_micro_batch_size": MICRO_BATCH_SIZE,
        "runtime_gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "optimizer_steps_executed": expected_updates,
        "learning_rate": LEARNING_RATE,
        "threshold": 0.0,
        "threshold_tuned": False,
        "gemma_soft_image_tokens": GEMMA_SOFT_IMAGE_TOKENS,
        "target_module_count": EXPECTED_TARGET_TOTAL,
        "target_module_counts": EXPECTED_TARGET_COUNTS,
        "adapter_reloaded": True,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "decision": "TECHNICAL_SMOKE_ONLY" if technical_smoke else "GO_EVALUATE",
    }
    mismatch = {k: {"expected": v, "actual": contract.get(k)} for k, v in expected.items() if contract.get(k) != v}
    if mismatch:
        raise ValueError(f"output contract mismatch: {mismatch}")
    if float(contract.get("reload_score_abs_delta", 1.0)) > 1e-4:
        raise ValueError("adapter reload parity failed")
    if contract.get("model_class") != "Gemma4ForConditionalGeneration":
        raise ValueError("unexpected model class")
    required_processor_keys = {
        "attention_mask",
        "image_position_ids",
        "input_ids",
        "mm_token_type_ids",
        "pixel_values",
    }
    if not required_processor_keys.issubset(contract.get("processor_output_keys", [])):
        raise ValueError("Gemma processor contract is incomplete")
    peak_memory = int(contract.get("peak_cuda_memory_bytes", 0))
    if peak_memory <= 0 or peak_memory >= 75 * 1024**3:
        raise ValueError("technical smoke has insufficient H100 memory reserve")
    if contract.get("artifacts", {}).get("predictions.jsonl") != sha256_bytes(predictions_payload):
        raise ValueError("prediction checksum mismatch")
    if len(predictions) != expected_rows:
        raise ValueError("prediction row count mismatch")
    required_schema = {"global_index", "id", "fold", "category", "score", "prediction", "model_id", "model_revision", "objective", "prompt_version", "preprocessing_version"}
    if any(set(row) != required_schema for row in predictions):
        raise ValueError("prediction schema mismatch")
    if any(not math.isfinite(float(row["score"])) or int(row["prediction"]) != int(float(row["score"]) >= 0) for row in predictions):
        raise ValueError("invalid prediction")
    runtime_rows = read_jsonl(runtime_dir / "validation.jsonl")[:expected_rows]
    keys = ("global_index", "id", "fold", "category")
    if any(tuple(pred[key] for key in keys) != tuple(row[key] for key in keys) for pred, row in zip(predictions, runtime_rows, strict=True)):
        raise ValueError("predictions are not bound to runtime order")
    targets = adapter_config.get("target_modules")
    if not isinstance(targets, list) or len(targets) != EXPECTED_TARGET_TOTAL:
        raise ValueError("adapter target topology mismatch")
    pattern = re.compile(
        r"^model\.language_model\.layers\.(\d+)\.self_attn\."
        r"(q_proj|k_proj|v_proj|o_proj)$"
    )
    target_counts = {name: 0 for name in EXPECTED_TARGET_COUNTS}
    for target in targets:
        match = pattern.fullmatch(str(target))
        if match is None:
            raise ValueError(f"non-text LoRA target: {target}")
        target_counts[match.group(2)] += 1
    if target_counts != EXPECTED_TARGET_COUNTS:
        raise ValueError("adapter target counts drifted")
    if adapter_config.get("base_model_name_or_path") not in {MODEL_ID, "/hf_models"}:
        raise ValueError("adapter base model mismatch")
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "fold": fold,
        "technical_smoke": technical_smoke,
        "archive_sha256": sha256_bytes(archive_payload),
        "predictions_sha256": sha256_bytes(predictions_payload),
        "runtime_contract_sha256": runtime_contract,
        "rows": len(predictions),
        "exact_runtime_binding": True,
        "adapter_reload_parity": True,
        "peak_cuda_memory_bytes": peak_memory,
        "target_module_count": len(targets),
        "decision": "ACCEPT_ARTIFACT",
    }
    result["acceptance_sha256"] = canonical_sha256(result)
    return result


def sha256_bytes(payload: bytes) -> str:
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=SCREEN_FOLDS, required=True)
    parser.add_argument("--technical-smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.runtime_dir, args.fold, args.technical_smoke), indent=2, sort_keys=True))

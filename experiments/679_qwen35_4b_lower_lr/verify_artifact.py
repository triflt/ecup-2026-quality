from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any


MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_TRAIN_OCCURRENCES = {0: 4892, 1: 4894, 2: 4892, 3: 4892, 4: 4894}
EXPECTED_VALIDATION_ROWS = {0: 2224, 1: 2223, 2: 2224, 3: 2224, 4: 2223}
EXPECTED_RUNTIME_CONTRACTS = {
    0: "38802115365cef7e3a0c1a82abc5efc5ce92a41e0046f1ddad4ab9e02647c568",
    1: "3d62eed9817bbbdb0ff4d55dd511904b4fa1dfef5db44deddb104dda0c60f576",
    2: "321b5e5165110fc729598956d121208ab14aee12af38e8f0b3ab81ddbd52f5e8",
    3: "e08a51c2db16163953c45841f3dd1e7b30b293a7265b2bbd8084d1479c20ea36",
    4: "22cad9c7a1510a73b6ec606826839328329a9c0469f2fb00c45fd24ed9dcf33f",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def verify(path: Path, *, fold: int, runtime_validation: Path) -> dict[str, Any]:
    if fold not in EXPECTED_VALIDATION_ROWS:
        raise ValueError("fold must be 0..4")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP member: {bad}")
        names = archive.namelist()
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("unsafe ZIP member")
        required_members = {
            "output_contract.json",
            "predictions.jsonl",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
        }
        if not required_members <= set(names):
            raise ValueError("required artifact member missing")
        contract = json.loads(archive.read("output_contract.json"))
        predictions_payload = archive.read("predictions.jsonl")
        predictions = [json.loads(line) for line in predictions_payload.decode().splitlines()]
        adapter_config = json.loads(archive.read("adapter/adapter_config.json"))
    contract_payload = dict(contract)
    contract_digest = contract_payload.pop("contract_sha256", None)
    if contract_digest != canonical_sha256(contract_payload):
        raise ValueError("output contract self-hash mismatch")
    expected = {
        "experiment_id": "679",
        "control_experiment_id": "641",
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": MODEL_REVISION,
        "objective": "class_only",
        "changed_factor": "learning_rate_only",
        "control_learning_rate": 0.0002,
        "candidate_learning_rate": 0.0001,
        "outer_fold": fold,
        "train_occurrences": EXPECTED_TRAIN_OCCURRENCES[fold],
        "validation_rows": EXPECTED_VALIDATION_ROWS[fold],
        "optimizer_steps_executed": 306,
        "runtime_micro_batch_size": 2,
        "runtime_gradient_accumulation": 8,
        "effective_batch_size": 16,
        "runtime_contract_sha256": EXPECTED_RUNTIME_CONTRACTS[fold],
        "technical_smoke": False,
        "threshold": 0.0,
        "threshold_tuned": False,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "uses_27b_at_training": False,
        "uses_27b_at_inference": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
        "decision": "GO_EVALUATE",
    }
    mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"output contract mismatch: {mismatch}")
    if contract.get("artifacts", {}).get("predictions.jsonl") != sha256_bytes(
        predictions_payload
    ):
        raise ValueError("prediction checksum mismatch")
    if len(predictions) != EXPECTED_VALIDATION_ROWS[fold]:
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
    if any(int(row["fold"]) != fold for row in predictions):
        raise ValueError("prediction fold mismatch")
    if any(not math.isfinite(float(row["score"])) for row in predictions):
        raise ValueError("non-finite score")
    if any(int(row["prediction"]) != int(float(row["score"]) >= 0.0) for row in predictions):
        raise ValueError("prediction differs from frozen zero threshold")
    runtime_payload = runtime_validation.read_bytes()
    runtime_rows = [json.loads(line) for line in runtime_payload.decode().splitlines()]
    if len(runtime_rows) != EXPECTED_VALIDATION_ROWS[fold]:
        raise ValueError("runtime validation row count mismatch")
    forbidden = {"label", "target", "gold", "answer", "sealed", "public"}
    if any(forbidden.intersection(row) for row in runtime_rows):
        raise ValueError("runtime validation contains supervision or sealed fields")
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
        "experiment_id": "679",
        "outer_fold": fold,
        "rows": len(predictions),
        "archive_sha256": sha256_bytes(path.read_bytes()),
        "predictions_sha256": sha256_bytes(predictions_payload),
        "runtime_validation_sha256": sha256_bytes(runtime_payload),
        "exact_runtime_binding": exact_runtime_binding,
        "candidate_learning_rate": contract["candidate_learning_rate"],
        "deployable_4b_only": True,
        "decision": "PASS",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--archive", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    result.add_argument("--runtime-validation", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            verify(args.archive, fold=args.fold, runtime_validation=args.runtime_validation),
            indent=2,
            sort_keys=True,
        )
    )

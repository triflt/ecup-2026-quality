from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from build_pair_runtime import canonical_sha256, sha256_bytes
from train_pair_fold import (
    EFFECTIVE_BATCH_ROWS,
    EXPERIMENT_ID,
    FLAMMABLE,
    GRADIENT_ACCUMULATION_PAIRS,
    LEARNING_RATE,
    MODES,
    RANK_LOSS_WEIGHT,
    SEED,
    load_inputs,
)


MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_VALIDATION_ROWS = {0: 943, 1: 943, 2: 944, 3: 943, 4: 943}


def verify_self_hash(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    digest = body.pop(field, None)
    if digest != canonical_sha256(body):
        raise ValueError(f"{field} self-hash mismatch")
    return str(digest)


def verify(
    output_dir: Path,
    *,
    fold: int,
    mode: str,
    source_runtime: Path,
    pair_runtime: Path,
    technical_smoke: bool,
) -> dict[str, Any]:
    if fold not in EXPECTED_VALIDATION_ROWS or mode not in MODES:
        raise ValueError("invalid fold or mode")
    train, validation, pairs, pair_acceptance, source_audit = load_inputs(
        source_runtime, pair_runtime, fold
    )
    required = {
        "output_contract.json": output_dir / "output_contract.json",
        "predictions.jsonl": output_dir / "predictions.jsonl",
        "adapter/adapter_config.json": output_dir / "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors": (
            output_dir / "adapter/adapter_model.safetensors"
        ),
    }
    if not output_dir.is_dir() or any(not path.is_file() for path in required.values()):
        raise ValueError("training output directory schema is incomplete")
    contract = json.loads(required["output_contract.json"].read_text(encoding="utf-8"))
    predictions_payload = required["predictions.jsonl"].read_bytes()
    predictions = [
        json.loads(line) for line in predictions_payload.decode().splitlines()
    ]
    adapter_config_payload = required["adapter/adapter_config.json"].read_bytes()
    adapter_model_payload = required[
        "adapter/adapter_model.safetensors"
    ].read_bytes()
    adapter_config = json.loads(adapter_config_payload)
    contract_sha = verify_self_hash(contract, "contract_sha256")
    expected_pairs = 8 if technical_smoke else len(pairs)
    expected_updates = expected_pairs // GRADIENT_ACCUMULATION_PAIRS
    expected_validation = 2 if technical_smoke else EXPECTED_VALIDATION_ROWS[fold]
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": "641",
        "outer_fold": fold,
        "mode": mode,
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": MODEL_REVISION,
        "objective": "class_only_pairwise_rank_distillation",
        "pair_runtime_contract_sha256": pair_acceptance[
            "runtime_contract_sha256"
        ],
        "pair_runtime_acceptance_sha256": pair_acceptance["acceptance_sha256"],
        "source_641_runtime_contract_sha256": source_audit["contract_sha256"],
        "seed": SEED,
        "epochs": 1,
        "learning_rate": LEARNING_RATE,
        "micro_batch_pairs": 1,
        "micro_batch_rows": 2,
        "gradient_accumulation_pairs": GRADIENT_ACCUMULATION_PAIRS,
        "effective_batch_rows": EFFECTIVE_BATCH_ROWS,
        "rank_loss_weight": RANK_LOSS_WEIGHT if mode == "rank_candidate" else 0.0,
        "pairs": expected_pairs,
        "optimizer_steps_executed": expected_updates,
        "validation_rows": expected_validation,
        "technical_smoke": technical_smoke,
        "runtime_backend": "legacy_eager",
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "threshold": 0.0,
        "threshold_tuned": False,
        "decision": "TECHNICAL_SMOKE_ONLY" if technical_smoke else "GO_EVALUATE",
    }
    mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"output contract mismatch: {mismatch}")
    if not isinstance(contract.get("ordered_pair_index_sha256"), str) or len(
        contract["ordered_pair_index_sha256"]
    ) != 64:
        raise ValueError("ordered pair schedule binding is missing")
    if contract.get("peak_cuda_bytes", 2**63) >= 75 * 1024**3:
        raise ValueError("measured peak memory exceeds one-H100 gate")
    if technical_smoke and (
        contract.get("reload_prediction_mismatches") != 0
        or float(contract.get("reload_max_abs_score_delta", math.inf)) > 1e-5
    ):
        raise ValueError("technical smoke save/reload parity failed")
    if contract.get("artifacts") != {
        "predictions.jsonl": sha256_bytes(predictions_payload),
        "adapter/adapter_config.json": sha256_bytes(adapter_config_payload),
        "adapter/adapter_model.safetensors": sha256_bytes(adapter_model_payload),
    }:
        raise ValueError("inner artifact checksum mismatch")
    if len(predictions) != expected_validation:
        raise ValueError("prediction row count mismatch")
    required_prediction_fields = {
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
    if any(set(row) != required_prediction_fields for row in predictions):
        raise ValueError("prediction schema mismatch")
    expected_validation_rows = validation[:expected_validation]
    binding_fields = ("global_index", "id", "fold", "category")
    if any(
        tuple(prediction[field] for field in binding_fields)
        != tuple(runtime_row[field] for field in binding_fields)
        for prediction, runtime_row in zip(
            predictions, expected_validation_rows, strict=True
        )
    ):
        raise ValueError("prediction/runtime binding mismatch")
    if any(
        row["category"] != FLAMMABLE
        or not math.isfinite(float(row["score"]))
        or int(row["prediction"]) != int(float(row["score"]) >= 0.0)
        for row in predictions
    ):
        raise ValueError("prediction value or threshold mismatch")
    if adapter_config.get("base_model_name_or_path") not in {
        "Qwen/Qwen3.5-4B",
        "/hf_models",
    }:
        raise ValueError("adapter base model is not deployable Qwen3.5-4B")
    if set(adapter_config.get("target_modules", [])) != {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }:
        raise ValueError("LoRA target modules drifted")
    if (
        int(adapter_config.get("r", -1)) != 16
        or int(adapter_config.get("lora_alpha", -1)) != 32
        or float(adapter_config.get("lora_dropout", -1)) != 0.05
        or adapter_config.get("use_rslora") is not True
    ):
        raise ValueError("LoRA configuration drifted")
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "outer_fold": fold,
        "mode": mode,
        "technical_smoke": technical_smoke,
        "rows": len(predictions),
        "pairs": expected_pairs,
        "predictions_sha256": sha256_bytes(predictions_payload),
        "adapter_config_sha256": sha256_bytes(adapter_config_payload),
        "adapter_model_sha256": sha256_bytes(adapter_model_payload),
        "output_contract_sha256": contract_sha,
        "pair_runtime_contract_sha256": pair_acceptance[
            "runtime_contract_sha256"
        ],
        "source_641_runtime_contract_sha256": source_audit["contract_sha256"],
        "exact_runtime_binding": True,
        "save_reload_parity": bool(technical_smoke),
        "deployable_4b_only": True,
        "uses_27b_at_inference": False,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "decision": "ACCEPT",
    }
    result["acceptance_sha256"] = canonical_sha256(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--pair-runtime", type=Path, required=True)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(
        args.output_dir,
        fold=args.fold,
        mode=args.mode,
        source_runtime=args.source_runtime,
        pair_runtime=args.pair_runtime,
        technical_smoke=args.technical_smoke,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite acceptance report")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")

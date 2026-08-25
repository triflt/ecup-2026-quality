from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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
    load_code_acceptance,
    load_inputs,
    load_promotion_receipt,
    load_transport_acceptance,
    load_vendor_acceptance,
)

MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_VALIDATION_ROWS = {0: 943, 1: 943, 2: 944, 3: 943, 4: 943}
EXPECTED_RUNTIME_PACKAGES = {
    "torch": "2.10.0+cu128",
    "transformers": "5.14.1",
    "peft": "0.20.0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    transport_acceptance: Path,
    code_acceptance: Path,
    promotion_receipt: Path | None,
    promotion_evaluation: Path | None,
    promotion_receipt_source: str | None,
    vendor_acceptance: Path,
    vendor_archive: Path,
    technical_smoke: bool,
) -> dict[str, Any]:
    if fold not in EXPECTED_VALIDATION_ROWS or mode not in MODES:
        raise ValueError("invalid fold or mode")
    _train, validation, pairs, pair_acceptance, source_audit = load_inputs(
        source_runtime, pair_runtime, fold
    )
    transport = load_transport_acceptance(
        transport_acceptance,
        fold=fold,
        pair_acceptance=pair_acceptance,
        source_audit=source_audit,
    )
    code = load_code_acceptance(code_acceptance)
    promotion = load_promotion_receipt(
        promotion_receipt,
        promotion_evaluation,
        fold=fold,
        expected_source=promotion_receipt_source,
    )
    vendor = load_vendor_acceptance(vendor_acceptance, vendor_archive)
    required = {
        "output_contract.json": output_dir / "output_contract.json",
        "predictions.jsonl": output_dir / "predictions.jsonl",
        "adapter/README.md": output_dir / "adapter/README.md",
        "adapter/adapter_config.json": output_dir / "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors": (
            output_dir / "adapter/adapter_model.safetensors"
        ),
    }
    if not output_dir.is_dir() or any(not path.is_file() for path in required.values()):
        raise ValueError("training output directory schema is incomplete")
    observed_files: set[str] = set()
    for path in output_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError("training output contains a symlink")
        if path.is_file():
            observed_files.add(path.relative_to(output_dir).as_posix())
        elif not path.is_dir():
            raise ValueError("training output contains a special filesystem entry")
    final_acceptance = output_dir / "acceptance.json"
    allowed_member_sets = (set(required), {*required, "acceptance.json"})
    if observed_files not in allowed_member_sets:
        raise ValueError(
            f"training output member set mismatch: {sorted(observed_files)}"
        )
    contract = json.loads(required["output_contract.json"].read_text(encoding="utf-8"))
    predictions_payload = required["predictions.jsonl"].read_bytes()
    predictions = [
        json.loads(line) for line in predictions_payload.decode().splitlines()
    ]
    adapter_config_payload = required["adapter/adapter_config.json"].read_bytes()
    adapter_readme_payload = required["adapter/README.md"].read_bytes()
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
        "objective": "class_only_additive_pairwise_rank_distillation",
        "pair_runtime_contract_sha256": pair_acceptance[
            "runtime_contract_sha256"
        ],
        "pair_runtime_acceptance_sha256": pair_acceptance["acceptance_sha256"],
        "transport_acceptance_sha256": transport["transport_acceptance_sha256"],
        "source_641_runtime_contract_sha256": source_audit["contract_sha256"],
        "code_bundle_sha256": code["bundle_sha256"],
        "code_revision": code["git_revision"],
        "code_manifest_sha256": code["manifest_sha256"],
        "code_acceptance_sha256": code["acceptance_sha256"],
        "promotion_receipt_sha256": (
            promotion["promotion_gate_sha256"] if promotion is not None else None
        ),
        "promotion_receipt_file_sha256": (
            sha256_file(promotion_receipt) if promotion_receipt is not None else None
        ),
        "promotion_evaluation_file_sha256": (
            sha256_file(promotion_evaluation)
            if promotion_evaluation is not None
            else None
        ),
        "promotion_receipt_source": promotion_receipt_source,
        "vendor_zip_sha256": vendor["vendor_zip_sha256"],
        "vendor_bridge_sha256": vendor["bridge_sha256"],
        "vendor_source_bundle_sha256": vendor["source_bundle_sha256"],
        "seed": SEED,
        "epochs": 1,
        "learning_rate": LEARNING_RATE,
        "micro_batch_pairs": 1,
        "micro_batch_rows": 2,
        "gradient_accumulation_pairs": GRADIENT_ACCUMULATION_PAIRS,
        "effective_batch_rows": EFFECTIVE_BATCH_ROWS,
        "rank_loss_weight": RANK_LOSS_WEIGHT if mode == "rank_candidate" else 0.0,
        "hard_loss_weight": 1.0,
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
    pair_indices = list(range(expected_pairs))
    random.Random(SEED).shuffle(pair_indices)
    expected_order_sha = hashlib.sha256(
        json.dumps(pair_indices, separators=(",", ":")).encode()
    ).hexdigest()
    if contract.get("ordered_pair_index_sha256") != expected_order_sha:
        raise ValueError("ordered pair schedule binding mismatch")
    for key in ("initial_trainable_state_sha256", "model_tree_sha256"):
        value = contract.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{key} is not a frozen SHA-256")
    if not isinstance(contract.get("model_tree_files"), int) or contract[
        "model_tree_files"
    ] <= 0:
        raise ValueError("model tree file count is invalid")
    runtime_packages = contract.get("runtime_packages")
    if runtime_packages != EXPECTED_RUNTIME_PACKAGES:
        raise ValueError("runtime package versions differ from frozen legacy 641")
    if contract.get("peak_cuda_bytes", 2**63) >= 75 * 1024**3:
        raise ValueError("measured peak memory exceeds one-H100 gate")
    loss_summary = contract.get("training_loss_mean")
    gradient_summary = contract.get("gradient_norm_preclip")
    if (
        not isinstance(loss_summary, dict)
        or set(loss_summary) != {"total", "hard", "rank"}
        or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in loss_summary.values()
        )
        or not isinstance(gradient_summary, dict)
        or set(gradient_summary) != {"mean", "max"}
        or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in gradient_summary.values()
        )
    ):
        raise ValueError("loss or gradient diagnostics are invalid")
    if technical_smoke and (
        contract.get("reload_prediction_mismatches") != 0
        or float(contract.get("reload_max_abs_score_delta", math.inf)) > 1e-5
    ):
        raise ValueError("technical smoke save/reload parity failed")
    if contract.get("artifacts") != {
        "predictions.jsonl": sha256_bytes(predictions_payload),
        "adapter/README.md": sha256_bytes(adapter_readme_payload),
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
        or adapter_config.get("bias") != "none"
        or adapter_config.get("task_type") != "CAUSAL_LM"
        or adapter_config.get("inference_mode") is not True
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
        "pair_runtime_acceptance_sha256": pair_acceptance["acceptance_sha256"],
        "transport_acceptance_sha256": transport["transport_acceptance_sha256"],
        "source_641_runtime_contract_sha256": source_audit["contract_sha256"],
        "code_bundle_sha256": contract["code_bundle_sha256"],
        "code_revision": contract["code_revision"],
        "code_manifest_sha256": contract["code_manifest_sha256"],
        "code_acceptance_sha256": contract["code_acceptance_sha256"],
        "promotion_receipt_sha256": contract["promotion_receipt_sha256"],
        "promotion_receipt_file_sha256": contract[
            "promotion_receipt_file_sha256"
        ],
        "promotion_evaluation_file_sha256": contract[
            "promotion_evaluation_file_sha256"
        ],
        "promotion_receipt_source": contract["promotion_receipt_source"],
        "vendor_zip_sha256": contract["vendor_zip_sha256"],
        "vendor_bridge_sha256": contract["vendor_bridge_sha256"],
        "vendor_source_bundle_sha256": contract["vendor_source_bundle_sha256"],
        "model_id": contract["model_id"],
        "model_revision": contract["model_revision"],
        "model_tree_sha256": contract["model_tree_sha256"],
        "model_tree_files": contract["model_tree_files"],
        "initial_trainable_state_sha256": contract[
            "initial_trainable_state_sha256"
        ],
        "ordered_pair_index_sha256": expected_order_sha,
        "seed": contract["seed"],
        "epochs": contract["epochs"],
        "learning_rate": contract["learning_rate"],
        "micro_batch_pairs": contract["micro_batch_pairs"],
        "micro_batch_rows": contract["micro_batch_rows"],
        "gradient_accumulation_pairs": contract["gradient_accumulation_pairs"],
        "effective_batch_rows": contract["effective_batch_rows"],
        "optimizer_steps_executed": contract["optimizer_steps_executed"],
        "training_loss_mean": loss_summary,
        "gradient_norm_preclip": gradient_summary,
        "rank_loss_weight": contract["rank_loss_weight"],
        "hard_loss_weight": contract["hard_loss_weight"],
        "runtime_backend": contract["runtime_backend"],
        "runtime_packages": runtime_packages,
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
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if final_acceptance.exists() and final_acceptance.read_text(encoding="utf-8") != payload:
        raise ValueError("existing final acceptance differs from independent replay")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--pair-runtime", type=Path, required=True)
    parser.add_argument("--transport-acceptance", type=Path, required=True)
    parser.add_argument("--code-acceptance", type=Path, required=True)
    parser.add_argument("--promotion-receipt", type=Path)
    parser.add_argument("--promotion-evaluation", type=Path)
    parser.add_argument("--promotion-receipt-source")
    parser.add_argument("--vendor-acceptance", type=Path, required=True)
    parser.add_argument("--vendor-archive", type=Path, required=True)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(
        args.output_dir,
        fold=args.fold,
        mode=args.mode,
        source_runtime=args.source_runtime,
        pair_runtime=args.pair_runtime,
        transport_acceptance=args.transport_acceptance,
        code_acceptance=args.code_acceptance,
        promotion_receipt=args.promotion_receipt,
        promotion_evaluation=args.promotion_evaluation,
        promotion_receipt_source=args.promotion_receipt_source,
        vendor_acceptance=args.vendor_acceptance,
        vendor_archive=args.vendor_archive,
        technical_smoke=args.technical_smoke,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            if args.output.read_text(encoding="utf-8") != payload:
                raise FileExistsError("refusing to overwrite different acceptance report")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")

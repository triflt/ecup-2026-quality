from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "645_qwen_scale_2x3_gate"
PREFLIGHT = ROOT / "653_qwen36_27b_lora_runtime_preflight"
for path in (SHARED, PREFLIGHT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grid_contract import base_prompt
from technical_smoke import (
    adapter_manifest,
    first_parameter_device,
    sha256_file,
    validate_device_map,
)
from train_lora import _processor_batch, messages, open_image

EXPERIMENT_ID = "654"
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
SEED = 42
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
EXPECTED_TRAIN_OCCURRENCES = {0: 4892, 1: 4894, 2: 4892, 3: 4892, 4: 4894}
EXPECTED_VALIDATION_ROWS = {0: 2224, 1: 2223, 2: 2224, 3: 2224, 4: 2223}
EXPECTED_OPTIMIZER_STEPS = 306
SCREEN_FOLDS = {0, 3}
ROUTE_EXTENSION_FOLDS = {1, 2, 4}


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_runtime(
    runtime: Path, *, fold: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_path = runtime / "train.jsonl"
    validation_path = runtime / "validation.jsonl"
    audit_path = runtime / "runtime_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payload = dict(audit)
    digest = payload.pop("contract_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("runtime audit self-hash mismatch")
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "objective": "class_only",
        "outer_fold": fold,
        "train_occurrences": EXPECTED_TRAIN_OCCURRENCES[fold],
        "validation_rows": EXPECTED_VALIDATION_ROWS[fold],
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "public_used": False,
        "decision": "GO_AFTER_653",
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("runtime contract differs from experiment 654")
    if audit["output_sha256"] != {
        "train.jsonl": sha256_file(train_path),
        "validation.jsonl": sha256_file(validation_path),
    }:
        raise ValueError("runtime file checksum mismatch")
    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    if any("label" in row or "evidence_target" in row for row in validation):
        raise ValueError("validation supervision is forbidden")
    return train, validation, audit


def last_token_score(model: Any, batch: Any, *, zero_token: int, one_token: int, input_device: Any):
    import torch

    device_batch = {key: value.to(input_device) for key, value in batch.items()}
    outputs = model(**device_batch, use_cache=False)
    mask = device_batch["attention_mask"]
    positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
    last = torch.where(mask.bool(), positions, -1).max(dim=1).values
    rows = torch.arange(mask.shape[0], device=outputs.logits.device)
    logits = outputs.logits[rows, last.to(outputs.logits.device)]
    return logits[:, one_token] - logits[:, zero_token]


def one_score(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    image: Any,
    zero: int,
    one: int,
    input_device: Any,
):
    local = SimpleNamespace(**row)
    batch = _processor_batch(
        processor,
        [messages(local, image, prompt_text=base_prompt(local))],
        add_generation_prompt=True,
    )
    return last_token_score(
        model,
        batch,
        zero_token=zero,
        one_token=one,
        input_device=input_device,
    )


def package(output: Path, report: Path, predictions: Path, adapter: Path) -> Path:
    archive = output / "qwen36_27b_class_only_fold.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(report, "report.json")
        bundle.write(predictions, "predictions.jsonl")
        for path in sorted(adapter.rglob("*")):
            if path.is_file():
                bundle.write(path, str(Path("adapter") / path.relative_to(adapter)))
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt ZIP member: {bad}")
    return archive


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.fold not in SCREEN_FOLDS | ROUTE_EXTENSION_FOLDS:
        raise ValueError("fold must be 0..4")
    if args.fold in ROUTE_EXTENSION_FOLDS:
        gate_path = getattr(args, "route_gate", None)
        if gate_path is None or not gate_path.is_file():
            raise ValueError("remaining folds require the passed experiment-659 route gate")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not (
            gate.get("experiment_id") == "659"
            and gate.get("large_component_experiment_id") == "654"
            and gate.get("passed") is True
            and gate.get("decision") == "OPEN_REMAINING_FOLDS"
            and gate.get("weights") == {"641": 0.5, "654": 0.5}
            and gate.get("threshold") == 0.0
            and gate.get("threshold_tuned") is False
            and gate.get("sealed_rows") == 0
            and gate.get("public_used") is False
        ):
            raise ValueError("experiment-659 route gate contract mismatch")
    if args.model_revision != MODEL_REVISION:
        raise ValueError("model revision differs from the frozen contract")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, validation_rows, runtime_audit = load_runtime(args.runtime_dir, fold=args.fold)

    import numpy as np
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.device_count() < args.expected_cuda_devices:
        raise RuntimeError("requested CUDA devices are unavailable")
    if not args.vendor.is_dir():
        raise FileNotFoundError("vendored PEFT directory is missing")
    sys.path.insert(0, str(args.vendor.resolve()))
    import peft

    if peft.__version__ != "0.20.0":
        raise RuntimeError("exact vendored PEFT 0.20.0 is required")
    from peft import LoraConfig, TaskType, get_peft_model

    processor = AutoProcessor.from_pretrained(args.model_root, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1 or zero == one:
        raise RuntimeError("0 and 1 are not distinct atomic tokens")
    started = time.monotonic()
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_root,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map="balanced",
        attn_implementation="eager",
    )
    device_summary = validate_device_map(
        getattr(model, "hf_device_map", {}), minimum_cuda_devices=args.expected_cuda_devices
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=True,
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    input_device = first_parameter_device(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.device.type != "cuda" for parameter in trainable):
        raise RuntimeError("trainable adapter parameters are not fully resident on CUDA")
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)
    warmup = 15

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, EXPECTED_OPTIMIZER_STEPS - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    optimizer.zero_grad(set_to_none=True)
    indices = list(range(len(train_rows)))
    random.Random(SEED).shuffle(indices)
    optimizer_steps = 0
    for step, index in enumerate(indices, start=1):
        row = train_rows[index]
        image = open_image(args.images, row)
        try:
            logits = one_score(model, processor, row, image, zero[0], one[0], input_device)
            label = torch.tensor([int(row["label"])], dtype=torch.float32, device=logits.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), label)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite training loss")
            (loss / GRADIENT_ACCUMULATION).backward()
        finally:
            image.close()
        if step % GRADIENT_ACCUMULATION == 0 or step == len(indices):
            if any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                for parameter in trainable
            ):
                raise RuntimeError("non-finite adapter gradients")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, foreach=False)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    if optimizer_steps != EXPECTED_OPTIMIZER_STEPS:
        raise RuntimeError("optimizer step count drift")

    model.eval()
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as output:
        for row in validation_rows:
            image = open_image(args.images, row)
            try:
                with torch.inference_mode():
                    value = float(
                        one_score(model, processor, row, image, zero[0], one[0], input_device)
                        .float()
                        .cpu()[0]
                    )
            finally:
                image.close()
            if not math.isfinite(value):
                raise RuntimeError("non-finite validation score")
            output.write(
                json.dumps(
                    {
                        "global_index": int(row["global_index"]),
                        "id": str(row["id"]),
                        "fold": int(row["fold"]),
                        "category": str(row["category"]),
                        "score": value,
                        "prediction": int(value >= 0.0),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "data_version": "competition_train_v1",
        "evaluation_version": "semantic_family_v3",
        "outer_fold": args.fold,
        "seed": SEED,
        "train_occurrences": len(train_rows),
        "validation_rows": len(validation_rows),
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch_size": MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION,
        "optimizer_steps": optimizer_steps,
        "threshold": 0.0,
        "threshold_tuned": False,
        "validation_labels_written": 0,
        "sealed_rows": 0,
        "public_used": False,
        "runtime_contract_sha256": runtime_audit["contract_sha256"],
        "device_map_summary": device_summary,
        "cpu_or_disk_offload": False,
        "predictions_sha256": sha256_file(predictions_path),
        "adapter_manifest": adapter_manifest(adapter_dir),
        "packages": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "peft": peft.__version__,
        },
        "elapsed_seconds": time.monotonic() - started,
        "decision": "READY_FOR_FROZEN_EVALUATION",
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive = package(args.output_dir, report_path, predictions_path, adapter_dir)
    (args.output_dir / "delivery.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "archive": archive.name,
                "archive_sha256": sha256_file(archive),
                "report_sha256": sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--fold", type=int, required=True)
    result.add_argument("--runtime-dir", type=Path, required=True)
    result.add_argument("--images", type=Path, required=True)
    result.add_argument("--model-root", type=Path, required=True)
    result.add_argument("--model-revision", default=MODEL_REVISION)
    result.add_argument("--vendor", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--expected-cuda-devices", type=int, default=4)
    result.add_argument("--route-gate", type=Path)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2, sort_keys=True))

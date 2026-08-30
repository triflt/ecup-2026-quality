from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import random
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from contract import (
    CONTROL_EXPERIMENT_ID,
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    EXPECTED_RUNTIME_CONTRACT,
    EXPECTED_TARGET_COUNTS,
    EXPECTED_TARGET_TOTAL,
    EXPECTED_TRAIN_OCCURRENCES,
    EXPECTED_VALIDATION_ROWS,
    EXPERIMENT_ID,
    GEMMA_SOFT_IMAGE_TOKENS,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    MAX_LENGTH,
    MAX_SOURCE_PIXELS,
    MICRO_BATCH_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    PREPROCESSING_VERSION,
    PROMPT_VERSION,
    SCREEN_FOLDS,
    SEED,
    canonical_sha256,
    sha256_file,
    verify_self_hash,
)

SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))
from grid_contract import base_prompt


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_runtime(runtime_dir: Path, fold: int) -> tuple[list[dict], list[dict], dict]:
    audit = json.loads((runtime_dir / "runtime_audit.json").read_text(encoding="utf-8"))
    verify_self_hash(audit)
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "control_experiment_id": CONTROL_EXPERIMENT_ID,
        "outer_fold": fold,
        "objective": "class_only",
        "source_runtime_contract_sha256": EXPECTED_RUNTIME_CONTRACT[fold],
        "train_occurrences": EXPECTED_TRAIN_OCCURRENCES,
        "validation_rows": EXPECTED_VALIDATION_ROWS[fold],
        "decision": "GO_TECHNICAL_SMOKE_ONLY",
    }
    mismatch = {k: {"expected": v, "actual": audit.get(k)} for k, v in expected.items() if audit.get(k) != v}
    if mismatch:
        raise ValueError(f"runtime audit mismatch: {mismatch}")
    for name in ("train.jsonl", "validation.jsonl"):
        if audit["output_sha256"][name] != sha256_file(runtime_dir / name):
            raise ValueError(f"runtime checksum mismatch: {name}")
    train = read_jsonl(runtime_dir / "train.jsonl")
    validation = read_jsonl(runtime_dir / "validation.jsonl")
    if any(int(row["fold"]) == fold for row in train):
        raise ValueError("outer validation fold entered training")
    if any(int(row["fold"]) != fold for row in validation):
        raise ValueError("validation fold mismatch")
    if any("label" in row for row in validation):
        raise ValueError("validation labels are forbidden")
    return train, validation, audit


def cache_path(cache: Path, row_id: str) -> Path:
    return cache / f"{hashlib.sha256(row_id.encode()).hexdigest()}.img"


def open_image(cache: Path, row: dict[str, Any]):
    from PIL import Image

    cache.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache, str(row["id"]))
    if not path.exists():
        request = urllib.request.Request(str(row["image_url"]), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        path.write_bytes(payload)
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width * height > MAX_SOURCE_PIXELS:
        scale = math.sqrt(MAX_SOURCE_PIXELS / (width * height))
        resized = image.resize(
            (max(48, int(width * scale)), max(48, int(height * scale))),
            getattr(Image, "Resampling", Image).LANCZOS,
        )
        image.close()
        image = resized
    return image


def messages(row: SimpleNamespace, image: Any) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": base_prompt(row)}]}]


def processor_batch(processor: Any, rows: list[SimpleNamespace], images: list[Any]):
    return processor.apply_chat_template(
        [messages(row, image) for row, image in zip(rows, images, strict=True)],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        enable_thinking=False,
        images_kwargs={"max_soft_tokens": GEMMA_SOFT_IMAGE_TOKENS},
    )


def score(model: Any, batch: dict[str, Any], zero_token: int, one_token: int):
    outputs = model(**batch, use_cache=False, logits_to_keep=1)
    logits = outputs.logits[:, -1, :]
    return logits[:, one_token] - logits[:, zero_token]


def resolve_text_targets(model: Any) -> tuple[list[str], dict[str, int]]:
    pattern = re.compile(r"^model\.language_model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")
    targets: list[str] = []
    counts: Counter[str] = Counter()
    for name, module in model.named_modules():
        match = pattern.fullmatch(name)
        if match is None:
            continue
        if module.__class__.__name__ != "Linear":
            raise RuntimeError(f"unexpected text target type: {name}={module.__class__.__name__}")
        targets.append(name)
        counts[match.group(2)] += 1
    if dict(counts) != EXPECTED_TARGET_COUNTS or len(targets) != EXPECTED_TARGET_TOTAL:
        raise RuntimeError(f"Gemma text target topology drifted: {dict(counts)}, total={len(targets)}")
    if any("vision" in name or "audio" in name for name in targets):
        raise RuntimeError("LoRA target escaped into vision/audio tower")
    return sorted(targets), dict(sorted(counts.items()))


def adapter_manifest(adapter_dir: Path) -> dict[str, str]:
    return {str(path.relative_to(adapter_dir)): sha256_file(path) for path in sorted(adapter_dir.rglob("*")) if path.is_file()}


def write_zip(output_dir: Path) -> Path:
    archive_path = output_dir / "gemma4_e4b_lora.zip"
    members = [output_dir / "output_contract.json", output_dir / "predictions.jsonl"]
    members.extend(path for path in sorted((output_dir / "adapter").rglob("*")) if path.is_file())
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, str(path.relative_to(output_dir)))
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("artifact ZIP is corrupt")
    return archive_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if args.model_revision != MODEL_REVISION:
        raise ValueError("model revision differs from frozen contract")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite output")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, validation_rows, runtime = load_runtime(args.runtime_dir, args.fold)
    if args.technical_smoke:
        train_rows = train_rows[:2]
        validation_rows = validation_rows[:2]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    if not args.vendor.is_dir():
        raise FileNotFoundError("vendored PEFT is missing")
    sys.path.insert(0, str(args.vendor.resolve()))
    import peft
    if peft.__version__ != "0.20.0":
        raise RuntimeError("exact PEFT 0.20.0 is required")
    from peft import LoraConfig, TaskType, get_peft_model

    processor = AutoProcessor.from_pretrained(args.model_root, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1 or zero == one:
        raise RuntimeError("0/1 are not distinct atomic tokens")
    started = time.monotonic()
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_root,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    if model.__class__.__name__ != "Gemma4ForConditionalGeneration":
        raise RuntimeError(f"unexpected model class: {model.__class__.__name__}")
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("CPU/disk/meta offload is forbidden")
    target_modules, target_counts = resolve_text_targets(model)
    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=target_modules, bias="none", task_type=TaskType.CAUSAL_LM, use_rslora=True,
    ))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.device.type != "cuda" for parameter in trainable):
        raise RuntimeError("trainable parameters are not entirely on CUDA")
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    full_updates = math.ceil(math.ceil(EXPECTED_TRAIN_OCCURRENCES / MICRO_BATCH_SIZE) / GRADIENT_ACCUMULATION)
    expected_updates = 1 if args.technical_smoke else full_updates
    warmup = max(1, int(full_updates * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * min((step - warmup) / max(1, full_updates - warmup), 1.0))))
    optimizer.zero_grad(set_to_none=True)
    indices = list(range(len(train_rows)))
    random.Random(SEED).shuffle(indices)
    losses: list[float] = []
    processor_output_keys: list[str] = []
    optimizer_steps = 0
    model.train()
    for offset in range(0, len(indices), MICRO_BATCH_SIZE):
        local = [train_rows[index] for index in indices[offset : offset + MICRO_BATCH_SIZE]]
        rows = [SimpleNamespace(**item) for item in local]
        images = [open_image(args.images, item) for item in local]
        try:
            batch = processor_batch(processor, rows, images).to("cuda:0")
            if not processor_output_keys:
                processor_output_keys = sorted(batch.keys())
                required = {"input_ids", "attention_mask", "pixel_values", "image_position_ids", "mm_token_type_ids"}
                if not required.issubset(batch):
                    raise RuntimeError(f"Gemma processor output is incomplete: {processor_output_keys}")
            scores = score(model, batch, zero[0], one[0])
            labels = torch.tensor([int(row.label) for row in rows], dtype=torch.float32, device=scores.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores.float(), labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite loss")
            (loss / GRADIENT_ACCUMULATION).backward()
            losses.append(float(loss.detach().cpu()))
        finally:
            for image in images:
                image.close()
        if args.technical_smoke or ((offset // MICRO_BATCH_SIZE + 1) % GRADIENT_ACCUMULATION == 0) or offset + MICRO_BATCH_SIZE >= len(indices):
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in trainable):
                raise RuntimeError("non-finite LoRA gradients")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, foreach=False)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    if optimizer_steps != expected_updates:
        raise RuntimeError(f"optimizer step drift: {optimizer_steps} != {expected_updates}")
    model.eval(); model.config.use_cache = True
    records = []
    for item in validation_rows:
        row = SimpleNamespace(**item); image = open_image(args.images, item)
        try:
            batch = processor_batch(processor, [row], [image]).to("cuda:0")
            with torch.inference_mode():
                value = float(score(model, batch, zero[0], one[0])[0].float().cpu())
        finally:
            image.close()
        if not math.isfinite(value):
            raise RuntimeError("non-finite validation score")
        records.append({"global_index": int(row.global_index), "id": str(row.id), "fold": int(row.fold), "category": str(row.category), "score": value, "prediction": int(value >= 0.0), "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "objective": "class_only", "prompt_version": PROMPT_VERSION, "preprocessing_version": PREPROCESSING_VERSION})
    predictions = args.output_dir / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    reference_score = float(records[0]["score"])
    model.load_adapter(str(adapter_dir), adapter_name="roundtrip", is_trainable=False)
    model.set_adapter("roundtrip")
    reload_item = validation_rows[0]
    reload_row = SimpleNamespace(**reload_item)
    reload_image = open_image(args.images, reload_item)
    try:
        reload_batch = processor_batch(processor, [reload_row], [reload_image]).to("cuda:0")
        with torch.inference_mode():
            reload_score = float(score(model, reload_batch, zero[0], one[0])[0].float().cpu())
    finally:
        reload_image.close()
    reload_delta = abs(reference_score - reload_score)
    if not math.isfinite(reload_score) or reload_delta > 1e-4:
        raise RuntimeError(f"adapter reload parity failed: delta={reload_delta}")
    report = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "control_experiment_id": CONTROL_EXPERIMENT_ID,
        "outer_fold": args.fold, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_class": model.get_base_model().__class__.__name__, "objective": "class_only",
        "changed_factor": "qwen35_4b_to_gemma4_e4b", "source_runtime_contract_sha256": runtime["source_runtime_contract_sha256"],
        "runtime_contract_sha256": runtime["contract_sha256"], "train_occurrences": len(train_rows), "validation_rows": len(validation_rows),
        "technical_smoke": bool(args.technical_smoke), "seed": SEED, "epochs": EPOCHS,
        "runtime_micro_batch_size": MICRO_BATCH_SIZE, "runtime_gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE, "optimizer_steps_executed": optimizer_steps,
        "learning_rate": LEARNING_RATE, "threshold": 0.0, "threshold_tuned": False,
        "gemma_soft_image_tokens": GEMMA_SOFT_IMAGE_TOKENS, "max_source_pixels": MAX_SOURCE_PIXELS,
        "target_module_count": len(target_modules), "target_module_counts": target_counts,
        "target_modules_sha256": canonical_sha256(target_modules), "trainable_parameters": sum(p.numel() for p in trainable),
        "processor_output_keys": processor_output_keys,
        "losses": losses, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "adapter_reloaded": True, "reload_score": reload_score, "reload_score_abs_delta": reload_delta,
        "validation_labels_read": 0, "sealed_rows_used": 0, "public_used": False,
        "adapter_manifest": adapter_manifest(adapter_dir), "artifacts": {"predictions.jsonl": sha256_file(predictions)},
        "packages": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "peft": peft.__version__},
        "runtime_minutes": (time.monotonic() - started) / 60,
        "decision": "TECHNICAL_SMOKE_ONLY" if args.technical_smoke else "GO_EVALUATE",
    }
    report["contract_sha256"] = canonical_sha256(report)
    (args.output_dir / "output_contract.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    archive = write_zip(args.output_dir)
    (args.output_dir / "delivery.json").write_text(json.dumps({"archive": archive.name, "archive_sha256": sha256_file(archive), "contract_sha256": report["contract_sha256"]}, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=SCREEN_FOLDS, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--technical-smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))

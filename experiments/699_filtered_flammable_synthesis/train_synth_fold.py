from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

RULES = {
    "БАД": (
        "Метка 1 только если в описании или на упаковке есть прямое указание БАД "
        "или dietary supplement. Спортивное питание без такой маркировки, явное "
        "отрицание или отсутствие маркировки — метка 0."
    ),
    "Легковоспламеняющиеся": (
        "Метка 1 для самостоятельного источника огня, горючего вещества или газа, "
        "либо если такой товар входит в комплект. Пустое оборудование, встроенный "
        "источник, горючий материал только как компонент или предмет не в комплекте — 0."
    ),
}
MODEL_CONTRACTS = {
    "qwen35_4b": {
        "model_id": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "loader": "multimodal",
        "micro_batch": 2,
        "accumulation": 8,
    },
    "qwen3vl_2b": {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": "e2378df056d88153dc44616229fa371fcb87e236",
        "loader": "image_text",
        "micro_batch": 4,
        "accumulation": 4,
    },
}
MAX_LENGTH = 1536
SEED = 42
LEARNING_RATE = 2e-4


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_runtime(
    runtime_dir: Path, fold: int, *, refit: bool = False
) -> tuple[list[dict], list[dict], dict]:
    train_path = runtime_dir / "train.jsonl"
    validation_path = runtime_dir / "validation.jsonl"
    audit_path = runtime_dir / "runtime_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payload = dict(audit)
    declared = payload.pop("contract_sha256", None)
    if declared != canonical_sha256(payload):
        raise ValueError("runtime audit self-hash mismatch")
    if refit:
        if fold != -1 or audit.get("decision") != "GO_FULL_REFIT":
            raise ValueError("full-refit runtime decision/fold mismatch")
        if int(audit.get("outer_fold", 0)) != -1:
            raise ValueError("full-refit runtime outer fold mismatch")
    elif (
        audit.get("decision") not in {"GO_GPU_SCREEN", "GO_GPU_APPEND_SCREEN"}
        or int(audit.get("outer_fold", -1)) != fold
    ):
        raise ValueError("runtime audit decision/fold mismatch")
    expected = {
        "train.jsonl": sha256_file(train_path),
        "validation.jsonl": sha256_file(validation_path),
    }
    if audit.get("output_sha256") != expected:
        raise ValueError("runtime payload checksum mismatch")
    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    expected_train = int(audit.get("train_occurrences", -1))
    synthetic_occurrences = int(audit.get("synthetic_occurrences", -1))
    count_valid = False
    if refit:
        count_valid = (
            expected_train == 5460
            and int(audit.get("original_real_occurrences", -1)) == 5450
            and synthetic_occurrences == 10
            and int(audit.get("appended_occurrences", -1)) == 10
            and int(audit.get("appended_positive_occurrences", -1)) == 10
            and int(audit.get("appended_negative_occurrences", -1)) == 0
            and int(audit.get("removed_real_occurrences", -1)) == 0
            and int(audit.get("original_real_rows_changed", -1)) == 0
            and int(audit.get("bad_rows_changed", -1)) == 0
            and int(audit.get("validation_rows", -1)) == 0
            and int(audit.get("validation_labels_written", -1)) == 0
            and int(audit.get("sealed_rows_written", -1)) == 0
            and int(audit.get("public_rows_used", -1)) == 0
        )
    elif audit.get("decision") == "GO_GPU_APPEND_SCREEN":
        original_real = int(audit.get("original_real_occurrences", -1))
        cap = int(audit.get("cap", -1))
        synthetic_repeat = int(audit.get("synthetic_repeat", 1))
        appended_positive = int(audit.get("appended_positive_occurrences", -1))
        appended_negative = int(audit.get("appended_negative_occurrences", -1))
        composition_valid = (
            audit.get("mode") == "positive_only_append"
            and synthetic_repeat in {1, 2, 4}
            and synthetic_occurrences == cap * synthetic_repeat
            and appended_positive == cap * synthetic_repeat
            and appended_negative == 0
        ) or (
            audit.get("mode") == "balanced_append"
            and synthetic_repeat == 1
            and synthetic_occurrences == 2 * cap
            and appended_positive == cap
            and appended_negative == cap
        )
        count_valid = (
            original_real in {4892, 4894}
            and synthetic_occurrences in {5, 10, 19, 20, 40, 160}
            and composition_valid
            and expected_train == original_real + synthetic_occurrences
            and int(audit.get("removed_real_occurrences", -1)) == 0
            and int(audit.get("original_real_rows_changed", -1)) == 0
            and int(audit.get("bad_rows_changed", -1)) == 0
        )
    else:
        count_valid = (
            expected_train in {4892, 4894}
            and synthetic_occurrences in {19, 20, 40, 160}
        )
    if len(train) != expected_train or not count_valid:
        raise ValueError("frozen train/synthetic occurrence count mismatch")
    if any("label" in row for row in validation):
        raise ValueError("validation labels are forbidden")
    if refit:
        real_folds = {int(row["fold"]) for row in train if not row.get("synthetic")}
        if real_folds != set(range(5)):
            raise ValueError("full-refit real rows must cover exact development folds0..4")
    elif any(int(row["fold"]) == fold for row in train if not row.get("synthetic")):
        raise ValueError("outer-fold real row entered training")
    return train, validation, audit


def clean_text(value: object, limit: int) -> str:
    text = html.unescape(str(value or ""))
    import re

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head].rstrip() + " … " + text[-(limit - head) :].lstrip()


def prompt(row: SimpleNamespace) -> str:
    category = str(row.category)
    return (
        f"Категория: {category}\n"
        f"Название: {clean_text(row.name, 320)}\n"
        f"Описание: {clean_text(row.description, 1800)}\n"
        f"Правило: {RULES[category]}\n"
        "Определи правильность категории. Ответь только одной цифрой: 1 или 0."
    )


def cache_path(cache: Path, row_id: str, architecture: str) -> Path:
    suffix = ".jpg" if architecture == "qwen3vl_2b" else ".img"
    return cache / f"{hashlib.sha256(row_id.encode()).hexdigest()}{suffix}"


def validate_image_cache(
    cache: Path, rows: list[dict[str, Any]], architecture: str
) -> dict[str, int]:
    real_ids = sorted({str(row["id"]) for row in rows if not row.get("synthetic")})
    missing = [
        cache_path(cache, row_id, architecture).name
        for row_id in real_ids
        if not cache_path(cache, row_id, architecture).is_file()
    ]
    if missing:
        raise ValueError(
            f"image cache incomplete: {len(missing)} of {len(real_ids)} entries missing"
        )
    return {"unique_real_ids": len(real_ids), "missing": 0}


def training_shape(
    train_occurrences: int, micro_batch: int, accumulation: int, epochs: int
) -> dict[str, int]:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (train_occurrences, micro_batch, accumulation, epochs)
    ):
        raise ValueError("training shape values must be positive integers")
    batches_per_epoch = math.ceil(train_occurrences / micro_batch)
    updates_per_epoch = math.ceil(batches_per_epoch / accumulation)
    return {
        "batches_per_epoch": batches_per_epoch,
        "updates_per_epoch": updates_per_epoch,
        "optimizer_steps": updates_per_epoch * epochs,
        "training_occurrences_seen": train_occurrences * epochs,
    }


def open_real_image(cache: Path, row: dict[str, Any], architecture: str):
    from PIL import Image

    if row.get("synthetic"):
        raise ValueError("synthetic rows must never request an image")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache, str(row["id"]), architecture)
    if not path.is_file():
        raise ValueError("image cache entry missing; network fallback is forbidden")
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if architecture == "qwen35_4b" and width * height > 262144:
        scale = math.sqrt(262144 / (width * height))
        resized = image.resize(
            (max(28, int(width * scale)), max(28, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
        image.close()
        image = resized
    return image


def conversation(
    row: SimpleNamespace, image: Any | None, answer: str | None = None
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt(row)})
    result = [{"role": "user", "content": content}]
    if answer is not None:
        result.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    return result


def processor_batch(
    processor: Any,
    conversations: list[list[dict[str, Any]]],
    *,
    add_generation_prompt: bool,
):
    return processor.apply_chat_template(
        conversations,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        enable_thinking=False,
    )


def assistant_suffix_labels(batch: Any, prompt_batch: Any):
    import torch

    labels = torch.full_like(batch["input_ids"], -100)
    for row_index in range(batch["input_ids"].shape[0]):
        full_positions = torch.nonzero(batch["attention_mask"][row_index]).flatten()
        prompt_positions = torch.nonzero(
            prompt_batch["attention_mask"][row_index]
        ).flatten()
        full_ids = batch["input_ids"][row_index, full_positions]
        prompt_ids = prompt_batch["input_ids"][row_index, prompt_positions]
        limit = min(len(full_ids), len(prompt_ids))
        mismatch = torch.nonzero(full_ids[:limit] != prompt_ids[:limit]).flatten()
        common = int(mismatch[0]) if len(mismatch) else limit
        if common < len(prompt_ids) - 2 or common >= len(full_ids):
            raise ValueError(
                "chat suffix alignment failed: "
                f"common={common} prompt={len(prompt_ids)} full={len(full_ids)}"
            )
        answer_positions = full_positions[common:]
        labels[row_index, answer_positions] = batch["input_ids"][
            row_index, answer_positions
        ]
    return labels


def batch_loss(
    model: Any,
    processor: Any,
    items: list[dict[str, Any]],
    image_cache: Path,
    zero_token: int,
    one_token: int,
    architecture: str,
):
    import torch
    from torch.nn import functional

    rows = [SimpleNamespace(**item) for item in items]
    images = [
        None
        if item.get("synthetic")
        else open_real_image(image_cache, item, architecture)
        for item in items
    ]
    try:
        prompts = [
            conversation(row, image)
            for row, image in zip(rows, images, strict=True)
        ]
        if architecture == "qwen3vl_2b":
            full = [
                conversation(row, image, str(int(row.label)))
                for row, image in zip(rows, images, strict=True)
            ]
            batch = processor_batch(
                processor, full, add_generation_prompt=False
            )
            prompt_batch = processor_batch(
                processor, prompts, add_generation_prompt=True
            )
            batch["labels"] = assistant_suffix_labels(batch, prompt_batch)
            batch = batch.to(model.device)
            return model(**batch, use_cache=False).loss
        batch = processor_batch(
            processor, prompts, add_generation_prompt=True
        ).to(model.device)
        outputs = model(**batch, use_cache=False)
        positions = torch.arange(batch["attention_mask"].shape[1], device=model.device)[None, :]
        last = torch.where(batch["attention_mask"].bool(), positions, -1).max(dim=1).values
        indices = torch.arange(len(items), device=model.device)
        logits = outputs.logits[indices, last]
        binary = logits[:, one_token].float() - logits[:, zero_token].float()
        labels = torch.tensor(
            [int(item["label"]) for item in items],
            dtype=torch.float32,
            device=model.device,
        )
        return functional.binary_cross_entropy_with_logits(binary, labels)
    finally:
        for image in images:
            if image is not None:
                image.close()


def predict(
    model: Any,
    processor: Any,
    items: list[dict[str, Any]],
    image_cache: Path,
    zero_token: int,
    one_token: int,
    architecture: str,
) -> list[dict[str, Any]]:
    import torch

    model.eval()
    output = []
    validation_batch = 8 if architecture == "qwen3vl_2b" else 1
    for start in range(0, len(items), validation_batch):
        local = items[start : start + validation_batch]
        rows = [SimpleNamespace(**item) for item in local]
        images = [open_real_image(image_cache, item, architecture) for item in local]
        try:
            batch = processor_batch(
                processor,
                [
                    conversation(row, image)
                    for row, image in zip(rows, images, strict=True)
                ],
                add_generation_prompt=True,
            ).to(model.device)
            with torch.inference_mode():
                raw = model(**batch, use_cache=True).logits
            positions = torch.arange(
                batch["attention_mask"].shape[1], device=model.device
            )[None, :]
            last = torch.where(batch["attention_mask"].bool(), positions, -1).max(
                dim=1
            ).values
            logits = raw[torch.arange(len(local), device=model.device), last]
            scores = (logits[:, one_token] - logits[:, zero_token]).float().cpu()
        finally:
            for image in images:
                image.close()
        for item, score_tensor in zip(local, scores, strict=True):
            score = float(score_tensor)
            output.append(
                {
                    "global_index": int(item["global_index"]),
                    "id": str(item["id"]),
                    "fold": int(item["fold"]),
                    "category": str(item["category"]),
                    "score": score,
                    "prediction": int(score >= 0.0),
                }
            )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor

    contract = MODEL_CONTRACTS[args.architecture]
    run_started = time.monotonic()
    if args.model_revision != contract["revision"]:
        raise ValueError("model revision mismatch")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite output")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, runtime_audit = load_runtime(
        args.runtime_dir, args.fold, refit=args.refit
    )
    cache_audit = validate_image_cache(
        args.images, train + validation, args.architecture
    )
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    sys.path.insert(0, str(args.vendor.resolve()))
    import peft

    if peft.__version__ != "0.20.0":
        raise ValueError("PEFT 0.20.0 required")
    from peft import LoraConfig, TaskType, get_peft_model

    processor_kwargs = {"local_files_only": True, "trust_remote_code": True}
    if args.architecture == "qwen3vl_2b":
        processor_kwargs.update(min_pixels=4 * 28 * 28, max_pixels=262144)
    processor = AutoProcessor.from_pretrained(
        args.model_root.resolve(), **processor_kwargs
    )
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1:
        raise ValueError("0/1 tokens must be atomic")
    loader = (
        AutoModelForMultimodalLM
        if contract["loader"] == "multimodal"
        else AutoModelForImageTextToText
    )
    model = loader.from_pretrained(
        args.model_root.resolve(),
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)

    # New preprocessing proof: two synthetic text-only rows, real CUDA forward/backward,
    # finite loss/gradients, and no optimizer step. Gradients are discarded before science.
    synthetic_rows = [row for row in train if row.get("synthetic")][:2]
    if len(synthetic_rows) != 2:
        raise ValueError("technical preflight requires two synthetic rows")
    smoke_loss = batch_loss(
        model,
        processor,
        synthetic_rows,
        args.images,
        zero[0],
        one[0],
        args.architecture,
    )
    if not torch.isfinite(smoke_loss):
        raise ValueError("non-finite technical smoke loss")
    smoke_loss.backward()
    finite_gradients = all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in trainable
        if parameter.grad is not None
    )
    if not finite_gradients:
        raise ValueError("non-finite technical smoke gradient")
    smoke_grad_norm = float(
        torch.sqrt(
            sum(
                parameter.grad.detach().float().pow(2).sum()
                for parameter in trainable
                if parameter.grad is not None
            )
        ).cpu()
    )
    optimizer.zero_grad(set_to_none=True)

    if args.technical_smoke_only:
        report = {
            "schema": "exp699_remote_compute_technical_smoke_v1",
            "experiment_id": "699",
            "architecture": args.architecture,
            "model_id": contract["model_id"],
            "model_revision": contract["revision"],
            "fold": args.fold,
            "runtime_contract_sha256": runtime_audit["contract_sha256"],
            "source": runtime_audit["source"],
            "mode": runtime_audit["mode"],
            "cap": runtime_audit["cap"],
            "rows": 2,
            "optimizer_steps": 0,
            "loss": float(smoke_loss.detach().cpu()),
            "grad_norm": smoke_grad_norm,
            "finite": True,
            "text_only_synthetic": True,
            "validation_labels_read": 0,
            "sealed_rows_used": 0,
            "public_rows_used": 0,
            "runtime_minutes": (time.monotonic() - run_started) / 60,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "packages": {
                "torch": importlib.metadata.version("torch"),
                "transformers": importlib.metadata.version("transformers"),
                "peft": importlib.metadata.version("peft"),
            },
            "decision": "ACCEPT_REMOTE_COMPUTE_TECHNICAL_SMOKE",
        }
        report["self_sha256"] = canonical_sha256(report)
        (args.output_dir / "technical_smoke.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report

    micro = int(contract["micro_batch"])
    accumulation = int(contract["accumulation"])
    shape = training_shape(len(train), micro, accumulation, args.epochs)
    updates = shape["optimizer_steps"]
    expected_updates = int(runtime_audit.get("expected_optimizer_steps", updates))
    if runtime_audit.get("epochs", 1) != args.epochs or updates != expected_updates:
        raise ValueError("runtime optimizer-step contract mismatch")
    warmup = max(1, int(updates * 0.05))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    started = time.monotonic()
    optimizer_steps = 0
    running_loss = 0.0
    training_batches = 0
    last_grad_norm = float("nan")
    model.train()
    for epoch in range(args.epochs):
        indices = list(range(len(train)))
        random.Random(SEED + epoch).shuffle(indices)
        for step, offset in enumerate(range(0, len(indices), micro), 1):
            items = [train[index] for index in indices[offset : offset + micro]]
            loss = batch_loss(
                model,
                processor,
                items,
                args.images,
                zero[0],
                one[0],
                args.architecture,
            )
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            (loss / accumulation).backward()
            running_loss += float(loss.detach().cpu())
            training_batches += 1
            if step % accumulation == 0 or offset + micro >= len(indices):
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if not torch.isfinite(grad_norm):
                    raise ValueError("non-finite training gradient norm")
                last_grad_norm = float(grad_norm.detach().cpu())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if optimizer_steps % 25 == 0 or optimizer_steps == updates:
                    elapsed_seconds = time.monotonic() - started
                    processed_rows = epoch * len(train) + min(offset + micro, len(indices))
                    total_rows = args.epochs * len(train)
                    throughput = processed_rows / max(elapsed_seconds, 1e-9)
                    print(
                        json.dumps(
                            {
                                "phase": "train",
                                "architecture": args.architecture,
                                "fold": args.fold,
                                "epoch": epoch + 1,
                                "epochs": args.epochs,
                                "update": optimizer_steps,
                                "updates": updates,
                                "mean_loss": running_loss / training_batches,
                                "lr": scheduler.get_last_lr()[0],
                                "grad_norm": last_grad_norm,
                                "throughput_rows_per_second": throughput,
                                "eta_seconds": max(0, total_rows - processed_rows)
                                / max(throughput, 1e-9),
                                "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
                            }
                        ),
                        flush=True,
                    )
    if optimizer_steps != updates:
        raise RuntimeError("optimizer update count mismatch")
    training_runtime_seconds = time.monotonic() - started
    training_throughput = (
        len(train) * args.epochs / max(training_runtime_seconds, 1e-9)
    )
    model.config.use_cache = True
    # Persist the trained adapter before held-out inference. This does not change
    # training or scoring, and prevents an image/cache transport failure during
    # validation from discarding a completed hour-long fit.
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    shutil.make_archive(str(args.output_dir / "adapter"), "zip", adapter_dir)
    adapter_path = args.output_dir / "adapter.zip"
    predictions = predict(
        model,
        processor,
        validation,
        args.images,
        zero[0],
        one[0],
        args.architecture,
    )
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "699",
        "architecture": args.architecture,
        "model_id": contract["model_id"],
        "model_revision": contract["revision"],
        "fold": args.fold,
        "runtime_contract_sha256": runtime_audit["contract_sha256"],
        "source": runtime_audit["source"],
        "mode": runtime_audit["mode"],
        "cap": runtime_audit["cap"],
        "augmentation_arm": runtime_audit.get("augmentation_arm"),
        "image_policy": runtime_audit["image_policy"],
        "image_cache_audit": cache_audit,
        "seed": SEED,
        "epochs": args.epochs,
        "learning_rate": LEARNING_RATE,
        "micro_batch": micro,
        "gradient_accumulation": accumulation,
        "effective_batch": micro * accumulation,
        "train_occurrences": len(train),
        "training_occurrences_seen": shape["training_occurrences_seen"],
        "synthetic_occurrences": sum(bool(row.get("synthetic")) for row in train),
        "optimizer_steps": optimizer_steps,
        "checkpoint_selection": (
            f"fixed_final_full_refit_after_{args.epochs}_epoch"
            f"{'s' if args.epochs != 1 else ''}_no_validation_selection"
            if args.refit
            else (
                "fixed_final_after_one_epoch_no_validation_selection"
                if args.epochs == 1
                else "fixed_final_after_two_epochs_no_validation_selection"
            )
        ),
        "last_grad_norm": last_grad_norm,
        "training_runtime_minutes": training_runtime_seconds / 60,
        "training_throughput_rows_per_second": training_throughput,
        "technical_smoke": {
            "rows": 2,
            "optimizer_steps": 0,
            "loss": float(smoke_loss.detach().cpu()),
            "grad_norm": smoke_grad_norm,
            "finite": True,
            "text_only": True,
        },
        "validation_rows": len(validation),
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_rows_used": 0,
        "threshold": 0.0,
        "threshold_tuned": False,
        "loss_contract": (
            "assistant_suffix_lm"
            if args.architecture == "qwen3vl_2b"
            else "binary_bce_last_token"
        ),
        "first_image_max_edge": 448 if args.architecture == "qwen3vl_2b" else None,
        "first_image_max_pixels": 262144,
        "runtime_minutes": (time.monotonic() - started) / 60,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "packages": {
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
        },
        "artifacts": {
            "predictions.jsonl": sha256_file(predictions_path),
            "adapter.zip": sha256_file(adapter_path),
        },
        "decision": "GO_PACKAGE" if args.refit else "GO_EVALUATE",
    }
    report["contract_sha256"] = canonical_sha256(report)
    (args.output_dir / "output_contract.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=tuple(MODEL_CONTRACTS), required=True)
    parser.add_argument("--fold", type=int, choices=range(-1, 5), required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--technical-smoke-only", action="store_true")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.refit != (args.fold == -1):
        raise ValueError("--refit requires --fold -1 and fold -1 requires --refit")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.nn import functional as F
from transformers import AutoModelForMultimodalLM, AutoProcessor

SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
sys.path.insert(0, str(SHARED))
from grid_contract import base_prompt  # noqa: E402

MODEL = Path("/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3.8-27B")
SEED = 42
MAX_LENGTH = 1536
DEFAULT_MICRO_BATCH = 2
DEFAULT_GRAD_ACCUM = 8
DEFAULT_VALIDATION_BATCH = 8
LEARNING_RATE = 2e-4
DEFAULT_CHECKPOINT_EVERY_UPDATES = 10
RESUME_SCHEMA = "exp697_training_resume_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_resume_checkpoint(
    path: Path,
    expected_contract: dict,
    *,
    micro_batch: int,
    rows: int,
) -> dict | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != RESUME_SCHEMA:
        raise ValueError("resume checkpoint schema mismatch")
    if payload.get("contract") != expected_contract:
        raise ValueError("resume checkpoint training contract mismatch")
    next_offset = int(payload.get("next_offset", -1))
    micro_step = int(payload.get("micro_step", -1))
    optimizer_steps = int(payload.get("optimizer_steps", -1))
    if not 0 <= next_offset <= rows or (
        next_offset != rows and next_offset % micro_batch != 0
    ):
        raise ValueError("resume checkpoint offset is invalid")
    if micro_step != math.ceil(next_offset / micro_batch) or optimizer_steps < 0:
        raise ValueError("resume checkpoint progress is invalid")
    required = {
        "adapter_state",
        "optimizer_state",
        "scheduler_state",
        "torch_rng_state",
        "cuda_rng_states",
        "elapsed_seconds",
    }
    if any(key not in payload for key in required):
        raise ValueError("resume checkpoint state is incomplete")
    return payload


def open_image(row: dict) -> Image.Image:
    image = Image.open(row["image_path"]).convert("RGB")
    width, height = image.size
    if width * height > 262144:
        scale = math.sqrt(262144 / (width * height))
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = image.resize(
            (max(28, int(width * scale)), max(28, int(height * scale))), resampling
        )
        image.close()
        image = resized
    return image


def conversations(rows: list[SimpleNamespace], images: list[Image.Image]) -> list[list[dict]]:
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": base_prompt(row)},
                ],
            }
        ]
        for row, image in zip(rows, images, strict=True)
    ]


def batch_inputs(processor, rows, images):
    return processor.apply_chat_template(
        conversations(rows, images),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        enable_thinking=False,
    )


def scores(model, batch, zero_token: int, one_token: int):
    batch = batch.to(model.device)
    outputs = model(**batch, use_cache=False)
    positions = torch.arange(batch["attention_mask"].shape[1], device=model.device)[None, :]
    last = torch.where(batch["attention_mask"].bool(), positions, -1).max(dim=1).values
    rows = torch.arange(batch["attention_mask"].shape[0], device=model.device)
    logits = outputs.logits[rows, last]
    return logits[:, one_token] - logits[:, zero_token]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--micro-batch", type=int, default=DEFAULT_MICRO_BATCH)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument(
        "--validation-batch", type=int, default=DEFAULT_VALIDATION_BATCH
    )
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_UPDATES,
    )
    args = parser.parse_args()
    if (
        args.micro_batch < 1
        or args.grad_accum < 1
        or args.validation_batch < 1
        or args.checkpoint_every_updates < 0
    ):
        raise ValueError("batch, accumulation and checkpoint interval are invalid")
    checkpoint_path = args.checkpoint_path or args.output_dir.with_suffix(".resume.pt")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        partial_names = {path.name for path in args.output_dir.iterdir()}
        recoverable = checkpoint_path.is_file() and partial_names <= {
            "adapter",
            "predictions.jsonl",
        }
        if not recoverable:
            raise FileExistsError("refusing to overwrite non-empty output")
        print(
            json.dumps(
                {
                    "recovering_partial_output": sorted(partial_names),
                    "checkpoint": str(checkpoint_path),
                }
            ),
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.runtime_dir / "runtime_audit.json").read_text())
    if audit["fold"] != args.fold or audit["experiment_id"] != "697":
        raise ValueError("runtime/fold mismatch")
    train_rows = read_jsonl(args.runtime_dir / "train.jsonl")
    validation_rows = read_jsonl(args.runtime_dir / "validation.jsonl")
    if args.technical_smoke:
        train_rows, validation_rows = train_rows[:8], validation_rows[:2]
    checkpoint_every_updates = (
        0 if args.technical_smoke else args.checkpoint_every_updates
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1:
        raise ValueError("0/1 are not atomic tokens")

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map="balanced",
        max_memory={index: "70GiB" for index in range(torch.cuda.device_count())},
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    micro_batches = math.ceil(len(train_rows) / args.micro_batch)
    updates = math.ceil(micro_batches / args.grad_accum)
    warmup = max(1, int(updates * 0.05))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    indices = list(range(len(train_rows)))
    random.Random(SEED).shuffle(indices)
    indices_sha256 = hashlib.sha256(
        np.asarray(indices, dtype="<i8").tobytes()
    ).hexdigest()
    resume_contract = {
        "experiment_id": "697",
        "fold": args.fold,
        "model": str(MODEL),
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "technical_smoke": args.technical_smoke,
        "runtime_audit_sha256": sha256(args.runtime_dir / "runtime_audit.json"),
        "train_sha256": sha256(args.runtime_dir / "train.jsonl"),
        "train_occurrences": len(train_rows),
        "indices_sha256": indices_sha256,
        "micro_batch": args.micro_batch,
        "gradient_accumulation": args.grad_accum,
        "learning_rate": LEARNING_RATE,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "scheduler": "cosine_5pct_warmup",
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "use_rslora": True,
        },
        "visible_cuda_devices": torch.cuda.device_count(),
    }
    resume = load_resume_checkpoint(
        checkpoint_path,
        resume_contract,
        micro_batch=args.micro_batch,
        rows=len(indices),
    )
    start_offset = 0
    optimizer_steps = 0
    elapsed_before_resume = 0.0
    resumed_from_checkpoint = resume is not None
    if resume is not None:
        set_peft_model_state_dict(model, resume["adapter_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        torch.set_rng_state(resume["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
        start_offset = int(resume["next_offset"])
        optimizer_steps = int(resume["optimizer_steps"])
        elapsed_before_resume = float(resume["elapsed_seconds"])
        print(
            json.dumps(
                {
                    "resume_checkpoint": str(checkpoint_path),
                    "processed": start_offset,
                    "optimizer_steps": optimizer_steps,
                }
            ),
            flush=True,
        )
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic() - elapsed_before_resume
    loss_window: list[float] = []
    initial_micro_step = start_offset // args.micro_batch
    for step, offset in enumerate(
        range(start_offset, len(indices), args.micro_batch),
        start=initial_micro_step + 1,
    ):
        local = [train_rows[index] for index in indices[offset : offset + args.micro_batch]]
        rows = [SimpleNamespace(**item) for item in local]
        images = [open_image(item) for item in local]
        try:
            score = scores(model, batch_inputs(processor, rows, images), zero[0], one[0])
            label = torch.tensor(
                [item["label"] for item in local], dtype=torch.float32, device=score.device
            )
            raw_loss = F.binary_cross_entropy_with_logits(score.float(), label)
            loss_window.append(float(raw_loss.detach().cpu()))
            (raw_loss / args.grad_accum).backward()
        finally:
            for image in images:
                image.close()
        did_update = (
            step % args.grad_accum == 0
            or offset + args.micro_batch >= len(indices)
        )
        if did_update:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        processed = min(offset + args.micro_batch, len(indices))
        if (
            did_update
            and checkpoint_every_updates > 0
            and optimizer_steps % checkpoint_every_updates == 0
        ):
            atomic_torch_save(
                {
                    "schema_version": RESUME_SCHEMA,
                    "contract": resume_contract,
                    "next_offset": processed,
                    "micro_step": step,
                    "optimizer_steps": optimizer_steps,
                    "adapter_state": {
                        key: value.detach().cpu()
                        for key, value in get_peft_model_state_dict(model).items()
                    },
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all(),
                    "elapsed_seconds": time.monotonic() - started,
                },
                checkpoint_path,
            )
            print(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint_path),
                        "processed": processed,
                        "optimizer_steps": optimizer_steps,
                    }
                ),
                flush=True,
            )
        if step % 25 == 0 or processed == len(indices):
            print(
                json.dumps(
                    {
                        "micro_step": step,
                        "processed": processed,
                        "rows": len(indices),
                        "optimizer_steps": optimizer_steps,
                        "mean_loss": round(float(np.mean(loss_window)), 6),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_min": round((time.monotonic() - started) / 60, 2),
                    }
                ),
                flush=True,
            )
            loss_window.clear()

    # Publish the trained adapter before the long outer-fold inference.  The
    # resume checkpoint intentionally remains until the complete output
    # contract is written, so a validation failure can recover without
    # repeating the full epoch.
    adapter = args.output_dir / "adapter"
    model.save_pretrained(adapter)

    model.eval()
    predictions = []
    validation_batch = args.validation_batch
    with torch.inference_mode():
        offset = 0
        while offset < len(validation_rows):
            local = validation_rows[offset : offset + validation_batch]
            rows = [SimpleNamespace(**item) for item in local]
            images = [open_image(item) for item in local]
            try:
                values = scores(
                    model,
                    batch_inputs(processor, rows, images),
                    zero[0],
                    one[0],
                )
            except torch.cuda.OutOfMemoryError:
                if validation_batch == 1:
                    raise
                validation_batch = max(1, validation_batch // 2)
                torch.cuda.empty_cache()
                print(
                    json.dumps(
                        {
                            "validation_batch_fallback": validation_batch,
                            "retry_offset": offset,
                        }
                    ),
                    flush=True,
                )
                continue
            finally:
                for image in images:
                    image.close()
            predictions.extend(
                {"id": item["id"], "fold": args.fold, "score": float(value)}
                for item, value in zip(local, values.float().cpu(), strict=True)
            )
            offset += len(local)
            if offset % 200 == 0 or offset == len(validation_rows):
                print(f"predicted={offset}/{len(validation_rows)}", flush=True)

    pred_path = args.output_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "experiment_id": "697",
        "fold": args.fold,
        "model": str(MODEL),
        "offline_teacher_only": True,
        "train_occurrences": len(train_rows),
        "validation_rows": len(validation_rows),
        "optimizer_steps": optimizer_steps,
        "micro_batch": args.micro_batch,
        "gradient_accumulation": args.grad_accum,
        "effective_batch": args.micro_batch * args.grad_accum,
        "validation_batch_requested": args.validation_batch,
        "validation_batch_final": validation_batch,
        "checkpoint_every_updates": checkpoint_every_updates,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "runtime_minutes": (time.monotonic() - started) / 60,
        "technical_smoke": args.technical_smoke,
        "predictions_sha256": sha256(pred_path),
    }
    (args.output_dir / "output_contract.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    try:
        checkpoint_path.unlink(missing_ok=True)
    except OSError as error:
        print(f"checkpoint_cleanup_warning={error}", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

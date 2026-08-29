from __future__ import annotations

import argparse
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

SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from grid_contract import CELL_SPECS, canonical_sha256, sha256_file
from train_lora import LEARNING_RATE, SEED, open_image, predict_class_only, primary_loss

EXPERIMENT_ID = "677"
SOURCE_SPEC_ID = "641"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_runtime(path: Path, inner_fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    audit_path = path / "runtime_audit.json"
    train_path = path / "train.jsonl"
    validation_path = path / "validation.jsonl"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payload = dict(audit)
    digest = payload.pop("contract_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("runtime audit self-hash mismatch")
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "objective": "class_only_checkpoint_dynamics",
        "outer_screen_fold": 0,
        "inner_validation_fold": inner_fold,
        "blind_confirmation_folds": [0],
        "fold3_is_blind": False,
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "public_used": False,
        "decision": "READY_FOR_TECHNICAL_SMOKE_AFTER_TERMINAL_659",
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("runtime audit contract mismatch")
    if audit["output_sha256"] != {
        "train.jsonl": sha256_file(train_path),
        "validation.jsonl": sha256_file(validation_path),
    }:
        raise ValueError("runtime payload checksum mismatch")
    train, validation = read_jsonl(train_path), read_jsonl(validation_path)
    if (
        len(train) != audit["train_occurrences"]
        or len(validation) != audit["validation_rows"]
        or len(train) % 4
    ):
        raise ValueError("nested train size does not preserve frozen micro-batch parity")
    if any(int(row["fold"]) in {0, inner_fold} for row in train):
        raise ValueError("outer or inner validation entered training")
    if any(int(row["fold"]) != inner_fold for row in validation):
        raise ValueError("inner validation fold mismatch")
    if any("label" in row or "evidence_target" in row for row in validation):
        raise ValueError("validation supervision is forbidden")
    return train, validation, audit


def checkpoint_steps(updates: int) -> list[int]:
    if updates < 4:
        return sorted({max(1, round(updates * fraction)) for fraction in CHECKPOINT_FRACTIONS})
    steps = [round(updates * fraction) for fraction in CHECKPOINT_FRACTIONS]
    if len(set(steps)) != len(steps) or steps[-1] != updates:
        raise ValueError("checkpoint schedule is not distinct or does not end at final update")
    return steps


def save_checkpoint(
    *,
    model: Any,
    processor: Any,
    validation_rows: list[dict[str, Any]],
    images: Path,
    spec: Any,
    zero_token: int,
    one_token: int,
    output_dir: Path,
    optimizer_step: int,
    total_updates: int,
    requested_fraction: float,
) -> dict[str, Any]:
    checkpoint_dir = output_dir / f"step_{optimizer_step:04d}"
    adapter_dir = checkpoint_dir / "adapter"
    checkpoint_dir.mkdir(parents=True)
    model.eval()
    model.config.use_cache = True
    model.save_pretrained(adapter_dir)
    records = predict_class_only(
        model, processor, validation_rows, images, spec, zero_token, one_token
    )
    for row in records:
        row["dynamics_experiment_id"] = EXPERIMENT_ID
        row["optimizer_step"] = optimizer_step
        row["training_fraction"] = optimizer_step / total_updates
        row["requested_training_fraction"] = requested_fraction
    predictions_path = checkpoint_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    archive = Path(shutil.make_archive(str(checkpoint_dir / "adapter"), "zip", adapter_dir))
    model.config.use_cache = False
    model.train()
    return {
        "optimizer_step": optimizer_step,
        "requested_training_fraction": requested_fraction,
        "training_fraction": optimizer_step / total_updates,
        "predictions": str(predictions_path.relative_to(output_dir)),
        "predictions_sha256": sha256_file(predictions_path),
        "adapter_archive": str(archive.relative_to(output_dir)),
        "adapter_archive_sha256": sha256_file(archive),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if args.model_revision != MODEL_REVISION:
        raise ValueError("model revision differs from the frozen 4B contract")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, validation_rows, runtime_audit = load_runtime(args.runtime_dir, args.inner_fold)
    if args.technical_smoke:
        train_rows = train_rows[:32]
        validation_rows = validation_rows[:4]
    if len(train_rows) % MICRO_BATCH_SIZE:
        raise ValueError("runtime must contain complete two-row micro-batches")
    micro_batches = len(train_rows) // MICRO_BATCH_SIZE
    updates = math.ceil(micro_batches / GRADIENT_ACCUMULATION)
    planned_checkpoints = checkpoint_steps(updates)
    requested_fractions = (
        CHECKPOINT_FRACTIONS
        if len(planned_checkpoints) == len(CHECKPOINT_FRACTIONS)
        else tuple(step / updates for step in planned_checkpoints)
    )
    requested_fraction_by_step = {
        step: fraction for step, fraction in zip(planned_checkpoints, requested_fractions, strict=True)
    }
    warmup = max(1, int(updates * 0.05))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.device_count() < 1:
        raise RuntimeError("one CUDA device is required")
    if not args.vendor.is_dir():
        raise FileNotFoundError("vendored PEFT directory is missing")
    sys.path.insert(0, str(args.vendor.resolve()))
    import peft

    if peft.__version__ != "0.20.0":
        raise RuntimeError("exact vendored PEFT 0.20.0 is required")
    from peft import LoraConfig, TaskType, get_peft_model

    spec = CELL_SPECS[SOURCE_SPEC_ID]
    if LEARNING_RATE != 2e-4 or SEED != 42 or spec.model_revision != MODEL_REVISION:
        raise RuntimeError("source 4B scientific recipe drifted")
    processor = AutoProcessor.from_pretrained(
        args.model_root.resolve(), local_files_only=True, trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1 or zero == one:
        raise RuntimeError("0 and 1 must be distinct atomic tokens")
    started = time.monotonic()
    model = AutoModelForMultimodalLM.from_pretrained(
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
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.device.type != "cuda" for parameter in trainable):
        raise RuntimeError("trainable adapter parameters are not fully on CUDA")
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    optimizer.zero_grad(set_to_none=True)
    indices = list(range(len(train_rows)))
    random.Random(SEED).shuffle(indices)
    optimizer_step = 0
    micro_step = 0
    local_losses: list[float] = []
    dynamics: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for offset in range(0, len(indices), MICRO_BATCH_SIZE):
        local = [train_rows[index] for index in indices[offset : offset + MICRO_BATCH_SIZE]]
        rows = [SimpleNamespace(**row) for row in local]
        images = [open_image(args.images, row) for row in local]
        try:
            loss = primary_loss(model, processor, rows, images, zero[0], one[0])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite training loss")
            local_losses.append(float(loss.detach().float().cpu()))
            (loss / GRADIENT_ACCUMULATION).backward()
        finally:
            for image in images:
                image.close()
        micro_step += 1
        update_boundary = micro_step % GRADIENT_ACCUMULATION == 0 or offset + MICRO_BATCH_SIZE >= len(
            indices
        )
        if not update_boundary:
            continue
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        ):
            raise RuntimeError("non-finite adapter gradients")
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, foreach=False)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        dynamics.append(
            {
                "optimizer_step": optimizer_step,
                "mean_micro_batch_loss": sum(local_losses) / len(local_losses),
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm_before_clip": float(gradient_norm.float().cpu()),
                "micro_batches_in_update": len(local_losses),
            }
        )
        local_losses.clear()
        if optimizer_step in planned_checkpoints:
            checkpoints.append(
                save_checkpoint(
                    model=model,
                    processor=processor,
                    validation_rows=validation_rows,
                    images=args.images,
                    spec=spec,
                    zero_token=zero[0],
                    one_token=one[0],
                    output_dir=args.output_dir,
                    optimizer_step=optimizer_step,
                    total_updates=updates,
                    requested_fraction=requested_fraction_by_step[optimizer_step],
                )
            )
    if optimizer_step != updates or [row["optimizer_step"] for row in checkpoints] != planned_checkpoints:
        raise RuntimeError("optimizer or checkpoint schedule drift")
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_SPEC_ID,
        "outer_screen_fold": 0,
        "inner_validation_fold": args.inner_fold,
        "blind_confirmation_folds": [0],
        "fold3_is_blind": False,
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "objective": "class_only_binary_bce_checkpoint_dynamics",
        "changed_factor": "optimizer_stop_fraction_only",
        "seed": SEED,
        "train_occurrences": len(train_rows),
        "validation_rows": len(validation_rows),
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch_size": MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION,
        "optimizer_steps": updates,
        "checkpoint_fractions": requested_fractions,
        "checkpoint_steps": planned_checkpoints,
        "warmup_steps": warmup,
        "learning_rate": LEARNING_RATE,
        "scheduler": "cosine_to_zero",
        "runtime_contract_sha256": runtime_audit["contract_sha256"],
        "technical_smoke": bool(args.technical_smoke),
        "validation_labels_read": 0,
        "sealed_rows": 0,
        "public_used": False,
        "threshold_tuned": False,
        "dynamics": dynamics,
        "checkpoints": checkpoints,
        "packages": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "peft": peft.__version__,
        },
        "elapsed_seconds": time.monotonic() - started,
        "decision": "TECHNICAL_SMOKE_ONLY" if args.technical_smoke else "READY_FOR_INNER_EVALUATION",
    }
    report["contract_sha256"] = canonical_sha256(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--inner-fold", type=int, choices=(1, 2, 3, 4), required=True)
    result.add_argument("--runtime-dir", type=Path, required=True)
    result.add_argument("--images", type=Path, required=True)
    result.add_argument("--model-root", type=Path, required=True)
    result.add_argument("--model-revision", default=MODEL_REVISION)
    result.add_argument("--vendor", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--technical-smoke", action="store_true")
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2, sort_keys=True))

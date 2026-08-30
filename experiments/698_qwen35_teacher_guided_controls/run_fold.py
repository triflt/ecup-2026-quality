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
from torch.nn import functional as F

SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
sys.path.append(str(SHARED))
base_prompt = None

EXPERIMENT_ID = "698"
SOURCE_EXPERIMENT_ID = "697"
MODEL = Path("/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3.5-4B")
TEACHER_MODEL = Path("/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3.8-27B")
FLAMMABLE = "Легковоспламеняющиеся"
MODES = ("gold_control", "hardneg_candidate", "rank_candidate")
SEED = 42
MAX_LENGTH = 1536
MICRO_BATCH = 2
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
HARD_WEIGHT_MAX = 2.0
RANK_COEFFICIENT = 0.5
IMAGE_PREPROCESSING = "solution140_first_image_thumbnail_448_lanczos_v1"
SAMPLER = "exp697_label_only_v1"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MATCHED_TRAIN_OCCURRENCES = 5390
DEFAULT_CHECKPOINT_EVERY_UPDATES = 10
RESUME_SCHEMA = "exp698_training_resume_v1"


def common_training_contract(occurrence_order_sha256: str) -> dict:
    return {
        "seed": SEED,
        "sampler": SAMPLER,
        "epochs": 1,
        "max_length": MAX_LENGTH,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": GRAD_ACCUM,
        "effective_batch": MICRO_BATCH * GRAD_ACCUM,
        "occurrence_order": {
            "rule": "python_random_seeded_shuffle_of_occurrence_indices",
            "sha256": occurrence_order_sha256,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.01,
            "clip_grad_norm": 1.0,
        },
        "scheduler": {"name": "cosine", "warmup_fraction": 0.05},
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": LORA_TARGET_MODULES,
            "bias": "none",
            "use_rslora": True,
        },
        "model_dtype": "bfloat16",
        "attention_implementation": "eager",
        "image_preprocessing": IMAGE_PREPROCESSING,
    }


def objective_contract(mode: str) -> dict:
    if mode == "gold_control":
        return {"name": "hard_gold_bce", "teacher_target_consumed": False}
    if mode == "hardneg_candidate":
        return {
            "name": "flammable_teacher_disagreement_weighted_hard_gold_bce",
            "teacher_target_consumed": True,
            "hard_weight_max": HARD_WEIGHT_MAX,
            "teacher_verdict_replaces_gold": False,
        }
    if mode == "rank_candidate":
        return {
            "name": "flammable_same_label_within_batch_teacher_rank_plus_hard_gold_bce",
            "teacher_target_consumed": True,
            "rank_coefficient": RANK_COEFFICIENT,
            "opposite_label_pairs_allowed": False,
            "teacher_verdict_replaces_gold": False,
        }
    raise ValueError(f"unknown training mode: {mode}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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


def load_resume_checkpoint(path: Path, expected_contract: dict, rows: int) -> dict | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != RESUME_SCHEMA:
        raise ValueError("student resume checkpoint schema mismatch")
    if payload.get("contract") != expected_contract:
        raise ValueError("student resume checkpoint contract mismatch")
    next_offset = int(payload.get("next_offset", -1))
    if not 0 <= next_offset <= rows or (
        next_offset != rows and next_offset % MICRO_BATCH != 0
    ):
        raise ValueError("student resume checkpoint offset is invalid")
    required = {
        "adapter_state",
        "optimizer_state",
        "scheduler_state",
        "torch_rng_state",
        "cuda_rng_states",
        "optimizer_steps",
        "weighted_occurrences",
        "rank_pairs",
        "elapsed_seconds",
    }
    if any(key not in payload for key in required):
        raise ValueError("student resume checkpoint state is incomplete")
    return payload


def open_image(row: dict) -> Image.Image:
    image = Image.open(row["image_path"]).convert("RGB")
    image.thumbnail((448, 448), Image.Resampling.LANCZOS)
    return image


def conversations(rows: list[SimpleNamespace], images: list[Image.Image]) -> list[list[dict]]:
    if base_prompt is None:
        raise RuntimeError("grid prompt contract is not loaded")
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


def load_teacher_targets(
    target_dir: Path,
    train_rows: list[dict],
    fold: int,
    *,
    runtime_audit_sha256: str,
    train_runtime_sha256: str,
) -> tuple[list[float], dict]:
    target_path = target_dir / "teacher_targets.jsonl"
    contract_path = target_dir / "teacher_target_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_contract = {
        "schema_version": "exp697_teacher_targets_v1",
        "experiment_id": SOURCE_EXPERIMENT_ID,
        "fold": fold,
        "model": str(TEACHER_MODEL),
        "outer_safe_validation_excluded": True,
        "target_scope": "outer_train_occurrences",
        "targets_are_in_sample_within_outer_train": True,
        "teacher_frozen_before_target_generation": True,
        "atomic_directory_publish": True,
        "rows": len(train_rows),
        "unique_ids": len({str(row["id"]) for row in train_rows}),
        "inference_scope": "unique_consumed_flammable_ids_only",
        "teacher_signal_consumed_categories": [FLAMMABLE],
        "neutral_score_for_unconsumed_categories": 0.0,
        "inference_occurrences": sum(
            str(row["category"]) == FLAMMABLE for row in train_rows
        ),
        "inference_unique_ids": len(
            {
                str(row["id"])
                for row in train_rows
                if str(row["category"]) == FLAMMABLE
            }
        ),
        "deduplicated_by_id": True,
        "runtime_audit_sha256": runtime_audit_sha256,
        "train_runtime_sha256": train_runtime_sha256,
        "teacher_targets_sha256": sha256(target_path),
    }
    mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected_contract.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"teacher target contract mismatch: {mismatch}")
    adapter_sha256 = str(contract.get("adapter_model_sha256", ""))
    if len(adapter_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in adapter_sha256
    ):
        raise ValueError("teacher adapter checksum is invalid")
    targets = read_jsonl(target_path)
    if len(targets) != len(train_rows):
        raise ValueError("teacher target occurrence count mismatch")
    values: list[float] = []
    for occurrence, (row, target) in enumerate(zip(train_rows, targets, strict=True)):
        expected = {
            "occurrence_index": occurrence,
            "id": str(row["id"]),
            "outer_fold": fold,
            "source_fold": int(row["fold"]),
            "category": str(row["category"]),
            "label": int(row["label"]),
        }
        if set(target) != {*expected, "score"} or any(
            target.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("teacher target occurrence binding mismatch")
        value = float(target["score"])
        if not math.isfinite(value):
            raise ValueError("teacher score is non-finite")
        if str(row["category"]) != FLAMMABLE and value != 0.0:
            raise ValueError("unconsumed teacher category must have neutral score")
        values.append(value)
    return values, {
        "teacher_target_contract_sha256": sha256(contract_path),
        "teacher_targets_sha256": sha256(target_path),
        "adapter_output_contract_sha256": contract["adapter_output_contract_sha256"],
        "adapter_model_sha256": contract["adapter_model_sha256"],
        "outer_safe_validation_excluded": True,
        "targets_are_in_sample_within_outer_train": True,
    }


def frozen_order(length: int) -> list[int]:
    indices = list(range(length))
    random.Random(SEED).shuffle(indices)
    return indices


def hard_example_weight(row: dict, teacher_score: float) -> float:
    if str(row["category"]) != FLAMMABLE:
        return 1.0
    label = int(row["label"])
    disagrees = (teacher_score >= 0.0) != bool(label)
    if not disagrees:
        return 1.0
    confidence = min(abs(teacher_score), 3.0) / 3.0
    return 1.0 + confidence * (HARD_WEIGHT_MAX - 1.0)


def within_stratum_rank_loss(
    student_scores: torch.Tensor,
    rows: list[dict],
    teacher_scores: list[float],
) -> tuple[torch.Tensor, int]:
    losses = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if str(rows[left]["category"]) != FLAMMABLE:
                continue
            if str(rows[right]["category"]) != FLAMMABLE:
                continue
            if int(rows[left]["label"]) != int(rows[right]["label"]):
                continue
            delta = teacher_scores[left] - teacher_scores[right]
            if delta == 0.0:
                continue
            sign = 1.0 if delta > 0 else -1.0
            losses.append(F.softplus(-sign * (student_scores[left] - student_scores[right])))
    if not losses:
        return student_scores.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def smoke_indices(
    mode: str, rows: list[dict], teacher_scores: list[float] | None
) -> list[int]:
    selected: list[int] = []
    if mode == "hardneg_candidate":
        assert teacher_scores is not None
        selected = [
            index
            for index, (row, score) in enumerate(zip(rows, teacher_scores, strict=True))
            if hard_example_weight(row, score) > 1.0
        ][:1]
        if not selected:
            raise ValueError("hard-negative smoke lacks a weighted occurrence")
    elif mode == "rank_candidate":
        assert teacher_scores is not None
        for left, row in enumerate(rows):
            if str(row["category"]) != FLAMMABLE:
                continue
            for right in range(left + 1, len(rows)):
                other = rows[right]
                if (
                    str(other["category"]) == FLAMMABLE
                    and int(other["label"]) == int(row["label"])
                    and teacher_scores[left] != teacher_scores[right]
                ):
                    selected = [left, right]
                    break
            if selected:
                break
        if not selected:
            raise ValueError("rank smoke lacks a same-label teacher-ordered pair")
    selected_set = set(selected)
    selected.extend(index for index in range(len(rows)) if index not in selected_set)
    return selected[:8]


def main() -> None:
    global base_prompt

    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from grid_contract import base_prompt as grid_base_prompt

    base_prompt = grid_base_prompt

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--teacher-target-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_UPDATES,
    )
    args = parser.parse_args()
    if args.checkpoint_every_updates < 0:
        raise ValueError("checkpoint interval must be non-negative")
    if args.mode != "gold_control" and args.teacher_target_dir is None:
        raise ValueError("teacher-guided modes require --teacher-target-dir")
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
    args.output_dir.mkdir(parents=True)
    checkpoint_every_updates = (
        0 if args.technical_smoke else args.checkpoint_every_updates
    )

    runtime_audit_path = args.runtime_dir / "runtime_audit.json"
    train_path = args.runtime_dir / "train.jsonl"
    validation_path = args.runtime_dir / "validation.jsonl"
    audit = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
    if audit.get("fold") != args.fold or audit.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("runtime/fold mismatch")
    if audit.get("train_sha256") != sha256(train_path):
        raise ValueError("training runtime checksum mismatch")
    if audit.get("validation_sha256") != sha256(validation_path):
        raise ValueError("validation runtime checksum mismatch")
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    if len(train_rows) != int(audit["train_occurrences"]):
        raise ValueError("training runtime row count mismatch")
    if len(validation_rows) != int(audit["validation_rows"]):
        raise ValueError("validation runtime row count mismatch")

    teacher_values: list[float] | None = None
    teacher_binding: dict | None = None
    if args.teacher_target_dir is not None:
        teacher_values, teacher_binding = load_teacher_targets(
            args.teacher_target_dir,
            train_rows,
            args.fold,
            runtime_audit_sha256=sha256(runtime_audit_path),
            train_runtime_sha256=sha256(train_path),
        )
    if len(train_rows) < MATCHED_TRAIN_OCCURRENCES:
        raise ValueError("runtime is too short for the matched-control prefix")
    train_rows = train_rows[:MATCHED_TRAIN_OCCURRENCES]
    if teacher_values is not None:
        teacher_values = teacher_values[:MATCHED_TRAIN_OCCURRENCES]
    full_train_count = len(train_rows)
    full_validation_count = len(validation_rows)
    if args.technical_smoke:
        selected = smoke_indices(args.mode, train_rows, teacher_values)
        train_rows = [train_rows[index] for index in selected]
        validation_rows = validation_rows[:2]
        if teacher_values is not None:
            teacher_values = [teacher_values[index] for index in selected]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1 or zero == one:
        raise ValueError("0/1 are not distinct atomic tokens")

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL,
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
            target_modules=LORA_TARGET_MODULES,
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
    indices = list(range(len(train_rows))) if args.technical_smoke else frozen_order(len(train_rows))
    occurrence_order_sha256 = hashlib.sha256(
        np.asarray(indices, dtype="<i8").tobytes()
    ).hexdigest()
    micro_batches = math.ceil(len(indices) / MICRO_BATCH)
    updates = math.ceil(micro_batches / GRAD_ACCUM)
    warmup = max(1, int(updates * 0.05))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    resume_contract = {
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "fold": args.fold,
        "mode": args.mode,
        "runtime_audit_sha256": sha256(runtime_audit_path),
        "train_runtime_sha256": sha256(train_path),
        "validation_runtime_sha256": sha256(validation_path),
        "common_training_contract": common_training_contract(
            occurrence_order_sha256
        ),
        "objective_contract": objective_contract(args.mode),
        "teacher_binding": teacher_binding,
        "technical_smoke": args.technical_smoke,
    }
    resume = load_resume_checkpoint(checkpoint_path, resume_contract, len(indices))
    start_offset = 0
    optimizer_steps = 0
    weighted_occurrences = 0
    rank_pairs = 0
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
        weighted_occurrences = int(resume["weighted_occurrences"])
        rank_pairs = int(resume["rank_pairs"])
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
    initial_micro_step = start_offset // MICRO_BATCH
    for step, offset in enumerate(
        range(start_offset, len(indices), MICRO_BATCH),
        start=initial_micro_step + 1,
    ):
        local_indices = indices[offset : offset + MICRO_BATCH]
        local = [train_rows[index] for index in local_indices]
        local_teacher = (
            [teacher_values[index] for index in local_indices]
            if teacher_values is not None
            else None
        )
        rows = [SimpleNamespace(**item) for item in local]
        images = [open_image(item) for item in local]
        try:
            student_scores = scores(model, batch_inputs(processor, rows, images), zero[0], one[0])
            labels = torch.tensor(
                [item["label"] for item in local],
                dtype=torch.float32,
                device=student_scores.device,
            )
            per_row = F.binary_cross_entropy_with_logits(
                student_scores.float(), labels, reduction="none"
            )
            if args.mode == "hardneg_candidate":
                assert local_teacher is not None
                weights = torch.tensor(
                    [
                        hard_example_weight(item, teacher_score)
                        for item, teacher_score in zip(local, local_teacher, strict=True)
                    ],
                    dtype=torch.float32,
                    device=student_scores.device,
                )
                weighted_occurrences += int((weights > 1.0).sum().item())
                hard_loss = (per_row * weights).mean()
            else:
                hard_loss = per_row.mean()
            if args.mode == "rank_candidate":
                assert local_teacher is not None
                rank_loss, pairs = within_stratum_rank_loss(
                    student_scores.float(), local, local_teacher
                )
                rank_pairs += pairs
                raw_loss = hard_loss + RANK_COEFFICIENT * rank_loss
            else:
                raw_loss = hard_loss
            loss_window.append(float(raw_loss.detach().cpu()))
            (raw_loss / GRAD_ACCUM).backward()
        finally:
            for image in images:
                image.close()
        if step % GRAD_ACCUM == 0 or offset + MICRO_BATCH >= len(indices):
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        processed = min(offset + MICRO_BATCH, len(indices))
        if (
            checkpoint_every_updates > 0
            and optimizer_steps > 0
            and optimizer_steps % checkpoint_every_updates == 0
            and (step % GRAD_ACCUM == 0 or processed == len(indices))
        ):
            atomic_torch_save(
                {
                    "schema_version": RESUME_SCHEMA,
                    "contract": resume_contract,
                    "next_offset": processed,
                    "optimizer_steps": optimizer_steps,
                    "adapter_state": {
                        key: value.detach().cpu()
                        for key, value in get_peft_model_state_dict(model).items()
                    },
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all(),
                    "weighted_occurrences": weighted_occurrences,
                    "rank_pairs": rank_pairs,
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
        if step % 50 == 0 or processed == len(indices):
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "processed": processed,
                        "rows": len(indices),
                        "optimizer_steps": optimizer_steps,
                        "mean_loss": round(float(np.mean(loss_window)), 6),
                        "weighted_occurrences": weighted_occurrences,
                        "rank_pairs": rank_pairs,
                        "elapsed_min": round((time.monotonic() - started) / 60, 2),
                    }
                ),
                flush=True,
            )
            loss_window.clear()

    if args.mode == "hardneg_candidate" and weighted_occurrences == 0:
        raise ValueError("hard-negative objective never exercised a weighted occurrence")
    if args.mode == "rank_candidate" and rank_pairs == 0:
        raise ValueError("rank objective never exercised a same-label teacher pair")

    model.eval()
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(validation_rows), 8):
            local = validation_rows[offset : offset + 8]
            rows = [SimpleNamespace(**item) for item in local]
            images = [open_image(item) for item in local]
            try:
                values = scores(model, batch_inputs(processor, rows, images), zero[0], one[0])
            finally:
                for image in images:
                    image.close()
            predictions.extend(
                {
                    "id": str(item["id"]),
                    "fold": args.fold,
                    "mode": args.mode,
                    "score": float(value),
                }
                for item, value in zip(local, values.float().cpu(), strict=True)
            )
            processed = offset + len(local)
            if processed % 200 == 0 or processed == len(validation_rows):
                print(f"predicted={processed}/{len(validation_rows)}", flush=True)

    prediction_path = args.output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    adapter = args.output_dir / "adapter"
    model.save_pretrained(adapter)
    report = {
        "schema_version": "exp698_fold_output_v2",
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "fold": args.fold,
        "mode": args.mode,
        "model": str(MODEL),
        "image_preprocessing": IMAGE_PREPROCESSING,
        "submission_eligible_base_model": True,
        "teacher_model_required_at_inference": False,
        "train_occurrences": len(train_rows),
        "full_train_occurrences": full_train_count,
        "validation_rows": len(validation_rows),
        "full_validation_rows": full_validation_count,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": GRAD_ACCUM,
        "effective_batch": MICRO_BATCH * GRAD_ACCUM,
        "checkpoint_every_updates": checkpoint_every_updates,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "optimizer_steps": optimizer_steps,
        "weighted_occurrences": weighted_occurrences,
        "rank_pairs": rank_pairs,
        "runtime_minutes": (time.monotonic() - started) / 60,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "technical_smoke": args.technical_smoke,
        "runtime_audit_sha256": sha256(runtime_audit_path),
        "predictions_sha256": sha256(prediction_path),
        "adapter_model_sha256": sha256(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
        "training_contract": common_training_contract(occurrence_order_sha256),
        "objective_contract": objective_contract(args.mode),
        "teacher_binding": teacher_binding,
    }
    report["contract_sha256"] = canonical_sha256(report)
    (args.output_dir / "output_contract.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        checkpoint_path.unlink(missing_ok=True)
    except OSError as error:
        print(f"checkpoint_cleanup_warning={error}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

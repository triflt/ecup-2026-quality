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
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "experiments/645_qwen_scale_2x3_gate"
STUDENT_DIR = ROOT / "experiments/698_qwen35_teacher_guided_controls"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(STUDENT_DIR))

import grid_contract
import run_fold as student


EXPERIMENT_ID = "706"
FLAMMABLE = "Легковоспламеняющиеся"
MODEL = Path("/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3.5-4B")
SEED = 20260829
LR = 1e-5
GRAD_ACCUM = 8
ANCHOR_WEIGHT = 1.0
HARD_WEIGHT = 0.15
RANK_WEIGHT = 0.10
CHECKPOINT_UPDATES = {10, 20, 40, 80}
RESUME_SCHEMA = "exp706_parent_anchored_resume_v1"


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
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def unique_flammable_rows(rows: list[dict], targets: list[dict]) -> list[dict]:
    if len(rows) != len(targets):
        raise ValueError("runtime and teacher targets have different lengths")
    by_id: dict[str, dict] = {}
    for occurrence, (row, target) in enumerate(zip(rows, targets, strict=True)):
        expected = {
            "occurrence_index": occurrence,
            "id": str(row["id"]),
            "source_fold": int(row["fold"]),
            "category": str(row["category"]),
            "label": int(row["label"]),
        }
        if any(target.get(key) != value for key, value in expected.items()):
            raise ValueError("teacher target binding mismatch")
        if str(row["category"]) != FLAMMABLE:
            continue
        item = dict(row)
        item["teacher_score"] = float(target["score"])
        item["teacher_rank"] = float(target["rank_score"])
        old = by_id.get(str(row["id"]))
        if old is not None:
            for key in ("label", "teacher_score", "teacher_rank"):
                if old[key] != item[key]:
                    raise ValueError(f"duplicate id has inconsistent {key}")
            continue
        by_id[str(row["id"])] = item
    result = list(by_id.values())
    if len(result) < 1000 or {int(row["label"]) for row in result} != {0, 1}:
        raise ValueError("flammable target coverage is insufficient")
    return result


def pair_manifest(rows: list[dict], max_updates: int) -> list[tuple[int, int]]:
    rng = random.Random(SEED)
    strata: dict[int, list[int]] = {0: [], 1: []}
    for index, row in enumerate(rows):
        strata[int(row["label"])].append(index)
    pools: dict[int, list[tuple[int, int]]] = {}
    for label, indices in strata.items():
        ordered = sorted(indices, key=lambda index: (rows[index]["teacher_rank"], rows[index]["id"]))
        half = len(ordered) // 2
        low = ordered[:half]
        high = ordered[-half:]
        pairs = [
            (right, left)
            for left, right in zip(low, reversed(high), strict=True)
            if rows[right]["teacher_rank"] > rows[left]["teacher_rank"]
        ]
        if not pairs:
            raise ValueError(f"teacher ranks produce no pairs for label={label}")
        rng.shuffle(pairs)
        pools[label] = pairs
    required_pairs = max_updates * GRAD_ACCUM
    result = []
    offsets = {0: 0, 1: 0}
    for position in range(required_pairs):
        label = position % 2
        pool = pools[label]
        if offsets[label] and offsets[label] % len(pool) == 0:
            rng.shuffle(pool)
        result.append(pool[offsets[label] % len(pool)])
        offsets[label] += 1
    return result


def adapter_snapshot(model, directory: Path, update: int) -> Path:
    target = directory / "checkpoints" / f"update_{update:04d}" / "adapter"
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    return target


def main() -> None:
    from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--teacher-target-dir", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=80)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--anchor-weight", type=float, default=ANCHOR_WEIGHT)
    parser.add_argument("--hard-weight", type=float, default=HARD_WEIGHT)
    parser.add_argument("--rank-weight", type=float, default=RANK_WEIGHT)
    args = parser.parse_args()
    if args.max_updates < 1 or args.max_updates > max(CHECKPOINT_UPDATES):
        raise ValueError("max updates must be between 1 and 80")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    if args.anchor_weight <= 0.0 or args.hard_weight < 0.0 or args.rank_weight < 0.0:
        raise ValueError("loss weights are invalid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = args.resume or args.output_dir / "resume.pt"

    runtime_path = args.runtime_dir / "train.jsonl"
    target_path = args.teacher_target_dir / "teacher_targets.jsonl"
    target_contract_path = args.teacher_target_dir / "teacher_target_contract.json"
    rows = read_jsonl(runtime_path)
    targets = read_jsonl(target_path)
    unique_rows = unique_flammable_rows(rows, targets)
    pairs = pair_manifest(unique_rows, args.max_updates)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "runtime_sha256": sha256(runtime_path),
        "teacher_targets_sha256": sha256(target_path),
        "teacher_target_contract_sha256": sha256(target_contract_path),
        "parent_adapter_sha256": sha256(args.parent_adapter / "adapter_model.safetensors"),
        "unique_flammable_rows": len(unique_rows),
        "positive_rows": sum(int(row["label"]) == 1 for row in unique_rows),
        "negative_rows": sum(int(row["label"]) == 0 for row in unique_rows),
        "pairs": len(pairs),
        "max_updates": args.max_updates,
        "seed": SEED,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": GRAD_ACCUM,
        "loss_weights": {
            "parent_anchor_smooth_l1": args.anchor_weight,
            "hard_gold_bce": args.hard_weight,
            "teacher_pair_rank": args.rank_weight,
        },
        "bad_route": "immutable_original_exp140_adapter",
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps({**manifest, "manifest_sha256": manifest_sha}, ensure_ascii=False, indent=2) + "\n"
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    student.base_prompt = grid_contract.base_prompt
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1:
        raise ValueError("binary verdict tokens are not atomic")
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model = PeftModel.from_pretrained(base, args.parent_adapter, is_trainable=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min(1.0, (update + 1) / 5)
    )

    # Cache exact production-parent logits before the first optimizer update.
    used = sorted({index for pair in pairs for index in pair})
    anchor_cache_path = args.output_dir / "parent_anchor_scores.json"
    if anchor_cache_path.is_file():
        anchor_scores = {
            int(key): float(value)
            for key, value in json.loads(anchor_cache_path.read_text()).items()
        }
        if set(anchor_scores) != set(used):
            raise ValueError("parent anchor cache coverage mismatch")
    else:
        model.eval()
        anchor_scores: dict[int, float] = {}
        started_anchor = time.monotonic()
        for start in range(0, len(used), 8):
            local_indices = used[start : start + 8]
            local = [unique_rows[index] for index in local_indices]
            images = [student.open_image(row) for row in local]
            try:
                with torch.inference_mode():
                    values = student.scores(
                        model,
                        student.batch_inputs(
                            processor,
                            [SimpleNamespace(**row) for row in local],
                            images,
                        ),
                        zero[0],
                        one[0],
                    )
            finally:
                for image in images:
                    image.close()
            anchor_scores.update(
                (index, float(value)) for index, value in zip(local_indices, values.cpu(), strict=True)
            )
            if start % 160 == 0 or start + len(local) == len(used):
                print(
                    json.dumps(
                        {
                            "anchor_cached": min(start + 8, len(used)),
                            "anchor_total": len(used),
                            "elapsed_min": round((time.monotonic() - started_anchor) / 60, 2),
                        }
                    ),
                    flush=True,
                )
        temporary = anchor_cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(anchor_scores, sort_keys=True) + "\n")
        os.replace(temporary, anchor_cache_path)

    start_pair = 0
    optimizer_updates = 0
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume.get("schema_version") != RESUME_SCHEMA or resume.get("manifest_sha256") != manifest_sha:
            raise ValueError("resume contract mismatch")
        set_peft_model_state_dict(model, resume["adapter_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        start_pair = int(resume["next_pair"])
        optimizer_updates = int(resume["optimizer_updates"])
        torch.set_rng_state(resume["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
        print(json.dumps({"resumed_pair": start_pair, "optimizer_updates": optimizer_updates}), flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    window: list[dict[str, float]] = []
    for pair_index in range(start_pair, len(pairs)):
        high_index, low_index = pairs[pair_index]
        local_indices = [high_index, low_index]
        local = [unique_rows[index] for index in local_indices]
        images = [student.open_image(row) for row in local]
        try:
            values = student.scores(
                model,
                student.batch_inputs(
                    processor, [SimpleNamespace(**row) for row in local], images
                ),
                zero[0],
                one[0],
            ).float()
        finally:
            for image in images:
                image.close()
        anchors = torch.tensor(
            [anchor_scores[index] for index in local_indices],
            dtype=torch.float32,
            device=values.device,
        )
        labels = torch.tensor(
            [int(row["label"]) for row in local],
            dtype=torch.float32,
            device=values.device,
        )
        anchor_loss = F.smooth_l1_loss(values, anchors)
        hard_loss = F.binary_cross_entropy_with_logits(values, labels)
        rank_loss = F.softplus(-(values[0] - values[1]))
        loss = (
            args.anchor_weight * anchor_loss
            + args.hard_weight * hard_loss
            + args.rank_weight * rank_loss
        )
        (loss / GRAD_ACCUM).backward()
        window.append(
            {
                "loss": float(loss.detach()),
                "anchor": float(anchor_loss.detach()),
                "hard": float(hard_loss.detach()),
                "rank": float(rank_loss.detach()),
            }
        )
        completed_pair = pair_index + 1
        if completed_pair % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1
            if optimizer_updates in CHECKPOINT_UPDATES or optimizer_updates == args.max_updates:
                adapter = adapter_snapshot(model, args.output_dir, optimizer_updates)
                state = {
                    "schema_version": RESUME_SCHEMA,
                    "manifest_sha256": manifest_sha,
                    "next_pair": completed_pair,
                    "optimizer_updates": optimizer_updates,
                    "adapter_state": get_peft_model_state_dict(model),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all(),
                }
                atomic_torch_save(state, resume_path)
                report = {
                    "schema_version": "exp706_parent_anchored_checkpoint_v1",
                    "experiment_id": EXPERIMENT_ID,
                    "optimizer_updates": optimizer_updates,
                    "pairs_consumed": completed_pair,
                    "adapter_model_sha256": sha256(adapter / "adapter_model.safetensors"),
                    "manifest_sha256": manifest_sha,
                    "parent_anchor": True,
                    "bad_route_immutable": True,
                    "teacher_at_inference": False,
                }
                (adapter.parent / "output_contract.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
                )
            if optimizer_updates % 5 == 0:
                print(
                    json.dumps(
                        {
                            "optimizer_updates": optimizer_updates,
                            "pairs": completed_pair,
                            "mean_loss": round(float(np.mean([row["loss"] for row in window])), 6),
                            "mean_anchor": round(float(np.mean([row["anchor"] for row in window])), 6),
                            "mean_hard": round(float(np.mean([row["hard"] for row in window])), 6),
                            "mean_rank": round(float(np.mean([row["rank"] for row in window])), 6),
                            "elapsed_min": round((time.monotonic() - started) / 60, 2),
                        }
                    ),
                    flush=True,
                )
                window.clear()
            if optimizer_updates >= args.max_updates:
                break


if __name__ == "__main__":
    main()

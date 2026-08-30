from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import train_parent_anchored as shared


SCHEMA = "exp706_parent_anchored_full_resume_v1"
CONTRACT_SCHEMA = "exp706_parent_anchored_full_v1"


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_parent_scores(paths: list[Path], adapter_sha256: str) -> tuple[dict[str, float], list[dict]]:
    scores: dict[str, float] = {}
    sources: list[dict] = []
    for path in paths:
        metrics_path = path.with_name("metrics.json")
        if not path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"parent prediction contract missing for {path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("adapter_sha256") != adapter_sha256:
            raise ValueError(f"parent adapter mismatch for {path}")
        if metrics.get("predictions_sha256") != shared.sha256(path):
            raise ValueError(f"parent prediction checksum mismatch for {path}")
        count = 0
        for row in shared.read_jsonl(path):
            row_id = str(row["id"])
            value = float(row["score"])
            if not math.isfinite(value) or row_id in scores:
                raise ValueError(f"invalid or duplicate parent score for id={row_id}")
            scores[row_id] = value
            count += 1
        sources.append(
            {
                "path": str(path),
                "predictions_sha256": shared.sha256(path),
                "metrics_sha256": shared.sha256(metrics_path),
                "rows": count,
            }
        )
    return scores, sources


def coverage_pools(rows: list[dict]) -> dict[int, list[tuple[int, int, bool]]]:
    strata: dict[int, list[int]] = {0: [], 1: []}
    for index, row in enumerate(rows):
        strata[int(row["label"])].append(index)
    pools: dict[int, list[tuple[int, int, bool]]] = {}
    for label, indices in strata.items():
        ordered = sorted(
            indices, key=lambda index: (rows[index]["teacher_rank"], rows[index]["id"])
        )
        if len(ordered) < 2:
            raise ValueError(f"insufficient rows for label={label}")
        pairs: list[tuple[int, int, bool]] = []
        for left_position in range((len(ordered) + 1) // 2):
            low = ordered[left_position]
            high = ordered[-1 - left_position]
            if low == high:
                low = ordered[0] if high != ordered[0] else ordered[-1]
            active = rows[high]["teacher_rank"] > rows[low]["teacher_rank"]
            pairs.append((high, low, active))
        pools[label] = pairs
    return pools


def pair_schedule(
    rows: list[dict], optimizer_updates: int, seed: int
) -> tuple[list[tuple[int, int, bool]], int, dict[int, int]]:
    pools = coverage_pools(rows)
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    minimum_pairs = 2 * max(len(pool) for pool in pools.values())
    pairs_per_update = math.ceil(minimum_pairs / optimizer_updates)
    total_pairs = pairs_per_update * optimizer_updates
    schedule: list[tuple[int, int, bool]] = []
    offsets = {0: 0, 1: 0}
    for position in range(total_pairs):
        label = position % 2
        pool = pools[label]
        if offsets[label] and offsets[label] % len(pool) == 0:
            rng.shuffle(pool)
        schedule.append(pool[offsets[label] % len(pool)])
        offsets[label] += 1
    covered = {index for high, low, _ in schedule for index in (high, low)}
    if covered != set(range(len(rows))):
        missing = len(rows) - len(covered)
        raise ValueError(f"coverage schedule omitted {missing} rows")
    return schedule, pairs_per_update, {label: len(pool) for label, pool in pools.items()}


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--teacher-target-dir", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--parent-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimizer-updates", type=int, default=40)
    parser.add_argument("--pairs-per-microbatch", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=shared.LR)
    parser.add_argument("--anchor-weight", type=float, default=shared.ANCHOR_WEIGHT)
    parser.add_argument("--hard-weight", type=float, default=shared.HARD_WEIGHT)
    parser.add_argument("--rank-weight", type=float, default=shared.RANK_WEIGHT)
    parser.add_argument("--seed", type=int, default=shared.SEED)
    parser.add_argument("--row-limit-per-label", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.optimizer_updates < 1 or args.pairs_per_microbatch < 1:
        raise ValueError("update and microbatch sizes must be positive")
    if args.learning_rate <= 0 or args.anchor_weight <= 0:
        raise ValueError("learning rate and anchor weight must be positive")
    if args.hard_weight < 0 or args.rank_weight < 0:
        raise ValueError("loss weights must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = args.resume or args.output_dir / "resume.pt"

    runtime_path = args.runtime_dir / "train.jsonl"
    target_path = args.teacher_target_dir / "teacher_targets.jsonl"
    target_contract_path = args.teacher_target_dir / "teacher_target_contract.json"
    target_contract = json.loads(target_contract_path.read_text(encoding="utf-8"))
    if target_contract.get("strict_oof") is not True:
        raise ValueError("full refit requires strict OOF teacher targets")
    rows = shared.unique_flammable_rows(
        shared.read_jsonl(runtime_path), shared.read_jsonl(target_path)
    )
    if args.row_limit_per_label is not None:
        if args.row_limit_per_label < 2:
            raise ValueError("row limit must be at least two per label")
        rows = [
            row
            for label in (0, 1)
            for row in [item for item in rows if int(item["label"]) == label][
                : args.row_limit_per_label
            ]
        ]
    adapter_path = args.parent_adapter / "adapter_model.safetensors"
    adapter_sha = shared.sha256(adapter_path)
    parent_by_id, parent_sources = load_parent_scores(args.parent_predictions, adapter_sha)
    missing_parent = sorted({str(row["id"]) for row in rows} - set(parent_by_id))
    if missing_parent:
        raise ValueError(f"missing {len(missing_parent)} parent scores")

    schedule, pairs_per_update, pool_sizes = pair_schedule(
        rows, args.optimizer_updates, args.seed
    )
    rank_active = sum(active for _, _, active in schedule)
    manifest = {
        "schema_version": "exp706_parent_anchored_full_manifest_v1",
        "experiment_id": "706",
        "runtime_sha256": shared.sha256(runtime_path),
        "teacher_targets_sha256": shared.sha256(target_path),
        "teacher_target_contract_sha256": shared.sha256(target_contract_path),
        "teacher_folds": target_contract.get("folds"),
        "strict_oof": True,
        "parent_adapter_sha256": adapter_sha,
        "parent_prediction_sources": parent_sources,
        "unique_flammable_rows": len(rows),
        "unique_flammable_rows_covered": len(rows),
        "positive_rows": sum(int(row["label"]) == 1 for row in rows),
        "negative_rows": sum(int(row["label"]) == 0 for row in rows),
        "pair_pool_sizes": pool_sizes,
        "pairs": len(schedule),
        "rank_active_pairs": rank_active,
        "optimizer_updates": args.optimizer_updates,
        "pairs_per_update": pairs_per_update,
        "pairs_per_microbatch": args.pairs_per_microbatch,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "loss_weights": {
            "parent_anchor_smooth_l1": args.anchor_weight,
            "hard_gold_bce": args.hard_weight,
            "teacher_pair_rank": args.rank_weight,
        },
        "bad_route": "immutable_original_exp140_adapter",
        "technical_row_limit_per_label": args.row_limit_per_label,
    }
    manifest_sha = canonical_sha256(manifest)
    atomic_json({**manifest, "manifest_sha256": manifest_sha}, args.output_dir / "training_manifest.json")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    shared.student.base_prompt = shared.grid_contract.base_prompt
    processor = AutoProcessor.from_pretrained(
        shared.MODEL, local_files_only=True, trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1:
        raise ValueError("binary verdict tokens are not atomic")
    base_model = AutoModelForMultimodalLM.from_pretrained(
        shared.MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model = PeftModel.from_pretrained(base_model, args.parent_adapter, is_trainable=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min(1.0, (update + 1) / 5)
    )

    start_update = 0
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume.get("schema_version") != SCHEMA or resume.get("manifest_sha256") != manifest_sha:
            raise ValueError("resume contract mismatch")
        set_peft_model_state_dict(model, resume["adapter_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        start_update = int(resume["optimizer_updates"])
        torch.set_rng_state(resume["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
        print(json.dumps({"resumed_update": start_update}), flush=True)

    model.train()
    started = time.monotonic()
    for update in range(start_update, args.optimizer_updates):
        optimizer.zero_grad(set_to_none=True)
        window = {"anchor": 0.0, "hard": 0.0, "rank": 0.0, "pairs": 0}
        update_pairs = schedule[update * pairs_per_update : (update + 1) * pairs_per_update]
        for offset in range(0, len(update_pairs), args.pairs_per_microbatch):
            local_pairs = update_pairs[offset : offset + args.pairs_per_microbatch]
            indices = [index for high, low, _ in local_pairs for index in (high, low)]
            local = [rows[index] for index in indices]
            images = [shared.student.open_image(row) for row in local]
            try:
                values = shared.student.scores(
                    model,
                    shared.student.batch_inputs(
                        processor, [SimpleNamespace(**row) for row in local], images
                    ),
                    zero[0],
                    one[0],
                ).float()
            finally:
                for image in images:
                    image.close()
            anchors = torch.tensor(
                [parent_by_id[str(row["id"])] for row in local],
                dtype=torch.float32,
                device=values.device,
            )
            labels = torch.tensor(
                [int(row["label"]) for row in local],
                dtype=torch.float32,
                device=values.device,
            )
            anchor_sum = F.smooth_l1_loss(values, anchors, reduction="sum")
            hard_sum = F.binary_cross_entropy_with_logits(values, labels, reduction="sum")
            rank_values = F.softplus(-(values[0::2] - values[1::2]))
            active = torch.tensor(
                [float(item[2]) for item in local_pairs],
                dtype=torch.float32,
                device=values.device,
            )
            rank_sum = (rank_values * active).sum()
            loss = (
                args.anchor_weight * anchor_sum / (2 * pairs_per_update)
                + args.hard_weight * hard_sum / (2 * pairs_per_update)
                + args.rank_weight * rank_sum / pairs_per_update
            )
            loss.backward()
            window["anchor"] += float(anchor_sum.detach())
            window["hard"] += float(hard_sum.detach())
            window["rank"] += float(rank_sum.detach())
            window["pairs"] += len(local_pairs)
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        optimizer.step()
        scheduler.step()
        completed = update + 1
        if completed % args.checkpoint_every == 0 or completed == args.optimizer_updates:
            adapter = shared.adapter_snapshot(model, args.output_dir, completed)
            state = {
                "schema_version": SCHEMA,
                "manifest_sha256": manifest_sha,
                "optimizer_updates": completed,
                "adapter_state": get_peft_model_state_dict(model),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all(),
            }
            shared.atomic_torch_save(state, resume_path)
            checkpoint_contract = {
                "schema_version": "exp706_parent_anchored_full_checkpoint_v1",
                "experiment_id": "706",
                "optimizer_updates": completed,
                "pairs_consumed": completed * pairs_per_update,
                "unique_flammable_rows_covered": len(rows),
                "adapter_model_sha256": shared.sha256(adapter / "adapter_model.safetensors"),
                "manifest_sha256": manifest_sha,
                "strict_oof": True,
                "parent_anchor": True,
                "bad_route_immutable": True,
                "teacher_at_inference": False,
            }
            atomic_json(checkpoint_contract, adapter.parent / "output_contract.json")
        print(
            json.dumps(
                {
                    "optimizer_updates": completed,
                    "optimizer_updates_total": args.optimizer_updates,
                    "pairs": completed * pairs_per_update,
                    "pairs_total": len(schedule),
                    "mean_anchor": window["anchor"] / (2 * window["pairs"]),
                    "mean_hard": window["hard"] / (2 * window["pairs"]),
                    "mean_rank": window["rank"] / window["pairs"],
                    "elapsed_min": round((time.monotonic() - started) / 60, 2),
                }
            ),
            flush=True,
        )

    adapter = args.output_dir / "checkpoints" / f"update_{args.optimizer_updates:04d}" / "adapter"
    final = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment_id": "706",
        "selected_from_strict_holdout_update": args.optimizer_updates,
        "full_oof": True,
        "strict_oof": True,
        "teacher_folds": target_contract.get("folds"),
        "unique_flammable_rows": len(rows),
        "unique_flammable_rows_covered": len(rows),
        "adapter_model_sha256": shared.sha256(adapter / "adapter_model.safetensors"),
        "adapter_bytes": (adapter / "adapter_model.safetensors").stat().st_size,
        "manifest_sha256": manifest_sha,
        "parent_adapter_sha256": adapter_sha,
        "bad_route": "immutable_original_exp140_adapter",
        "teacher_at_inference": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
        "technical_smoke": args.row_limit_per_label is not None,
    }
    final["contract_sha256"] = canonical_sha256(final)
    atomic_json(final, args.output_dir / "output_contract.json")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

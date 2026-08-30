from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import train_parent_anchored as shared
import train_parent_anchored_full as full


SCHEMA = "exp706_parent_anchored_continuation_resume_v1"


def continuation_schedule(
    rows: list[dict], additional_updates: int, pairs_per_update: int, seed: int
) -> list[tuple[int, int, bool]]:
    pools = full.coverage_pools(rows)
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    offsets = {0: 0, 1: 0}
    result: list[tuple[int, int, bool]] = []
    for position in range(additional_updates * pairs_per_update):
        label = position % 2
        pool = pools[label]
        if offsets[label] and offsets[label] % len(pool) == 0:
            rng.shuffle(pool)
        result.append(pool[offsets[label] % len(pool)])
        offsets[label] += 1
    return result


def main() -> None:
    from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--teacher-target-dir", type=Path, required=True)
    parser.add_argument("--original-parent-adapter", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--prior-contract", type=Path, required=True)
    parser.add_argument("--parent-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--additional-updates", type=int, default=20)
    parser.add_argument("--pairs-per-update", type=int, default=133)
    parser.add_argument("--pairs-per-microbatch", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--anchor-weight", type=float, default=4.0)
    parser.add_argument("--hard-weight", type=float, default=0.10)
    parser.add_argument("--rank-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=shared.SEED + 17)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(
        args.additional_updates,
        args.pairs_per_update,
        args.pairs_per_microbatch,
        args.checkpoint_every,
    ) < 1:
        raise ValueError("continuation sizes must be positive")
    if args.learning_rate <= 0 or args.anchor_weight <= 0:
        raise ValueError("learning rate and anchor weight must be positive")
    if args.hard_weight < 0 or args.rank_weight < 0:
        raise ValueError("loss weights must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = args.resume or args.output_dir / "resume.pt"

    prior = json.loads(args.prior_contract.read_text(encoding="utf-8"))
    prior_payload = dict(prior)
    prior_digest = prior_payload.pop("contract_sha256", None)
    if prior_digest != full.canonical_sha256(prior_payload):
        raise ValueError("prior full-refit contract self-hash mismatch")
    expected_prior = {
        "schema_version": "exp706_parent_anchored_full_v1",
        "selected_from_strict_holdout_update": 40,
        "full_oof": True,
        "strict_oof": True,
        "technical_smoke": False,
    }
    mismatch = {
        key: {"expected": value, "actual": prior.get(key)}
        for key, value in expected_prior.items()
        if prior.get(key) != value
    }
    if mismatch:
        raise ValueError(f"prior full-refit contract mismatch: {mismatch}")
    initial_path = args.initial_adapter / "adapter_model.safetensors"
    if shared.sha256(initial_path) != prior.get("adapter_model_sha256"):
        raise ValueError("initial adapter does not match selected update40")

    runtime_path = args.runtime_dir / "train.jsonl"
    target_path = args.teacher_target_dir / "teacher_targets.jsonl"
    target_contract_path = args.teacher_target_dir / "teacher_target_contract.json"
    target_contract = json.loads(target_contract_path.read_text(encoding="utf-8"))
    if target_contract.get("folds") != [0, 1, 2, 3, 4] or target_contract.get("strict_oof") is not True:
        raise ValueError("continuation requires five-fold strict OOF targets")
    rows = shared.unique_flammable_rows(
        shared.read_jsonl(runtime_path), shared.read_jsonl(target_path)
    )
    original_parent_sha = shared.sha256(
        args.original_parent_adapter / "adapter_model.safetensors"
    )
    if original_parent_sha != prior.get("parent_adapter_sha256"):
        raise ValueError("original parent binding changed")
    parent_by_id, parent_sources = full.load_parent_scores(
        args.parent_predictions, original_parent_sha
    )
    missing_parent = sorted({str(row["id"]) for row in rows} - set(parent_by_id))
    if missing_parent:
        raise ValueError(f"missing {len(missing_parent)} original-parent scores")
    schedule = continuation_schedule(
        rows, args.additional_updates, args.pairs_per_update, args.seed
    )
    additional_covered = {
        index for high, low, _ in schedule for index in (high, low)
    }
    manifest = {
        "schema_version": "exp706_parent_anchored_continuation_manifest_v1",
        "experiment_id": "706",
        "prior_contract_sha256": shared.sha256(args.prior_contract),
        "initial_adapter_sha256": shared.sha256(initial_path),
        "original_parent_adapter_sha256": original_parent_sha,
        "runtime_sha256": shared.sha256(runtime_path),
        "teacher_targets_sha256": shared.sha256(target_path),
        "teacher_target_contract_sha256": shared.sha256(target_contract_path),
        "parent_prediction_sources": parent_sources,
        "strict_oof": True,
        "teacher_folds": [0, 1, 2, 3, 4],
        "unique_flammable_rows": len(rows),
        "cumulative_unique_flammable_rows_covered": int(prior["unique_flammable_rows_covered"]),
        "additional_unique_flammable_rows_covered": len(additional_covered),
        "additional_updates": args.additional_updates,
        "pairs": len(schedule),
        "pairs_per_update": args.pairs_per_update,
        "pairs_per_microbatch": args.pairs_per_microbatch,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "loss_weights": {
            "parent_anchor_smooth_l1": args.anchor_weight,
            "hard_gold_bce": args.hard_weight,
            "teacher_pair_rank": args.rank_weight,
        },
        "bad_route": "immutable_original_exp140_adapter",
    }
    manifest_sha = full.canonical_sha256(manifest)
    full.atomic_json(
        {**manifest, "manifest_sha256": manifest_sha},
        args.output_dir / "training_manifest.json",
    )

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
    model = PeftModel.from_pretrained(base_model, args.initial_adapter, is_trainable=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min(1.0, (update + 1) / 3)
    )

    start_update = 0
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume.get("schema_version") != SCHEMA or resume.get("manifest_sha256") != manifest_sha:
            raise ValueError("continuation resume contract mismatch")
        set_peft_model_state_dict(model, resume["adapter_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        start_update = int(resume["additional_updates"])
        torch.set_rng_state(resume["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
        print(json.dumps({"resumed_additional_update": start_update}), flush=True)

    model.train()
    started = time.monotonic()
    for update in range(start_update, args.additional_updates):
        optimizer.zero_grad(set_to_none=True)
        window = {"anchor": 0.0, "hard": 0.0, "rank": 0.0, "pairs": 0}
        update_pairs = schedule[
            update * args.pairs_per_update : (update + 1) * args.pairs_per_update
        ]
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
                args.anchor_weight * anchor_sum / (2 * args.pairs_per_update)
                + args.hard_weight * hard_sum / (2 * args.pairs_per_update)
                + args.rank_weight * rank_sum / args.pairs_per_update
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
        if completed % args.checkpoint_every == 0 or completed == args.additional_updates:
            adapter = shared.adapter_snapshot(model, args.output_dir, completed)
            shared.atomic_torch_save(
                {
                    "schema_version": SCHEMA,
                    "manifest_sha256": manifest_sha,
                    "additional_updates": completed,
                    "adapter_state": get_peft_model_state_dict(model),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all(),
                },
                resume_path,
            )
            full.atomic_json(
                {
                    "schema_version": "exp706_parent_anchored_continuation_checkpoint_v1",
                    "experiment_id": "706",
                    "additional_updates": completed,
                    "pairs_consumed": completed * args.pairs_per_update,
                    "adapter_model_sha256": shared.sha256(adapter / "adapter_model.safetensors"),
                    "manifest_sha256": manifest_sha,
                    "prior_contract_sha256": shared.sha256(args.prior_contract),
                    "strict_oof": True,
                    "teacher_at_inference": False,
                },
                adapter.parent / "output_contract.json",
            )
        print(
            json.dumps(
                {
                    "additional_updates": completed,
                    "additional_updates_total": args.additional_updates,
                    "pairs": completed * args.pairs_per_update,
                    "pairs_total": len(schedule),
                    "mean_anchor": window["anchor"] / (2 * window["pairs"]),
                    "mean_hard": window["hard"] / (2 * window["pairs"]),
                    "mean_rank": window["rank"] / window["pairs"],
                    "elapsed_min": round((time.monotonic() - started) / 60, 2),
                }
            ),
            flush=True,
        )

    adapter = (
        args.output_dir
        / "checkpoints"
        / f"update_{args.additional_updates:04d}"
        / "adapter"
    )
    contract = {
        "schema_version": "exp706_parent_anchored_continuation_v1",
        "experiment_id": "706",
        "selected_from_strict_holdout_update": 40,
        "additional_updates": args.additional_updates,
        "additional_pairs": len(schedule),
        "full_oof": True,
        "strict_oof": True,
        "teacher_folds": [0, 1, 2, 3, 4],
        "unique_flammable_rows": len(rows),
        "unique_flammable_rows_covered": int(prior["unique_flammable_rows_covered"]),
        "additional_unique_flammable_rows_covered": len(additional_covered),
        "adapter_model_sha256": shared.sha256(adapter / "adapter_model.safetensors"),
        "adapter_bytes": (adapter / "adapter_model.safetensors").stat().st_size,
        "manifest_sha256": manifest_sha,
        "prior_contract_sha256": shared.sha256(args.prior_contract),
        "parent_adapter_sha256": original_parent_sha,
        "bad_route": "immutable_original_exp140_adapter",
        "teacher_at_inference": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
        "technical_smoke": False,
        "selection_status": "SPECULATIVE_NO_NEW_HOLDOUT",
    }
    contract["contract_sha256"] = full.canonical_sha256(contract)
    full.atomic_json(contract, args.output_dir / "output_contract.json")
    print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

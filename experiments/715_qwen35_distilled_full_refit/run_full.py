from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

EXP698 = Path(__file__).resolve().parents[1] / "698_qwen35_teacher_guided_controls"
SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
sys.path.append(str(EXP698))
sys.path.append(str(SHARED))
import run_fold as student  # noqa: E402

MODEL = student.MODEL
MODES = ("hardneg_candidate", "rank_candidate")
ALL_MODES = ("gold_control", *MODES)
EXPECTED_SCORE_CALIBRATION = "category_and_outer_fold_percentile_average_ties"
MATCHED_CONTROL_VERIFICATION = {
    "schema_version": "exp698_matched_control_verification_v1",
    "fold_output_schema": "exp698_fold_output_v2",
    "common_training_factors_equal_per_fold": True,
    "candidate_teacher_bindings_equal_per_fold": True,
    "gold_teacher_binding_absent": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def resolve_winner(evaluation_path: Path, guard_path: Path) -> tuple[str, dict]:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    if evaluation.get("schema_version") != "exp698_evaluation_v2":
        raise ValueError("experiment-698 evaluation schema mismatch")
    if evaluation.get("matched_control_verification") != MATCHED_CONTROL_VERIFICATION:
        raise ValueError("experiment-698 matched-control verification mismatch")
    if guard.get("schema_version") != "exp698_connected_family_guard_v2":
        raise ValueError("experiment-698 guard schema mismatch")
    if evaluation.get("image_preprocessing") != student.IMAGE_PREPROCESSING:
        raise ValueError("experiment-698 image preprocessing mismatch")
    for mode in ("gold_control", *MODES):
        if (
            evaluation.get("modes", {}).get(mode, {}).get("score_calibration")
            != EXPECTED_SCORE_CALIBRATION
        ):
            raise ValueError(f"experiment-698 score calibration mismatch for {mode}")
    if guard.get("input_sha256", {}).get("evaluation") != sha256(evaluation_path):
        raise ValueError("guard/evaluation checksum mismatch")
    promoted = guard.get("promoted_candidates")
    if not isinstance(promoted, list) or not promoted:
        raise ValueError("no distillation candidate passed both promotion gates")
    if any(mode not in MODES for mode in promoted):
        raise ValueError("guard promoted an unknown mode")
    winner = max(
        sorted(promoted),
        key=lambda mode: float(
            evaluation["comparisons"][mode]["nested_macro_delta_vs_gold_control"]
        ),
    )
    return winner, {
        "evaluation_sha256": sha256(evaluation_path),
        "connected_guard_sha256": sha256(guard_path),
        "promoted_candidates": promoted,
        "selection_rule": "highest_primary_nested_macro_delta_among_dual_gate_promoted",
        "score_calibration": EXPECTED_SCORE_CALIBRATION,
        "image_preprocessing": student.IMAGE_PREPROCESSING,
        "selected_primary_delta": evaluation["comparisons"][winner][
            "nested_macro_delta_vs_gold_control"
        ],
    }


def resolve_explicit_mode(
    evaluation_path: Path, guard_path: Path, mode: str
) -> tuple[str, dict]:
    if mode not in ALL_MODES:
        raise ValueError(f"unknown explicit full-refit mode: {mode}")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    if evaluation.get("schema_version") != "exp698_evaluation_v2":
        raise ValueError("experiment-698 evaluation schema mismatch")
    if evaluation.get("matched_control_verification") != MATCHED_CONTROL_VERIFICATION:
        raise ValueError("experiment-698 matched-control verification mismatch")
    if guard.get("schema_version") != "exp698_connected_family_guard_v2":
        raise ValueError("experiment-698 guard schema mismatch")
    if guard.get("input_sha256", {}).get("evaluation") != sha256(evaluation_path):
        raise ValueError("guard/evaluation checksum mismatch")
    if evaluation.get("image_preprocessing") != student.IMAGE_PREPROCESSING:
        raise ValueError("experiment-698 image preprocessing mismatch")
    for candidate in ALL_MODES:
        if (
            evaluation.get("modes", {}).get(candidate, {}).get("score_calibration")
            != EXPECTED_SCORE_CALIBRATION
        ):
            raise ValueError(
                f"experiment-698 score calibration mismatch for {candidate}"
            )
    return mode, {
        "evaluation_sha256": sha256(evaluation_path),
        "connected_guard_sha256": sha256(guard_path),
        "promoted_candidates": guard.get("promoted_candidates", []),
        "selection_rule": "explicit_three_arm_auxiliary_refit",
        "score_calibration": EXPECTED_SCORE_CALIBRATION,
        "image_preprocessing": student.IMAGE_PREPROCESSING,
        "selected_primary_delta": (
            None
            if mode == "gold_control"
            else evaluation["comparisons"][mode][
                "nested_macro_delta_vs_gold_control"
            ]
        ),
    }


def load_targets(
    target_dir: Path,
    rows: list[dict],
    audit_path: Path,
    train_path: Path,
    *,
    allow_deadline_fast_targets: bool = False,
) -> tuple[list[float], list[float], dict]:
    target_path = target_dir / "teacher_targets.jsonl"
    contract_path = target_dir / "teacher_target_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    deadline_fast = contract.get("schema_version") == "exp705_f3_full_targets_v1"
    if deadline_fast and not allow_deadline_fast_targets:
        raise ValueError("deadline fast targets require explicit authorization")
    expected = (
        {
            "schema_version": "exp705_f3_full_targets_v1",
            "experiment_id": "705",
            "teacher_experiment_id": "697",
            "teacher_source_fold": 3,
            "target_scope": "full_train_occurrences",
            "teacher_scores_are_strict_oof_by_id": False,
            "teacher_never_trained_on_target_row": False,
            "oof_source_folds": [3],
            "in_sample_source_folds": [0, 1, 2, 4],
            "rank_score_calibration": "category_and_source_fold_percentile_average_ties",
            "rows": len(rows),
            "unique_ids": len({row["id"] for row in rows}),
            "runtime_audit_sha256": sha256(audit_path),
            "train_runtime_sha256": sha256(train_path),
            "teacher_targets_sha256": sha256(target_path),
            "decision": "DEADLINE_FAST_F3_TARGETS_FROZEN",
        }
        if deadline_fast
        else {
            "schema_version": "exp715_full_oof_targets_v1",
            "experiment_id": "715",
            "teacher_experiment_id": "697",
            "target_scope": "full_train_occurrences",
            "teacher_scores_are_strict_oof_by_id": True,
            "teacher_never_trained_on_target_row": True,
            "rank_score_calibration": "category_and_source_fold_percentile_average_ties",
            "rows": len(rows),
            "unique_ids": len({row["id"] for row in rows}),
            "runtime_audit_sha256": sha256(audit_path),
            "train_runtime_sha256": sha256(train_path),
            "teacher_targets_sha256": sha256(target_path),
            "decision": "FULL_OOF_TARGETS_FROZEN",
        }
    )
    mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"full target contract mismatch: {mismatch}")
    targets = read_jsonl(target_path)
    if len(targets) != len(rows):
        raise ValueError("full target row count mismatch")
    raw_scores = []
    rank_scores = []
    for occurrence_index, (row, target) in enumerate(zip(rows, targets, strict=True)):
        expected_target = {
            "occurrence_index": occurrence_index,
            "id": str(row["id"]),
            "source_fold": int(row["fold"]),
            "category": str(row["category"]),
            "label": int(row["label"]),
        }
        if set(target) != {*expected_target, "score", "rank_score"} or any(
            target.get(key) != value for key, value in expected_target.items()
        ):
            raise ValueError("full target occurrence binding mismatch")
        raw = float(target["score"])
        rank = float(target["rank_score"])
        if not math.isfinite(raw) or not math.isfinite(rank) or not 0.0 < rank <= 1.0:
            raise ValueError("full target score is invalid")
        raw_scores.append(raw)
        rank_scores.append(rank)
    return raw_scores, rank_scores, {
        "target_schema_version": contract["schema_version"],
        "deadline_fast_non_oof": deadline_fast,
        "teacher_target_contract_sha256": sha256(contract_path),
        "teacher_targets_sha256": sha256(target_path),
        "teacher_oof_sha256": contract.get("teacher_oof_sha256"),
        "teacher_oof_report_sha256": contract.get("teacher_oof_report_sha256"),
        "teacher_adapter_model_sha256": contract.get("teacher_adapter_model_sha256"),
    }


def main() -> None:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from grid_contract import base_prompt

    student.base_prompt = base_prompt

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--teacher-target-dir", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--connected-guard", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--mode", choices=ALL_MODES)
    parser.add_argument("--deadline-fast-track", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty full-refit output")
    args.output_dir.mkdir(parents=True)
    if args.deadline_fast_track:
        if args.mode is None:
            raise ValueError("deadline fast track requires an explicit --mode")
        mode = args.mode
        winner_binding = {
            "evaluation_sha256": None,
            "connected_guard_sha256": None,
            "promoted_candidates": [],
            "selection_rule": "deadline_explicit_full_data_no_oof_selection",
            "score_calibration": None,
            "image_preprocessing": student.IMAGE_PREPROCESSING,
            "selected_primary_delta": None,
        }
    elif args.mode is None:
        if args.evaluation is None or args.connected_guard is None:
            raise ValueError("guarded winner refit requires evaluation and connected guard")
        mode, winner_binding = resolve_winner(args.evaluation, args.connected_guard)
    else:
        if args.evaluation is None or args.connected_guard is None:
            raise ValueError("explicit guarded refit requires evaluation and connected guard")
        mode, winner_binding = resolve_explicit_mode(
            args.evaluation, args.connected_guard, args.mode
        )
    if mode != "gold_control" and args.teacher_target_dir is None:
        raise ValueError("teacher-guided full refit requires --teacher-target-dir")

    audit_path = args.runtime_dir / "runtime_audit.json"
    train_path = args.runtime_dir / "train.jsonl"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_audit = {
        "schema_version": "exp715_full_runtime_v1",
        "experiment_id": "715",
        "full_data": True,
        "decision": "FULL_RUNTIME_FROZEN",
        "train_sha256": sha256(train_path),
    }
    mismatch = {
        key: {"expected": value, "actual": audit.get(key)}
        for key, value in expected_audit.items()
        if audit.get(key) != value
    }
    if mismatch:
        raise ValueError(f"full runtime mismatch: {mismatch}")
    rows = read_jsonl(train_path)
    if len(rows) != int(audit["train_occurrences"]):
        raise ValueError("full runtime row count mismatch")
    if mode == "gold_control":
        raw_teacher = [0.0] * len(rows)
        rank_teacher = [0.0] * len(rows)
        target_binding = None
    else:
        assert args.teacher_target_dir is not None
        raw_teacher, rank_teacher, target_binding = load_targets(
            args.teacher_target_dir,
            rows,
            audit_path,
            train_path,
            allow_deadline_fast_targets=args.deadline_fast_track,
        )
    full_count = len(rows)
    if args.technical_smoke:
        teacher_for_mode = raw_teacher if mode == "hardneg_candidate" else rank_teacher
        selected = student.smoke_indices(mode, rows, teacher_for_mode)
        rows = [rows[index] for index in selected]
        raw_teacher = [raw_teacher[index] for index in selected]
        rank_teacher = [rank_teacher[index] for index in selected]

    torch.manual_seed(student.SEED)
    np.random.seed(student.SEED)
    random.seed(student.SEED)
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
    optimizer = torch.optim.AdamW(trainable, lr=student.LEARNING_RATE, weight_decay=0.01)
    indices = (
        list(range(len(rows)))
        if args.technical_smoke
        else student.frozen_order(len(rows))
    )
    micro_batches = math.ceil(len(indices) / student.MICRO_BATCH)
    updates = math.ceil(micro_batches / student.GRAD_ACCUM)
    warmup = max(1, int(updates * 0.05))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    optimizer_steps = 0
    weighted_occurrences = 0
    rank_pairs = 0
    loss_window: list[float] = []
    for step, offset in enumerate(
        range(0, len(indices), student.MICRO_BATCH), start=1
    ):
        local_indices = indices[offset : offset + student.MICRO_BATCH]
        local = [rows[index] for index in local_indices]
        local_raw = [raw_teacher[index] for index in local_indices]
        local_rank = [rank_teacher[index] for index in local_indices]
        namespaces = [SimpleNamespace(**row) for row in local]
        images = [student.open_image(row) for row in local]
        try:
            values = student.scores(
                model, student.batch_inputs(processor, namespaces, images), zero[0], one[0]
            )
            labels = torch.tensor(
                [row["label"] for row in local],
                dtype=torch.float32,
                device=values.device,
            )
            per_row = F.binary_cross_entropy_with_logits(
                values.float(), labels, reduction="none"
            )
            if mode == "gold_control":
                raw_loss = per_row.mean()
            elif mode == "hardneg_candidate":
                weights = torch.tensor(
                    [
                        student.hard_example_weight(row, score)
                        for row, score in zip(local, local_raw, strict=True)
                    ],
                    dtype=torch.float32,
                    device=values.device,
                )
                weighted_occurrences += int((weights > 1.0).sum().item())
                raw_loss = (per_row * weights).mean()
            elif mode == "rank_candidate":
                rank_loss, pairs = student.within_stratum_rank_loss(
                    values.float(), local, local_rank
                )
                rank_pairs += pairs
                raw_loss = per_row.mean() + student.RANK_COEFFICIENT * rank_loss
            else:  # pragma: no cover - argparse and explicit resolver fail closed
                raise AssertionError(mode)
            loss_window.append(float(raw_loss.detach().cpu()))
            (raw_loss / student.GRAD_ACCUM).backward()
        finally:
            for image in images:
                image.close()
        if step % student.GRAD_ACCUM == 0 or offset + student.MICRO_BATCH >= len(indices):
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        processed = min(offset + student.MICRO_BATCH, len(indices))
        if step % 50 == 0 or processed == len(indices):
            print(
                json.dumps(
                    {
                        "mode": mode,
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
    if mode == "hardneg_candidate" and weighted_occurrences == 0:
        raise ValueError("full hard-negative objective never exercised")
    if mode == "rank_candidate" and rank_pairs == 0:
        raise ValueError("full rank objective never exercised")
    adapter = args.output_dir / "adapter"
    model.save_pretrained(adapter)
    adapter_path = adapter / "adapter_model.safetensors"
    report = {
        "schema_version": "exp715_full_refit_v1",
        "experiment_id": "715",
        "source_experiment_id": "698",
        "selected_mode": mode,
        "model": str(MODEL),
        "image_preprocessing": student.IMAGE_PREPROCESSING,
        "full_data": True,
        "technical_smoke": args.technical_smoke,
        "train_occurrences": len(rows),
        "full_train_occurrences": full_count,
        "optimizer_steps": optimizer_steps,
        "weighted_occurrences": weighted_occurrences,
        "rank_pairs": rank_pairs,
        "runtime_minutes": (time.monotonic() - started) / 60,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "adapter_model_sha256": sha256(adapter_path),
        "adapter_bytes": adapter_path.stat().st_size,
        "teacher_required_at_inference": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
        "winner_binding": winner_binding,
        "deadline_fast_track": args.deadline_fast_track,
        "runtime_audit_sha256": sha256(audit_path),
        "target_binding": target_binding,
    }
    report["contract_sha256"] = student.canonical_sha256(report)
    (args.output_dir / "output_contract.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

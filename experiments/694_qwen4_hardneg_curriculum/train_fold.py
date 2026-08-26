from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))
CONSUMER = Path(__file__).resolve().parents[1] / "693_qwen4_causal_distillation"
if str(CONSUMER) not in sys.path:
    sys.path.insert(0, str(CONSUMER))

import train_lora as control
from exp691_consumer import load_fold

EXPERIMENT_ID = "694"
SOURCE_EXPERIMENT_ID = "641"
FLAMMABLE = "Легковоспламеняющиеся"
MAX_HARD_EXAMPLE_WEIGHT = 2.0
MODES = ("hard_bce_control", "hardneg_candidate")


def reset_cuda_peak_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def measured_cuda_peak_memory_bytes() -> int:
    import torch

    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def preserve_frozen_base_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Teacher hardness changes only BCE weights, never the parent row order."""
    return list(rows)


def frozen_hard_example_weight(row: Any) -> float:
    """Use teacher disagreement/confidence only as a frozen hard-gold BCE weight."""
    if row.category != FLAMMABLE:
        return 1.0
    score = float(row.teacher_score)
    label = int(row.label)
    if label not in (0, 1):
        raise ValueError("hard label must be binary")
    disagrees = (score >= 0.0) != bool(label)
    if not disagrees:
        return 1.0
    confidence = min(abs(score), 3.0) / 3.0
    return 1.0 + confidence * (MAX_HARD_EXAMPLE_WEIGHT - 1.0)


def weighted_hard_bce_loss(model, processor, rows, images, zero_token, one_token):
    import torch
    from torch.nn import functional

    conversations = [
        control.messages(row, image, prompt_text=control.base_prompt(row))
        for row, image in zip(rows, images, strict=True)
    ]
    batch = control._processor_batch(processor, conversations, add_generation_prompt=True).to(
        model.device
    )
    scores = control._last_logits(model, batch, zero_token, one_token).float()
    labels = torch.tensor(
        [int(row.label) for row in rows], dtype=torch.float32, device=model.device
    )
    weights = torch.tensor(
        [frozen_hard_example_weight(row) for row in rows],
        dtype=torch.float32,
        device=model.device,
    )
    per_row = functional.binary_cross_entropy_with_logits(scores, labels, reduction="none")
    return (per_row * weights).mean()


def run(args: Any) -> dict[str, Any]:
    if args.mode not in MODES:
        raise ValueError("unknown mode")
    original_load = control.load_runtime
    original_loss = control.primary_loss

    teacher_binding: dict[str, Any] = {}
    smoke_diagnostics: dict[str, Any] = {}

    def load_runtime(runtime_dir: Path, spec_id: str, fold: int):
        train, validation, audit = original_load(runtime_dir, spec_id, fold)
        train, binding = load_fold(
            args.teacher_root,
            fold=fold,
            train=train,
            runtime_contract_sha256=audit["contract_sha256"],
            require_evidence=False,
            acceptance_path=args.teacher_acceptance,
            expected_acceptance_file_sha256=args.teacher_acceptance_sha256,
            winner_path=args.teacher_winner,
            expected_winner_file_sha256=args.teacher_winner_sha256,
        )
        teacher_binding.update(binding)
        if args.technical_smoke and args.mode == "hardneg_candidate":
            hard = [row for row in train if frozen_hard_example_weight(type("R", (), row)()) > 1.0]
            if not hard:
                raise ValueError("hardneg smoke lacks a flammable teacher-disagreement row")
            hard_ids = {id(row) for row in hard}
            train = hard + [row for row in train if id(row) not in hard_ids]
        # The candidate retains the exact frozen parent order. Its only change
        # is fixed per-occurrence hard-BCE weighting.
        return preserve_frozen_base_order(train), validation, audit

    control.load_runtime = load_runtime
    if args.mode == "hardneg_candidate":

        def tracked_weighted_loss(model, processor, rows, images, zero_token, one_token):
            loss = weighted_hard_bce_loss(model, processor, rows, images, zero_token, one_token)
            if args.technical_smoke:
                weights = [frozen_hard_example_weight(row) for row in rows]
                smoke_diagnostics.update(
                    {
                        "weighted_rows": sum(weight > 1.0 for weight in weights),
                        "max_weight": max(weights),
                        "weighted_bce_loss": float(loss.detach().float().item()),
                    }
                )
            return loss

        control.primary_loss = tracked_weighted_loss
    reset_cuda_peak_memory()
    try:
        report = control.run(SOURCE_EXPERIMENT_ID, args)
        peak_gpu_memory_bytes = measured_cuda_peak_memory_bytes()
    finally:
        control.load_runtime = original_load
        control.primary_loss = original_loss
    if (
        args.technical_smoke
        and args.mode == "hardneg_candidate"
        and (
            smoke_diagnostics.get("weighted_rows", 0) < 1
            or float(smoke_diagnostics.get("max_weight", 1.0)) <= 1.0
            or not math.isfinite(float(smoke_diagnostics.get("weighted_bce_loss", math.nan)))
        )
    ):
        raise ValueError("hardneg changed-factor smoke did not exercise finite weighted BCE")
    report.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "mode": args.mode,
            "hard_bce_scope": "all_categories",
            "hard_bce_coefficient": 1.0,
            "hard_gold_targets_only": True,
            "kd_category": FLAMMABLE,
            "changed_factor": (
                "frozen_teacher_disagreement_confidence_weights"
                if args.mode == "hardneg_candidate"
                else "none"
            ),
            "paired_order": "identical_frozen_parent_order",
            "max_hard_example_weight": (
                MAX_HARD_EXAMPLE_WEIGHT if args.mode == "hardneg_candidate" else 1.0
            ),
            "teacher_outer_safe_required": True,
            "exp691_binding": teacher_binding,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "changed_factor_smoke": smoke_diagnostics if args.technical_smoke else None,
        }
    )
    report.pop("contract_sha256", None)
    report["contract_sha256"] = control.canonical_sha256(report)
    (args.output_dir / "output_contract.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = control.parser_for(SOURCE_EXPERIMENT_ID)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--teacher-acceptance", type=Path, required=True)
    parser.add_argument("--teacher-acceptance-sha256", required=True)
    parser.add_argument("--teacher-winner", type=Path, required=True)
    parser.add_argument("--teacher-winner-sha256", required=True)
    parsed = parser.parse_args()
    print(json.dumps(run(parsed), ensure_ascii=False, indent=2, sort_keys=True))

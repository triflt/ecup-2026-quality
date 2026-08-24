from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import train_lora as control


EXPERIMENT_ID = "681"
CONTROL_EXPERIMENT_ID = "641"
FLAMMABLE = "Легковоспламеняющиеся"
TEMPERATURE = 2.0
SOFT_LOSS_WEIGHT = 0.5


def distillation_primary_loss(
    model: Any,
    processor: Any,
    rows: list[SimpleNamespace],
    images: list[Any],
    zero_token: int,
    one_token: int,
):
    import torch
    from torch.nn import functional

    if any(row.category != FLAMMABLE for row in rows):
        raise ValueError("distillation batch contains a non-flammable row")
    teacher_values = [float(row.teacher_score) for row in rows]
    if not all(math.isfinite(value) for value in teacher_values):
        raise ValueError("distillation batch contains a non-finite teacher score")
    conversations = [
        control.messages(row, image, prompt_text=control.base_prompt(row))
        for row, image in zip(rows, images, strict=True)
    ]
    batch = control._processor_batch(
        processor, conversations, add_generation_prompt=True
    ).to(model.device)
    student = control._last_logits(model, batch, zero_token, one_token).float()
    hard_targets = torch.tensor(
        [int(row.label) for row in rows], dtype=torch.float32, device=model.device
    )
    teacher = torch.tensor(teacher_values, dtype=torch.float32, device=model.device)
    hard = functional.binary_cross_entropy_with_logits(student, hard_targets)
    soft_targets = torch.sigmoid(teacher / TEMPERATURE)
    soft = functional.binary_cross_entropy_with_logits(
        student / TEMPERATURE, soft_targets
    ) * (TEMPERATURE**2)
    return (1.0 - SOFT_LOSS_WEIGHT) * hard + SOFT_LOSS_WEIGHT * soft


def rewrite_contract(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("experiment_id") != CONTROL_EXPERIMENT_ID:
        raise ValueError("unexpected control report")
    if report.get("model_id") != "Qwen/Qwen3.5-4B" or report.get("objective") != "class_only":
        raise ValueError("unexpected model/objective")
    source_hash = report.get("contract_sha256")
    result = dict(report)
    result.pop("contract_sha256")
    result.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "control_experiment_id": CONTROL_EXPERIMENT_ID,
            "changed_factor": "hard_bce_to_fixed_hard_plus_teacher_soft_bce",
            "category_scope": FLAMMABLE,
            "temperature": TEMPERATURE,
            "soft_loss_weight": SOFT_LOSS_WEIGHT,
            "source_control_contract_sha256": source_hash,
            "teacher_target_semantics": "raw_last_token_logit_1_minus_logit_0",
            "teacher_target_scope": "outer_train_in_sample_outer_validation_unread",
            "ordinary_oof_merge_used": False,
            "submission_base_model": "Qwen/Qwen3.5-4B",
            "uses_27b_at_inference": False,
        }
    )
    result["contract_sha256"] = control.canonical_sha256(result)
    return result


def run(args: Any) -> dict[str, Any]:
    original = control.primary_loss
    control.primary_loss = distillation_primary_loss
    try:
        parent = control.run(CONTROL_EXPERIMENT_ID, args)
    finally:
        control.primary_loss = original
    result = rewrite_contract(parent)
    (args.output_dir / "output_contract.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    args = control.parser_for(CONTROL_EXPERIMENT_ID).parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


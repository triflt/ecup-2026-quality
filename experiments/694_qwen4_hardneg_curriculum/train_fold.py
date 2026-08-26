from __future__ import annotations

import json
import random
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


def curriculum_key(row: dict[str, Any]) -> tuple[int, float, int]:
    if row["category"] != FLAMMABLE:
        stage, hardness = 3, 0.0
    else:
        score = float(row["teacher_score"])
        label = int(row["label"])
        if label == 0:
            stage, hardness = 0, -score  # strongest teacher false-positive risk first
        elif label == 1:
            stage, hardness = 1, score  # strongest missed-positive risk first
        else:
            raise ValueError("hard label must be binary")
    return stage, hardness, int(row["global_index"])


def arrange_for_frozen_shuffle(
    rows: list[dict[str, Any]], *, seed: int = 42
) -> list[dict[str, Any]]:
    """Invert the parent's fixed shuffle so its observed sequence is the curriculum."""
    desired = sorted(rows, key=curriculum_key)
    permutation = list(range(len(rows)))
    random.Random(seed).shuffle(permutation)
    arranged: list[dict[str, Any] | None] = [None] * len(rows)
    for desired_index, source_index in enumerate(permutation):
        arranged[source_index] = desired[desired_index]
    return [row for row in arranged if row is not None]


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
        )
        teacher_binding.update(binding)
        # Both paired arms receive the exact same teacher-guided order.  The
        # candidate differs only by its fixed per-occurrence hard-BCE weights.
        return arrange_for_frozen_shuffle(train), validation, audit

    control.load_runtime = load_runtime
    if args.mode == "hardneg_candidate":
        control.primary_loss = weighted_hard_bce_loss
    try:
        report = control.run(SOURCE_EXPERIMENT_ID, args)
    finally:
        control.load_runtime = original_load
        control.primary_loss = original_loss
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
            "paired_order": "identical_teacher_guided_curriculum",
            "max_hard_example_weight": (
                MAX_HARD_EXAMPLE_WEIGHT if args.mode == "hardneg_candidate" else 1.0
            ),
            "teacher_outer_safe_required": True,
            "exp691_binding": teacher_binding,
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
    parsed = parser.parse_args()
    print(json.dumps(run(parsed), ensure_ascii=False, indent=2, sort_keys=True))

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))
CONSUMER = Path(__file__).resolve().parents[1] / "693_qwen4_causal_distillation"
if str(CONSUMER) not in sys.path:
    sys.path.insert(0, str(CONSUMER))

import train_lora as control
from exp691_consumer import load_fold

EXPERIMENT_ID = "695"
SOURCE_EXPERIMENT_ID = "641"
FLAMMABLE = "Легковоспламеняющиеся"
RANK_COEFFICIENT = 0.50
RANK_CAP_FRACTION = 0.25
MODES = ("hard_bce_control", "rank_candidate")


def within_stratum_pairs(rows: list[SimpleNamespace]) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            a, b = rows[left], rows[right]
            if a.category != FLAMMABLE or b.category != FLAMMABLE:
                continue
            if int(a.label) != int(b.label):
                continue
            delta = float(a.teacher_score) - float(b.teacher_score)
            if delta != 0.0:
                pairs.append((left, right, 1.0 if delta > 0 else -1.0))
    return pairs


def arrange_matched_batches(rows: list[dict[str, Any]], *, seed: int = 42) -> list[dict[str, Any]]:
    """Give both arms identical microbatches with same-label flammable pairs."""
    paired: list[dict[str, Any]] = []
    residual: list[dict[str, Any]] = []
    consumed: set[int] = set()
    for label in (0, 1):
        group = sorted(
            (row for row in rows if row["category"] == FLAMMABLE and int(row["label"]) == label),
            key=lambda row: int(row["global_index"]),
        )
        even = len(group) - len(group) % 2
        paired.extend(group[:even])
        residual.extend(group[even:])
        consumed.update(id(row) for row in group)
    residual.extend(row for row in rows if id(row) not in consumed)
    desired = paired + residual
    permutation = list(range(len(rows)))
    random.Random(seed).shuffle(permutation)
    arranged: list[dict[str, Any] | None] = [None] * len(rows)
    for desired_index, source_index in enumerate(permutation):
        arranged[source_index] = desired[desired_index]
    return [row for row in arranged if row is not None]


def bounded_listwise_loss(scores: Any, rows: list[SimpleNamespace], hard: Any):
    import torch
    from torch.nn import functional

    pairs = within_stratum_pairs(rows)
    if not pairs:
        return scores.sum() * 0.0
    losses = [
        functional.softplus(-sign * (scores[left] - scores[right])) for left, right, sign in pairs
    ]
    raw = torch.stack(losses).mean() * RANK_COEFFICIENT
    return torch.minimum(raw, hard.detach() * RANK_CAP_FRACTION)


def candidate_loss(model, processor, rows, images, zero_token, one_token):
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
    hard = functional.binary_cross_entropy_with_logits(scores, labels)
    return hard + bounded_listwise_loss(scores, rows, hard)


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
        return arrange_matched_batches(train), validation, audit

    control.load_runtime = load_runtime
    if args.mode == "rank_candidate":
        control.primary_loss = candidate_loss
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
            "rank_coefficient": RANK_COEFFICIENT if args.mode == "rank_candidate" else 0.0,
            "rank_scope": "flammable_same_hard_label_only",
            "rank_cap_fraction_of_hard": RANK_CAP_FRACTION,
            "lambda_grid": False,
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

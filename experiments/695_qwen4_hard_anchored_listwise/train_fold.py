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
CONSUMER = Path(__file__).resolve().parents[1] / "693_qwen4_causal_distillation"
if str(CONSUMER) not in sys.path:
    sys.path.insert(0, str(CONSUMER))

import train_lora as control
from exp691_consumer import load_fold

EXPERIMENT_ID = "695"
SOURCE_EXPERIMENT_ID = "641"
FLAMMABLE = "Легковоспламеняющиеся"
RANK_COEFFICIENT = 0.50
MODES = ("hard_bce_control", "rank_candidate")


def reset_cuda_peak_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def measured_cuda_peak_memory_bytes() -> int:
    import torch

    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


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


def preserve_frozen_base_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank pairs are selected inside each batch without reordering base rows."""
    return list(rows)


def listwise_loss(scores: Any, rows: list[SimpleNamespace]):
    import torch
    from torch.nn import functional

    pairs = within_stratum_pairs(rows)
    if not pairs:
        return scores.sum() * 0.0
    losses = [
        functional.softplus(-sign * (scores[left] - scores[right])) for left, right, sign in pairs
    ]
    return torch.stack(losses).mean()


def candidate_loss(
    model,
    processor,
    rows,
    images,
    zero_token,
    one_token,
    *,
    diagnostics: dict[str, Any] | None = None,
):
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
    rank = listwise_loss(scores, rows)
    if diagnostics is not None:
        diagnostics.update(
            {
                "pair_count": len(within_stratum_pairs(rows)),
                "hard_loss": float(hard.detach().float().item()),
                "rank_loss": float(rank.detach().float().item()),
            }
        )
    return hard + RANK_COEFFICIENT * rank


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
        if args.technical_smoke and args.mode == "rank_candidate":
            namespaces = [SimpleNamespace(**row) for row in train]
            pairs = within_stratum_pairs(namespaces)
            if not pairs:
                raise ValueError("rank smoke lacks a same-label teacher-ordered pair")
            left, right, _ = pairs[0]
            selected = [train[left], train[right]]
            selected_ids = {id(row) for row in selected}
            train = selected + [row for row in train if id(row) not in selected_ids]
        return preserve_frozen_base_order(train), validation, audit

    control.load_runtime = load_runtime
    if args.mode == "rank_candidate":

        def tracked_candidate_loss(model, processor, rows, images, zero_token, one_token):
            return candidate_loss(
                model,
                processor,
                rows,
                images,
                zero_token,
                one_token,
                diagnostics=smoke_diagnostics if args.technical_smoke else None,
            )

        control.primary_loss = tracked_candidate_loss
    reset_cuda_peak_memory()
    try:
        report = control.run(SOURCE_EXPERIMENT_ID, args)
        peak_gpu_memory_bytes = measured_cuda_peak_memory_bytes()
    finally:
        control.load_runtime = original_load
        control.primary_loss = original_loss
    if (
        args.technical_smoke
        and args.mode == "rank_candidate"
        and (
            smoke_diagnostics.get("pair_count", 0) < 1
            or not math.isfinite(float(smoke_diagnostics.get("hard_loss", math.nan)))
            or not math.isfinite(float(smoke_diagnostics.get("rank_loss", math.nan)))
            or float(smoke_diagnostics.get("rank_loss", 0.0)) <= 0.0
        )
    ):
        raise ValueError("rank changed-factor smoke did not exercise finite nonzero rank loss")
    report.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "mode": args.mode,
            "hard_bce_scope": "all_categories",
            "hard_bce_coefficient": 1.0,
            "rank_coefficient": RANK_COEFFICIENT if args.mode == "rank_candidate" else 0.0,
            "rank_scope": "flammable_same_hard_label_only",
            "rank_pair_selection": "deterministic_within_frozen_batch_without_reorder",
            "loss_formula": "L_hard + 0.5 * L_rank",
            "lambda_grid": False,
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

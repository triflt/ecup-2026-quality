from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import train_lora as control
from exp691_consumer import load_fold

EXPERIMENT_ID = "693"
SOURCE_EXPERIMENT_ID = "641"
FLAMMABLE = "Легковоспламеняющиеся"
AUXILIARY_COEFFICIENT = 0.10
MODES = ("hard_bce_control", "causal_candidate")


def structured_target(row: SimpleNamespace) -> str:
    evidence = dict(row.teacher_evidence)
    payload = {
        "verdict": evidence["verdict"],
        "sold_object": evidence["sold_object"],
        "substance": evidence["substance"],
        "relation": evidence["relation"],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def structured_auxiliary_loss(
    model: Any, processor: Any, rows: list[SimpleNamespace], images: list[Any]
):
    import torch

    selected = []
    for row, image in zip(rows, images, strict=True):
        evidence = getattr(row, "teacher_evidence", {})
        if (
            row.category == FLAMMABLE
            and evidence.get("parse_valid") is True
            and evidence.get("grounded") is True
            and evidence.get("abstain") is False
            and all(
                evidence.get(field, {}).get("source") != "unknown"
                and evidence.get(field, {}).get("value") != "unknown"
                for field in ("sold_object", "substance", "relation")
            )
        ):
            selected.append((row, image))
    if not selected:
        return None
    aux_rows, aux_images = zip(*selected, strict=True)
    answers = [structured_target(row) for row in aux_rows]
    prompt = "Верни только компактный JSON с ключами verdict, sold_object, substance, relation."
    conversations = [
        control.messages(
            row,
            image,
            prompt_text=control.base_prompt(row) + "\n" + prompt,
            answer=answer,
        )
        for row, image, answer in zip(aux_rows, aux_images, answers, strict=True)
    ]
    batch = control._processor_batch(processor, conversations, add_generation_prompt=False)
    labels = torch.full_like(batch["input_ids"], -100)
    for row_index, answer in enumerate(answers):
        active = torch.nonzero(batch["attention_mask"][row_index]).flatten().tolist()
        active_ids = batch["input_ids"][row_index, active].tolist()
        target_ids = processor.tokenizer.encode(answer, add_special_tokens=False)
        start = control._locate_last(active_ids, target_ids)
        for local_index, token_id in enumerate(target_ids):
            labels[row_index, active[start + local_index]] = int(token_id)
    device_batch = {key: value.to(model.device) for key, value in batch.items()}
    return model(**device_batch, labels=labels.to(model.device), use_cache=False).loss


def combine_losses(hard: Any, auxiliary: Any | None, *, mode: str) -> Any:
    if mode == "hard_bce_control":
        return hard
    if mode != "causal_candidate":
        raise ValueError("unknown mode")
    return hard if auxiliary is None else hard + AUXILIARY_COEFFICIENT * auxiliary


def run(args: Any) -> dict[str, Any]:
    if args.mode not in MODES:
        raise ValueError("unknown mode")
    original_load = control.load_runtime
    original_loss = control.primary_loss
    teacher_binding: dict[str, Any] = {}

    def load_runtime(runtime_dir: Path, spec_id: str, fold: int):
        train, validation, audit = original_load(runtime_dir, spec_id, fold)
        enriched, binding = load_fold(
            args.teacher_root,
            fold=fold,
            train=train,
            runtime_contract_sha256=audit["contract_sha256"],
            require_evidence=args.mode == "causal_candidate",
            acceptance_path=args.teacher_acceptance,
            expected_acceptance_file_sha256=args.teacher_acceptance_sha256,
        )
        teacher_binding.update(binding)
        return enriched, validation, audit

    def primary_loss(model, processor, rows, images, zero_token, one_token):
        hard = original_loss(model, processor, rows, images, zero_token, one_token)
        auxiliary = (
            structured_auxiliary_loss(model, processor, rows, images)
            if args.mode == "causal_candidate"
            else None
        )
        return combine_losses(hard, auxiliary, mode=args.mode)

    control.load_runtime = load_runtime
    control.primary_loss = primary_loss
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
            "kd_category": FLAMMABLE,
            "auxiliary_coefficient": (
                AUXILIARY_COEFFICIENT if args.mode == "causal_candidate" else 0.0
            ),
            "auxiliary_training_only": True,
            "teacher_outer_safe_required": True,
            "exp691_binding": teacher_binding,
        }
    )
    report.pop("contract_sha256", None)
    report["contract_sha256"] = control.canonical_sha256(report)
    (args.output_dir / "output_contract.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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

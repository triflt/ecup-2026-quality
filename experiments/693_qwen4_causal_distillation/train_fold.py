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

EXPERIMENT_ID = "693"
SOURCE_EXPERIMENT_ID = "641"
FLAMMABLE = "Легковоспламеняющиеся"
AUXILIARY_COEFFICIENT = 0.10
MODES = ("hard_bce_control", "causal_candidate")
TARGET_FIELDS = ("teacher_verdict", "sold_object", "substance", "relation", "evidence_ids")


def validate_teacher_contract(audit: dict[str, Any], fold: int) -> str:
    teacher = audit.get("teacher_contract", {})
    expected = {
        "schema": "qwen27_all_outer_safe_v1",
        "teacher_recipe": "qwen27-all",
        "outer_fold": fold,
        "train_scope": "outer_train_only",
        "outer_validation_labels_read": 0,
        "target_category": FLAMMABLE,
    }
    if any(teacher.get(key) != value for key, value in expected.items()):
        raise ValueError("teacher artifact contract is missing or not outer-safe")
    digest = teacher.get("artifact_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("teacher artifact SHA-256 is invalid")
    return digest


def validate_teacher_rows(
    train: list[dict[str, Any]], validation: list[dict[str, Any]], fold: int
) -> None:
    validation_keys = {(str(row["id"]), int(row["global_index"])) for row in validation}
    for row in validation:
        if any(field in row for field in TARGET_FIELDS):
            raise ValueError("outer-validation row contains forbidden teacher targets")
    for row in train:
        if row["category"] != FLAMMABLE:
            if any(field in row for field in TARGET_FIELDS):
                raise ValueError("KD target is present outside flammable")
            continue
        missing = [field for field in TARGET_FIELDS if field not in row]
        if missing:
            raise ValueError(f"flammable train row lacks teacher fields: {missing}")
        if (str(row["id"]), int(row["global_index"])) in validation_keys:
            raise ValueError("teacher target overlaps outer validation")
        if int(row.get("teacher_outer_fold", -1)) != fold:
            raise ValueError("teacher target is not bound to this outer fold")
        if int(row["teacher_verdict"]) not in (0, 1):
            raise ValueError("teacher verdict must be hard binary")
        if not isinstance(row["evidence_ids"], list):
            raise TypeError("evidence_ids must be a closed list")


def structured_target(row: SimpleNamespace) -> str:
    payload = {field: getattr(row, field) for field in TARGET_FIELDS}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def structured_auxiliary_loss(
    model: Any, processor: Any, rows: list[SimpleNamespace], images: list[Any]
):
    import torch

    selected = [
        (row, image) for row, image in zip(rows, images, strict=True) if row.category == FLAMMABLE
    ]
    if not selected:
        return None
    aux_rows, aux_images = zip(*selected, strict=True)
    answers = [structured_target(row) for row in aux_rows]
    prompt = (
        "Верни только компактный JSON с ключами teacher_verdict, sold_object, "
        "substance, relation, evidence_ids."
    )
    conversations = [
        control.messages(
            row, image, prompt_text=control.base_prompt(row) + "\n" + prompt, answer=answer
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

    teacher_binding: dict[str, str] = {}

    def load_runtime(runtime_dir: Path, spec_id: str, fold: int):
        train, validation, audit = original_load(runtime_dir, spec_id, fold)
        teacher_binding["artifact_sha256"] = validate_teacher_contract(audit, fold)
        validate_teacher_rows(train, validation, fold)
        return train, validation, audit

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
            "auxiliary_coefficient": AUXILIARY_COEFFICIENT
            if args.mode == "causal_candidate"
            else 0.0,
            "auxiliary_training_only": True,
            "teacher_outer_safe_required": True,
            "teacher_artifact_sha256": teacher_binding["artifact_sha256"],
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
    parsed = parser.parse_args()
    print(json.dumps(run(parsed), ensure_ascii=False, indent=2, sort_keys=True))

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from coverage_audit import build_coverage_audit
from parent_recipe import load_parent_module
from parent_selector import select_parent_training_records
from structured_target import FORMAT_VERSION, build_structured_target

_PARENT_ANSWER_INSTRUCTION = "Ответь только одной цифрой: 1 или 0."
_GROUNDED_ANSWER_INSTRUCTION = (
    "Сначала выведи ровно одну цифру 1 или 0. Затем на отдельных строках выведи "
    "CONCEPT, SOURCE, START, END и EVIDENCE по заданному закрытому формату."
)


def _load_expected_audit() -> dict[str, Any]:
    raw = os.environ.get("ECUP_GROUNDED_AUX_AUDIT_JSON")
    if not raw:
        raise ValueError("ECUP_GROUNDED_AUX_AUDIT_JSON is required; run coverage audit first")
    audit = json.loads(Path(raw).resolve().read_text(encoding="utf-8"))
    if audit.get("decision") != "GO":
        raise ValueError("coverage audit did not authorize launch")
    if audit.get("format_version") != FORMAT_VERSION:
        raise ValueError("coverage audit target format mismatch")
    return audit


def main() -> int:
    parent = load_parent_module()
    expected_audit = _load_expected_audit()
    if expected_audit.get("holdout_fold") != parent.HOLDOUT_FOLD:
        raise ValueError("coverage audit fold differs from training fold")

    original_user_text = parent.user_text
    original_messages = parent.messages
    original_training_batch = parent.training_batch
    original_select_training = parent.select_training

    def grounded_user_text(row):
        prompt = original_user_text(row)
        if not prompt.endswith(_PARENT_ANSWER_INSTRUCTION):
            raise ValueError("frozen parent answer instruction changed")
        return prompt[: -len(_PARENT_ANSWER_INSTRUCTION)] + _GROUNDED_ANSWER_INSTRUCTION

    def grounded_messages(row, with_answer=False, image=None):
        messages = original_messages(row, with_answer=False, image=image)
        if with_answer:
            target = build_structured_target(
                row_id=str(row.id),
                category=str(row.category),
                name=str(row["name"] or ""),
                description=str(row.description or ""),
                gold_verdict=int(row.label),
            )
            messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
        return messages

    def grounded_training_batch(processor, rows):
        batch = original_training_batch(processor, rows)
        for row_index, row in enumerate(rows):
            supervised = batch["labels"][row_index].ne(-100).nonzero(as_tuple=False).flatten()
            if len(supervised) == 0:
                raise ValueError(f"structured target was truncated for id={row.id}")
            actual = int(batch["labels"][row_index, supervised[0]])
            expected_ids = processor.tokenizer.encode(str(int(row.label)), add_special_tokens=False)
            if len(expected_ids) != 1 or actual != expected_ids[0]:
                raise ValueError(
                    f"first supervised token is not the atomic verdict for id={row.id}"
                )
        return batch

    def select_training_with_grounded_audit(frame, oof):
        records, parent_audit = original_select_training(frame, oof)
        light_records, light_parent_audit = select_parent_training_records(
            frame,
            oof,
            seed=parent.SEED,
            holdout_fold=parent.HOLDOUT_FOLD,
            full_train=parent.FULL_TRAIN,
        )
        if records != light_records:
            raise ValueError("dependency-light selector differs from frozen parent record order")
        runtime_audit = build_coverage_audit(
            frame,
            oof,
            records,
            holdout_fold=parent.HOLDOUT_FOLD,
            seed=parent.SEED,
            parent_selection=light_parent_audit,
        )
        frozen_keys = (
            "format_version",
            "holdout_fold",
            "seed",
            "training_records",
            "training_unique_rows",
            "record_multiset_sha256",
            "target_plan_sha256",
            "cohorts",
            "overall",
            "decision",
        )
        # NumPy may return the same sampled set in a different order across
        # versions.  The true parent and the dependency-light selector are
        # still required to agree exactly inside this runtime (checked above),
        # while the cross-runtime gate is the record multiset, target plan and
        # cohorts.  Batch order remains the unmodified parent order.
        mismatches = {
            key: {"expected": expected_audit.get(key), "actual": runtime_audit.get(key)}
            for key in frozen_keys
            if expected_audit.get(key) != runtime_audit.get(key)
        }
        if mismatches:
            raise ValueError(f"frozen grounded-target audit mismatch: {mismatches}")
        parent.OUTPUT.mkdir(parents=True, exist_ok=True)
        runtime_audit["true_parent_selection"] = parent_audit
        (parent.OUTPUT / "grounded_target_audit.runtime.json").write_text(
            json.dumps(runtime_audit, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        parent_audit = dict(parent_audit)
        parent_audit["grounded_auxiliary_sft"] = {
            "format_version": FORMAT_VERSION,
            "target_plan_sha256": runtime_audit["target_plan_sha256"],
            "safe_occurrence_rate": runtime_audit["overall"]["safe_occurrence_rate"],
        }
        return records, parent_audit

    parent.user_text = grounded_user_text
    parent.messages = grounded_messages
    parent.training_batch = grounded_training_batch
    parent.select_training = select_training_with_grounded_audit
    parent.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

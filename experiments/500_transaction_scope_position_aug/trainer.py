from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from parent_recipe import load_parent_module
from parent_selector import select_parent_training_records
from transaction_scope_position_aug import (
    apply_augmentation_plan,
    build_augmentation_plan,
)


def _load_expected_audit() -> dict[str, Any]:
    raw = os.environ.get("ECUP_AUGMENT_AUDIT_JSON")
    if not raw:
        raise ValueError("ECUP_AUGMENT_AUDIT_JSON is required; run build_manifest.py first")
    path = Path(raw).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parent = load_parent_module()
    expected_audit = _load_expected_audit()
    original_select_training = parent.select_training

    def select_training_with_position_augmentation(frame, oof):
        records, parent_audit = original_select_training(frame, oof)
        dependency_light_records, _ = select_parent_training_records(
            frame,
            oof,
            seed=parent.SEED,
            holdout_fold=parent.HOLDOUT_FOLD,
            full_train=parent.FULL_TRAIN,
        )
        if records != dependency_light_records:
            raise ValueError(
                "dependency-light audit selector differs from the frozen parent record order"
            )
        transforms, manifest, audit = build_augmentation_plan(
            frame,
            records,
            oof["fold_ids"].astype("int8"),
            holdout_fold=parent.HOLDOUT_FOLD,
            full_train=parent.FULL_TRAIN,
        )
        frozen_keys = (
            "augmentation_version",
            "outer_fold",
            "full_train",
            "parent_record_multiset_sha256",
            "parent_record_order_sha256",
            "manifest_sha256",
            "plan_sha256",
        )
        mismatches = {
            key: {"expected": expected_audit.get(key), "actual": audit.get(key)}
            for key in frozen_keys
            if expected_audit.get(key) != audit.get(key)
        }
        if mismatches:
            raise ValueError(f"frozen augmentation audit mismatch: {mismatches}")
        apply_augmentation_plan(frame, transforms)
        parent.OUTPUT.mkdir(parents=True, exist_ok=True)
        (parent.OUTPUT / "augmentation_manifest.runtime.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
            encoding="utf-8",
        )
        runtime_audit = dict(audit)
        runtime_audit["parent_selection"] = parent_audit
        (parent.OUTPUT / "augmentation_audit.runtime.json").write_text(
            json.dumps(runtime_audit, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        parent_audit = dict(parent_audit)
        parent_audit["transaction_scope_position_aug"] = audit
        return records, parent_audit

    parent.select_training = select_training_with_position_augmentation
    parent.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

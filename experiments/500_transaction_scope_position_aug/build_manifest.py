from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from parent_selector import select_parent_training_records
from transaction_scope_position_aug import build_augmentation_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and audit the frozen exp500 outer-train augmentation plan."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-train", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    oof_path = args.oof.resolve()
    output = args.output_dir.resolve()
    seed = int(args.seed)
    holdout_fold = int(args.fold)
    full_train = bool(args.full_train)

    frame = pd.read_csv(data_path)
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    oof = np.load(oof_path, allow_pickle=True)
    ids = frame["id"].astype(str).to_numpy()
    if not np.array_equal(ids, oof["ids"].astype(str)):
        raise ValueError("OOF id mismatch")
    records, parent_audit = select_parent_training_records(
        frame,
        oof,
        seed=seed,
        holdout_fold=holdout_fold,
        full_train=full_train,
    )
    _, manifest, audit = build_augmentation_plan(
        frame,
        records,
        oof["fold_ids"].astype(np.int8),
        holdout_fold=holdout_fold,
        full_train=full_train,
    )
    audit["parent_selection"] = parent_audit
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "augmentation_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
        encoding="utf-8",
    )
    audit_path = output / "augmentation_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audit": str(audit_path),
                "manifest": str(manifest_path),
                "plan_sha256": audit["plan_sha256"],
                "training_records": audit["training_records"],
                "candidate_unique_rows": audit["candidate_unique_rows"],
                "candidate_training_occurrences": audit["candidate_training_occurrences"],
                "eligible_unique_rows": audit["eligible_unique_rows"],
                "transformed_unique_rows": audit["transformed_unique_rows"],
                "eligible_training_occurrences": audit["eligible_training_occurrences"],
                "transformed_training_occurrences": audit["transformed_training_occurrences"],
                "observed_unique_row_selection_rate": audit["observed_unique_row_selection_rate"],
                "observed_occurrence_weighted_selection_rate": audit[
                    "observed_occurrence_weighted_selection_rate"
                ],
                "eligible_unique_row_coverage": audit["eligible_unique_row_coverage"],
                "transformed_unique_row_coverage": audit["transformed_unique_row_coverage"],
                "eligible_occurrence_weighted_coverage": audit[
                    "eligible_occurrence_weighted_coverage"
                ],
                "transformed_occurrence_weighted_coverage": audit[
                    "transformed_occurrence_weighted_coverage"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

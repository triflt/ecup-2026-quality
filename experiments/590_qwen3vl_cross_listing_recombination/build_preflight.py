from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from contract import SCREEN_FOLDS, SEED
from parent_selector import select_parent_training_records
from recombination_plan import (
    build_fold_plan,
    canonical_sha256,
    deterministic_blind_sample,
    eligible_outer_pairs,
    evaluate_reviews,
    file_sha256,
    load_frozen_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fail-closed exp590 CPU preflight.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--sealed-dir", required=True, type=Path)
    parser.add_argument("--draft-dir", required=True, type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path, oof_path = args.data.resolve(), args.oof.resolve()
    frame = pd.read_csv(data_path, dtype={"id": str})
    frame["name"] = frame.name.fillna("").astype(str)
    frame["description"] = frame.description.fillna("").astype(str)
    frame["category"] = frame.category.astype(str)
    oof = np.load(oof_path, allow_pickle=True)
    if not np.array_equal(frame.id.astype(str), oof["ids"].astype(str)):
        raise ValueError("OOF id mismatch")
    _, pairs, provenance = load_frozen_provenance(
        args.sealed_dir.resolve(), args.draft_dir.resolve(), frame
    )
    fold_by_id = dict(zip(frame.id.astype(str), oof["fold_ids"].astype(np.int8), strict=True))
    screen_union = {
        (pair.left_id, pair.right_id): pair
        for fold in SCREEN_FOLDS
        for pair in eligible_outer_pairs(pairs, fold_by_id, fold)
    }
    sample = deterministic_blind_sample(list(screen_union.values()))
    review_audit = evaluate_reviews(sample, args.reviews.resolve() if args.reviews else None)
    review_audit["forbidden_edge_count"] = sum(
        pair.stage not in provenance["allowed_stages"]
        or pair.key_degree > provenance["maximum_key_degree"]
        or "generic" in pair.corroboration.lower()
        for pair in sample
    )
    if review_audit["forbidden_edge_count"]:
        review_audit["decision"] = "NO_GO"
        review_audit["reason"] = "blind sample contains forbidden provenance"
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite frozen preflight artifacts")
    output.mkdir(parents=True)
    sample_rows = [
        {
            "audit_id": f"R{index:03d}",
            "left_id": pair.left_id,
            "right_id": pair.right_id,
            "semantic_component": pair.semantic_component,
            "stage": pair.stage,
            "key_degree": pair.key_degree,
            "corroboration": pair.corroboration,
        }
        for index, pair in enumerate(sample, 1)
    ]
    sample_frame = pd.DataFrame(sample_rows)
    sample_frame.to_csv(output / "blind_audit_pairs.csv", index=False)
    template = sample_frame.copy()
    template["reviewer_a_same_product"] = ""
    template["reviewer_b_same_product"] = ""
    template.to_csv(output / "blind_review_template.csv", index=False)
    fold_decisions = {}
    for fold in SCREEN_FOLDS:
        records = select_parent_training_records(frame, oof, seed=SEED, holdout_fold=fold)
        manifest, audit = build_fold_plan(
            frame,
            oof,
            records,
            pairs,
            holdout_fold=fold,
            review_audit=review_audit,
        )
        audit["provenance"] = provenance
        audit["data_sha256"] = file_sha256(data_path)
        audit["oof_sha256"] = file_sha256(oof_path)
        audit["blind_sample_sha256"] = canonical_sha256(sample_rows)
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir()
        (fold_dir / "recombination_manifest.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
            encoding="utf-8",
        )
        (fold_dir / "preflight_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fold_decisions[str(fold)] = {
            "decision": audit["decision"],
            "eligible_pairs": audit["eligible_pairs"],
            "eligible_components": audit["eligible_components"],
            "recombined_occurrences": audit["recombined_occurrences"],
        }
    summary = {
        "experiment_id": "590",
        "seed": SEED,
        "screen_folds": list(SCREEN_FOLDS),
        "provenance": provenance,
        "blind_sample_pairs": len(sample_rows),
        "blind_sample_sha256": canonical_sha256(sample_rows),
        "blind_audit": review_audit,
        "folds": fold_decisions,
        "decision": (
            "GO"
            if review_audit["decision"] == "GO"
            and all(item["decision"] == "GO" for item in fold_decisions.values())
            else "NO_GO"
        ),
    }
    (output / "preflight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from parent_selector import select_parent_training_records
from structured_target import (
    FORMAT_VERSION,
    NO_SAFE_EVIDENCE,
    build_structured_target,
    parse_first_atomic_verdict,
    parse_structured_target,
    validate_exact_evidence,
)

MIN_OVERALL_SAFE_OCCURRENCE_RATE = 0.50
MIN_COHORT_SAFE_OCCURRENCE_RATE = 0.25
MIN_COHORT_SAFE_OCCURRENCES = 25


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _cross_check_rows(path: Path | None) -> dict[str, dict[str, Any]] | None:
    if path is None:
        return None
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        row_id = str(item["row_id"])
        if row_id in rows:
            raise ValueError(f"duplicate cross-check row id: {row_id}")
        rows[row_id] = item
    return rows


def build_coverage_audit(
    frame: pd.DataFrame,
    oof,
    records: list[int],
    *,
    holdout_fold: int,
    seed: int,
    parent_selection: dict[str, Any],
    cross_check: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    counts = Counter(int(index) for index in records)
    fold_ids = oof["fold_ids"].astype(np.int8)
    holdout_occurrences = sum(
        occurrence for index, occurrence in counts.items() if int(fold_ids[index]) == holdout_fold
    )
    target_fingerprints: list[tuple[int, int, str]] = []
    cohort_rows: dict[tuple[str, int], list[tuple[int, int, bool]]] = {}
    cross_check_mismatches: list[str] = []
    format_failures: list[str] = []
    for index in sorted(counts):
        row = frame.iloc[index]
        category = str(row["category"])
        label = int(row["label"])
        row_id = str(row["id"])
        target = build_structured_target(
            row_id=row_id,
            category=category,
            name=str(row["name"]),
            description=str(row["description"]),
            gold_verdict=label,
        )
        try:
            parsed = parse_structured_target(target)
            if parse_first_atomic_verdict(target) != label:
                raise ValueError("first verdict differs from gold")
            if not validate_exact_evidence(
                parsed,
                name=str(row["name"]),
                description=str(row["description"]),
            ):
                raise ValueError("surface offsets do not recover exact evidence")
        except ValueError as error:
            format_failures.append(f"{row_id}:{error}")
            continue
        safe = parsed.concept != NO_SAFE_EVIDENCE
        occurrences = int(counts[index])
        cohort_rows.setdefault((category, label), []).append((index, occurrences, safe))
        target_fingerprints.append(
            (index, occurrences, hashlib.sha256(target.encode()).hexdigest())
        )
        if cross_check is not None:
            expected = cross_check.get(row_id)
            expected_safe = expected is not None and expected.get("status") == "SAFE"
            expected_concept = expected.get("concept") if expected_safe else NO_SAFE_EVIDENCE
            expected_span = expected.get("exact_surface_span") if expected_safe else ""
            if (
                expected is None
                or int(expected["frozen_prediction"]) != label
                or parsed.concept != expected_concept
                or parsed.evidence != expected_span
            ):
                cross_check_mismatches.append(row_id)

    categories = sorted(frame["category"].astype(str).unique())
    cohorts: dict[str, dict[str, Any]] = {}
    gate_failures: list[str] = []
    total_unique = total_occurrences = total_safe_unique = total_safe_occurrences = 0
    for category in categories:
        cohorts[category] = {}
        for label in (0, 1):
            local = cohort_rows.get((category, label), [])
            unique = len(local)
            occurrences = sum(item[1] for item in local)
            safe_unique = sum(item[2] for item in local)
            safe_occurrences = sum(item[1] for item in local if item[2])
            safe_unique_rate = safe_unique / unique if unique else 0.0
            safe_occurrence_rate = safe_occurrences / occurrences if occurrences else 0.0
            cohorts[category][str(label)] = {
                "unique_rows": unique,
                "training_occurrences": occurrences,
                "safe_unique_rows": safe_unique,
                "safe_training_occurrences": safe_occurrences,
                "fallback_unique_rows": unique - safe_unique,
                "fallback_training_occurrences": occurrences - safe_occurrences,
                "safe_unique_rate": safe_unique_rate,
                "safe_occurrence_rate": safe_occurrence_rate,
            }
            total_unique += unique
            total_occurrences += occurrences
            total_safe_unique += safe_unique
            total_safe_occurrences += safe_occurrences
            cohort_key = f"{category}/label={label}"
            if safe_occurrences < MIN_COHORT_SAFE_OCCURRENCES:
                gate_failures.append(f"{cohort_key}:too_few_safe_occurrences")
            if safe_occurrence_rate < MIN_COHORT_SAFE_OCCURRENCE_RATE:
                gate_failures.append(f"{cohort_key}:safe_rate_below_floor")

    overall_safe_rate = total_safe_occurrences / total_occurrences if total_occurrences else 0.0
    if overall_safe_rate < MIN_OVERALL_SAFE_OCCURRENCE_RATE:
        gate_failures.append("overall:fallback_targets_swamp_safe_signal")
    if holdout_occurrences:
        gate_failures.append("outer_validation_entered_training_records")
    if format_failures:
        gate_failures.append("structured_target_invariant_failure")
    if cross_check_mismatches:
        gate_failures.append("exp490_cross_check_mismatch")

    audit = {
        "experiment_id": "520",
        "format_version": FORMAT_VERSION,
        "holdout_fold": int(holdout_fold),
        "seed": int(seed),
        "training_records": len(records),
        "training_unique_rows": len(counts),
        "ordered_records_sha256": canonical_sha256([int(index) for index in records]),
        "record_multiset_sha256": canonical_sha256(sorted(counts.items())),
        "target_plan_sha256": canonical_sha256(target_fingerprints),
        "outer_validation_training_occurrences": int(holdout_occurrences),
        "cohorts": cohorts,
        "overall": {
            "unique_rows": total_unique,
            "training_occurrences": total_occurrences,
            "safe_unique_rows": total_safe_unique,
            "safe_training_occurrences": total_safe_occurrences,
            "fallback_unique_rows": total_unique - total_safe_unique,
            "fallback_training_occurrences": total_occurrences - total_safe_occurrences,
            "safe_unique_rate": total_safe_unique / total_unique if total_unique else 0.0,
            "safe_occurrence_rate": overall_safe_rate,
        },
        "format_failures": format_failures,
        "cross_check": {
            "enabled": cross_check is not None,
            "selected_rows_checked": len(counts) if cross_check is not None else 0,
            "mismatch_count": len(cross_check_mismatches),
            "mismatch_row_ids": cross_check_mismatches,
        },
        "gates": {
            "min_overall_safe_occurrence_rate": MIN_OVERALL_SAFE_OCCURRENCE_RATE,
            "min_cohort_safe_occurrence_rate": MIN_COHORT_SAFE_OCCURRENCE_RATE,
            "min_cohort_safe_occurrences": MIN_COHORT_SAFE_OCCURRENCES,
            "failures": gate_failures,
        },
        "parent_selection": parent_selection,
        "decision": "GO" if not gate_failures else "NO_GO",
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit outer-train grounded-target coverage.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int, choices=(0, 3))
    parser.add_argument("--seed", type=int, default=42, choices=(42,))
    parser.add_argument("--cross-check-jsonl", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.data.resolve())
    for column in ("name", "description"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    oof = np.load(args.oof.resolve(), allow_pickle=True)
    if not np.array_equal(frame["id"].astype(str).to_numpy(), oof["ids"].astype(str)):
        raise ValueError("OOF id mismatch")
    records, parent_selection = select_parent_training_records(
        frame,
        oof,
        seed=args.seed,
        holdout_fold=args.fold,
        full_train=False,
    )
    cross_check = _cross_check_rows(
        args.cross_check_jsonl.resolve() if args.cross_check_jsonl else None
    )
    audit = build_coverage_audit(
        frame,
        oof,
        records,
        holdout_fold=args.fold,
        seed=args.seed,
        parent_selection=parent_selection,
        cross_check=cross_check,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0 if audit["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())

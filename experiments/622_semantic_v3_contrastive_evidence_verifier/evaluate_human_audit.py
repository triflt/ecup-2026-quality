#!/usr/bin/env python3
"""Validate completed private exp622 ratings and emit aggregate-only gate output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

EXPERIMENT = Path(__file__).resolve().parent
SPEC = json.loads((EXPERIMENT / "frozen_spec.json").read_text(encoding="utf-8"))
SCHEMA = json.loads((EXPERIMENT / "human_audit_schema_v1.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_row(row: dict[str, str], index: int) -> None:
    required = set(SCHEMA["required"])
    if set(row) != required:
        raise ValueError(f"Row {index} columns differ from frozen schema")
    for field in ("strict_pass", "critical_unsupported", "scope_or_negation_failure"):
        if row[field] not in {"0", "1"}:
            raise ValueError(f"Row {index} has missing/invalid human rating: {field}")
    strict_pass = int(row["strict_pass"])
    critical = int(row["critical_unsupported"])
    scope = int(row["scope_or_negation_failure"])
    if strict_pass and (critical or scope):
        raise ValueError(f"Row {index} is both a strict pass and a failure")
    if not strict_pass and not row["review_notes"].strip():
        raise ValueError(f"Row {index} failure requires review_notes")
    expected_audit_id = f"A{index:03d}"
    if row["audit_id"] != expected_audit_id:
        raise ValueError(f"Row order/audit_id changed at row {index}")
    start, end = int(row["raw_start"]), int(row["raw_end"])
    if not 0 <= start < end <= len(row["source_text"]):
        raise ValueError(f"Row {index} has invalid source offsets")


def evaluate(review_path: Path, output_path: Path) -> dict[str, Any]:
    if ".local" not in review_path.resolve().parts:
        raise ValueError("Human review input must remain below .local")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")
    with review_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]
    sample_size = SPEC["audit"]["sample_size"]
    if len(rows) != sample_size:
        raise ValueError(f"Expected exactly {sample_size} review rows, got {len(rows)}")
    if len({row["row_id"] for row in rows}) != sample_size:
        raise ValueError("Review packet does not have 300 unique row IDs")
    if len({row["semantic_component"] for row in rows}) != sample_size:
        raise ValueError("Review packet does not have 300 unique product families")
    for index, row in enumerate(rows, 1):
        _validate_row(row, index)
    counts = Counter()
    for row in rows:
        counts["strict_pass"] += int(row["strict_pass"])
        counts["critical_unsupported"] += int(row["critical_unsupported"])
        counts["scope_or_negation_failure"] += int(row["scope_or_negation_failure"])
    passed = (
        counts["strict_pass"] >= SPEC["audit"]["minimum_human_strict_pass"]
        and counts["critical_unsupported"] <= SPEC["audit"]["maximum_critical_unsupported"]
        and counts["scope_or_negation_failure"]
        <= SPEC["audit"]["maximum_scope_or_negation_failure"]
    )
    summary = {
        "schema_version": "exp622_human_gate_summary_v1",
        "experiment_id": "622",
        "review_rows": sample_size,
        "strict_pass": counts["strict_pass"],
        "critical_unsupported": counts["critical_unsupported"],
        "scope_or_negation_failure": counts["scope_or_negation_failure"],
        "review_file_sha256": _sha256(review_path),
        "decision": "GO" if passed else "NO_GO",
        "gpu_gate_open": passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.review, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

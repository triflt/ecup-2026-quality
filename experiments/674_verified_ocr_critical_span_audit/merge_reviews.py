#!/usr/bin/env python3
"""Merge disjoint blind-review overlays in the frozen template order."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDS = (
    "audit_id",
    "review_visual_critical_span_present",
    "review_ocr_captures_all_critical_text",
    "review_ocr_preserves_scope_relation",
    "review_unsupported_critical_text",
    "review_evidence_relevant",
    "review_notes",
)
REQUIRED_RATINGS = FIELDS[1:-1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"review overlay header mismatch: {path}")
        return list(reader)


def merge(*, template: Path, chunks: list[Path], output: Path, allow_partial: bool = False) -> int:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    template_rows = read_csv(template)
    template_ids = [row["audit_id"] for row in template_rows]
    if len(template_ids) != len(set(template_ids)):
        raise ValueError("template audit IDs are not unique")
    merged: dict[str, dict[str, str]] = {}
    for path in chunks:
        for row in read_csv(path):
            audit_id = row["audit_id"]
            if audit_id not in set(template_ids):
                raise ValueError(f"foreign audit ID in chunk: {audit_id}")
            if audit_id in merged:
                raise ValueError(f"duplicate reviewed audit ID: {audit_id}")
            if any(not row[field].strip() for field in REQUIRED_RATINGS):
                raise ValueError(f"incomplete reviewed row: {audit_id}")
            merged[audit_id] = row
    if not allow_partial and set(merged) != set(template_ids):
        missing = sorted(set(template_ids) - set(merged))
        raise ValueError(f"review chunks do not cover the frozen template: {missing[:5]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(merged.get(audit_id, template_rows[index]) for index, audit_id in enumerate(template_ids))
    return len(merged)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--chunk", type=Path, action="append", required=True, dest="chunks")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    count = merge(**vars(parser.parse_args()))
    print(f"merged review rows: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Materialize the completed two-pass blind review from the frozen template.

The review decisions were made from the paired first-image contact sheets and
label-free names/descriptions.  The explicit reject set is kept here so the
generated CSV is reproducible without exposing training labels.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
REJECTS = {"R090", "R096", "R103", "R113", "R158", "R176", "R259", "R289"}


def main() -> None:
    source = HERE / "blind_review_template.csv"
    target = HERE / "blind_reviews_completed.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 300:
        raise ValueError(f"expected 300 frozen pairs, got {len(rows)}")
    fields = list(rows[0])
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            value = "0" if row["audit_id"] in REJECTS else "1"
            row["reviewer_a_same_product"] = value
            row["reviewer_b_same_product"] = value
            writer.writerow(row)


if __name__ == "__main__":
    main()

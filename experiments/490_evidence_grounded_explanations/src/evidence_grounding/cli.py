"""CSV command-line interface for deterministic evidence extraction."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from .extractor import extract_evidence, render_submission


def _prediction(value: str, *, row_number: int) -> int:
    stripped = value.strip()
    if stripped not in {"0", "1"}:
        raise ValueError(f"row {row_number}: frozen prediction must be 0 or 1")
    return int(stripped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Input CSV")
    parser.add_argument("--output", type=Path, required=True, help="Evidence sidecar JSONL")
    parser.add_argument("--submission-output", type=Path, help="Optional id/result CSV")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--category-column", default="category")
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--description-column", default="description")
    parser.add_argument("--prediction-column", default="frozen_prediction")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    required = {
        args.id_column,
        args.category_column,
        args.name_column,
        args.description_column,
        args.prediction_column,
    }
    results = []
    with args.data.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            results.append(
                extract_evidence(
                    row_id=row[args.id_column],
                    category=row[args.category_column],
                    name=row[args.name_column],
                    description=row[args.description_column],
                    frozen_prediction=_prediction(
                        row[args.prediction_column], row_number=row_number
                    ),
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        for result in results:
            output_file.write(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
            output_file.write("\n")

    if args.submission_output is not None:
        args.submission_output.parent.mkdir(parents=True, exist_ok=True)
        with args.submission_output.open("w", encoding="utf-8", newline="") as submission_file:
            writer = csv.DictWriter(submission_file, fieldnames=[args.id_column, "result"])
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {args.id_column: result.row_id, "result": render_submission(result)}
                )
    return 0

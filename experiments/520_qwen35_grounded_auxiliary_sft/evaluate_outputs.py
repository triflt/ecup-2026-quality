from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from structured_target import (
    CONCEPT_VOCABULARY,
    NO_SAFE_EVIDENCE,
    parse_first_atomic_verdict,
    parse_structured_target,
    render_explanation,
    validate_exact_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated exp520 structured outputs.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rendered-output", type=Path)
    parser.add_argument("--generated-column", default="generated_output")
    parser.add_argument("--label-column", default="label")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    details = []
    first_token_ok = structured_ok = exact_ok = concept_category_ok = verdict_correct = 0
    with args.input.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"id", "category", "name", "description", args.generated_column}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"input is missing columns: {sorted(missing)}")
        has_label = args.label_column in (reader.fieldnames or ())
        for row in reader:
            generated = row[args.generated_column]
            item = {
                "id": str(row["id"]),
                "first_token_valid": False,
                "structured_valid": False,
                "exact_evidence": False,
                "concept_category_valid": False,
                "verdict": None,
                "rendered_explanation": None,
                "error": None,
            }
            try:
                verdict = parse_first_atomic_verdict(generated)
                item["first_token_valid"] = True
                item["verdict"] = verdict
                first_token_ok += 1
                if has_label and verdict == int(row[args.label_column]):
                    verdict_correct += 1
                parsed = parse_structured_target(generated)
                item["structured_valid"] = True
                structured_ok += 1
                exact = validate_exact_evidence(
                    parsed, name=row["name"], description=row["description"]
                )
                item["exact_evidence"] = exact
                exact_ok += int(exact)
                concept_category = (
                    parsed.concept == NO_SAFE_EVIDENCE
                    or CONCEPT_VOCABULARY[parsed.concept]["category"] == row["category"]
                )
                item["concept_category_valid"] = concept_category
                concept_category_ok += int(concept_category)
                if exact and concept_category:
                    item["rendered_explanation"] = render_explanation(parsed)
            except (KeyError, TypeError, ValueError) as error:
                item["error"] = str(error)
            details.append(item)

    rows = len(details)
    report = {
        "experiment_id": "520",
        "rows": rows,
        "first_token_valid": first_token_ok,
        "first_token_valid_rate": first_token_ok / rows if rows else 0.0,
        "structured_valid": structured_ok,
        "structured_valid_rate": structured_ok / rows if rows else 0.0,
        "exact_evidence": exact_ok,
        "exact_evidence_rate": exact_ok / rows if rows else 0.0,
        "concept_category_valid": concept_category_ok,
        "concept_category_valid_rate": concept_category_ok / rows if rows else 0.0,
        "verdict_accuracy": verdict_correct / rows if rows and has_label else None,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.rendered_output is not None:
        args.rendered_output.parent.mkdir(parents=True, exist_ok=True)
        with args.rendered_output.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=["id", "verdict", "comment"])
            writer.writeheader()
            for item in details:
                writer.writerow(
                    {
                        "id": item["id"],
                        "verdict": item["verdict"],
                        "comment": item["rendered_explanation"] or "",
                    }
                )
    print(json.dumps({key: value for key, value in report.items() if key != "details"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

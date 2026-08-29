from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from prompt import validate_output


def load_data(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        url = os.environ.get("DATA_CSV_URL")
        if not url:
            raise ValueError("DATA_CSV_URL is required when --data is absent")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    with path.open(encoding="utf-8", newline="") as stream:
        return {str(row["id"]): row for row in csv.DictReader(stream)}


def load_records(root: Path) -> list[dict[str, Any]]:
    candidates = sorted(root.rglob("teacher_smoke20*.jsonl"))
    if len(candidates) != 1:
        raise ValueError(f"expected one teacher_smoke20*.jsonl, got {candidates}")
    with candidates[0].open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalized_comment(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip(" .")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("/work/input/data.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = load_data(args.data)
    records = load_records(args.artifact_root)
    audited: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    comments: list[str] = []
    elapsed_seconds: list[float] = []
    generation_attempts = 0
    normalization_actions: Counter[str] = Counter()
    for record in records:
        item_id = str(record["id"])
        row = data[item_id]
        parsed = record.get("parsed_output")
        validation = validate_output(
            parsed,
            category=str(row["category"]),
            expected_label=int(row["label"]),
            name=str(row["name"]),
            description=str(row["description"]),
            image_count=int(record["image_count"]),
        )
        reason = str(parsed.get("reason")) if isinstance(parsed, dict) else "INVALID"
        evidence = parsed.get("evidence") if isinstance(parsed, dict) else None
        explanation = parsed.get("explanation") if isinstance(parsed, dict) else None
        source = str(evidence.get("source")) if isinstance(evidence, dict) else "none"
        reasons[reason] += 1
        sources[source] += 1
        if isinstance(explanation, str):
            comments.append(normalized_comment(explanation))
        if isinstance(record.get("elapsed_seconds"), (int, float)):
            elapsed_seconds.append(float(record["elapsed_seconds"]))
        attempts = record.get("generation_attempts")
        if isinstance(attempts, list):
            generation_attempts += len(attempts)
        for action in record.get("normalization_actions") or []:
            normalization_actions[str(action).split(":", 1)[0]] += 1
        audited.append({
            "id": item_id,
            "category": str(row["category"]),
            "label": int(row["label"]),
            "name": str(row["name"])[:300],
            "description": str(row["description"])[:700],
            "image_count": int(record["image_count"]),
            "accepted_by_generator": bool(record.get("accepted")),
            "revalidation_errors": list(validation.errors),
            "output": parsed,
        })
    cell_counts = Counter((row["category"], row["label"]) for row in audited)
    report = {
        "schema_version": "teacher_smoke_audit_v1",
        "rows": len(audited),
        "revalidated": sum(not row["revalidation_errors"] for row in audited),
        "not_enough_evidence": reasons["not_enough_evidence"],
        "unique_comments": len(set(comments)),
        "duplicate_comments": len(comments) - len(set(comments)),
        "generation_attempts": generation_attempts,
        "mean_row_elapsed_seconds": statistics.fmean(elapsed_seconds) if elapsed_seconds else None,
        "median_row_elapsed_seconds": statistics.median(elapsed_seconds) if elapsed_seconds else None,
        "normalization_action_counts": dict(sorted(normalization_actions.items())),
        "cell_counts": {f"{key[0]}|{key[1]}": value for key, value in sorted(cell_counts.items())},
        "reason_counts": dict(sorted(reasons.items())),
        "evidence_source_counts": dict(sorted(sources.items())),
        "visual_evidence_rows_requiring_manual_image_audit": [
            row["id"] for row in audited
            if isinstance(row["output"], dict)
            and isinstance(row["output"].get("evidence"), dict)
            and str(row["output"]["evidence"].get("source", "")).startswith("image:")
        ],
        "records": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("AUDIT_SUMMARY=" + json.dumps({
        key: report[key] for key in (
            "rows", "revalidated", "not_enough_evidence", "unique_comments",
            "duplicate_comments", "reason_counts", "evidence_source_counts",
            "visual_evidence_rows_requiring_manual_image_audit", "generation_attempts",
            "mean_row_elapsed_seconds", "median_row_elapsed_seconds",
            "normalization_action_counts",
        )
    }, ensure_ascii=False, sort_keys=True), flush=True)
    for row in audited:
        if row["revalidation_errors"]:
            print("INVALID_RECORD=" + json.dumps({
                "id": row["id"], "category": row["category"], "label": row["label"],
                "name": row["name"], "output": row["output"],
                "revalidation_errors": row["revalidation_errors"],
            }, ensure_ascii=False, sort_keys=True), flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if len(audited) != 20 or report["revalidated"] != 20:
        raise SystemExit("smoke audit failed structural acceptance")


if __name__ == "__main__":
    main()

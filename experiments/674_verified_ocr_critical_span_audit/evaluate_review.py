#!/usr/bin/env python3
"""Evaluate a completed experiment-674 review overlay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "frozen_spec.json").read_text(encoding="utf-8"))
FIELDS = (
    "review_visual_critical_span_present",
    "review_ocr_captures_all_critical_text",
    "review_ocr_preserves_scope_relation",
    "review_unsupported_critical_text",
    "review_evidence_relevant",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def evaluate(*, packet: Path, reviews: Path, private_manifest: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = json.loads(private_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "exp674_private_manifest_v1":
        raise ValueError("private manifest schema mismatch")
    if sha256_file(packet) != manifest["blind_packet_sha256"]:
        raise ValueError("blind packet changed after freezing")
    packet_rows = read_csv(packet)
    review_rows = read_csv(reviews)
    packet_ids = {row["audit_id"] for row in packet_rows}
    reviews_by_id = {row["audit_id"]: row for row in review_rows}
    if (
        len(packet_rows) != SPEC["sample_rows"]
        or len(packet_ids) != SPEC["sample_rows"]
        or len(review_rows) != SPEC["sample_rows"]
        or set(reviews_by_id) != packet_ids
        or set(manifest["mapping"]) != packet_ids
    ):
        raise ValueError("packet, review and private scopes differ")
    completed = 0
    for audit_id, row in reviews_by_id.items():
        values = {field: row.get(field, "").strip().lower() for field in FIELDS}
        if all(not value for value in values.values()):
            continue
        if any(not value for value in values.values()):
            raise ValueError(f"partially completed review row: {audit_id}")
        if values["review_visual_critical_span_present"] not in {"yes", "no", "unclear"}:
            raise ValueError(f"invalid visual critical-span rating: {audit_id}")
        if values["review_evidence_relevant"] not in {"yes", "no", "unclear"}:
            raise ValueError(f"invalid relevance rating: {audit_id}")
        if values["review_unsupported_critical_text"] not in {"yes", "no"}:
            raise ValueError(f"invalid unsupported rating: {audit_id}")
        visual = values["review_visual_critical_span_present"]
        expected = {"yes", "no"} if visual == "yes" else {"na"}
        if values["review_ocr_captures_all_critical_text"] not in expected:
            raise ValueError(f"capture rating conflicts with visual rating: {audit_id}")
        if values["review_ocr_preserves_scope_relation"] not in expected:
            raise ValueError(f"scope rating conflicts with visual rating: {audit_id}")
        completed += 1
    result: dict[str, Any] = {
        "schema_version": "exp674_review_score_v1",
        "experiment_id": "674",
        "review_rows": len(review_rows),
        "completed_rows": completed,
        "labels_exposed_to_reviewer": False,
        "baseline_error_state_exposed_to_reviewer": False,
        "sealed_rows_read": 0,
        "public_used": False,
        "gpu_hours": 0.0,
    }
    if completed != SPEC["sample_rows"]:
        result.update({"decision": "WAIT_FOR_COMPLETE_120_ROW_REVIEW", "metrics": {}, "gates": {}})
    else:
        values = [
            {field: row[field].strip().lower() for field in FIELDS}
            for row in reviews_by_id.values()
        ]
        critical = [row for row in values if row["review_visual_critical_span_present"] == "yes"]
        captured = sum(row["review_ocr_captures_all_critical_text"] == "yes" for row in critical)
        scope = sum(row["review_ocr_preserves_scope_relation"] == "yes" for row in critical)
        unsupported = sum(row["review_unsupported_critical_text"] == "yes" for row in values)
        recall = captured / len(critical) if critical else 0.0
        scope_rate = scope / len(critical) if critical else 0.0
        unsupported_rate = unsupported / len(values)
        metrics = {
            "visual_critical_rows": len(critical),
            "critical_span_captured": captured,
            "critical_span_recall": recall,
            "scope_relation_preserved": scope,
            "scope_preservation_rate": scope_rate,
            "unsupported_critical_rows": unsupported,
            "unsupported_rate": unsupported_rate,
            "relevant_rows": sum(row["review_evidence_relevant"] == "yes" for row in values),
            "unclear_visual_rows": sum(
                row["review_visual_critical_span_present"] == "unclear" for row in values
            ),
        }
        gates = {
            "at_least_50_visual_critical_rows": len(critical) >= SPEC["minimum_visual_critical_rows"],
            "critical_span_recall_at_least_0_90": recall >= SPEC["minimum_critical_span_recall"],
            "scope_preservation_at_least_0_95": scope_rate >= SPEC["minimum_scope_preservation"],
            "unsupported_rate_at_most_0_01": unsupported_rate <= SPEC["maximum_unsupported_rate"],
        }
        result.update(
            {
                "decision": (
                    SPEC["passing_action"]
                    if all(gates.values())
                    else "NO_GO_REJECT_VERIFIED_OCR_FEATURE_SCREEN"
                ),
                "metrics": metrics,
                "gates": gates,
                "full_training_authorized": False,
                "public_submission_authorized": False,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = evaluate(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

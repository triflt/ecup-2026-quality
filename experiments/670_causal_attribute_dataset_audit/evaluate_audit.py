"""Score the frozen 300-row causal target review without inventing ratings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

YES_NO = {"yes", "no"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_reviews(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "audit_id",
        "review_sold_object_correct",
        "review_regulated_substance_correct",
        "review_relation_correct",
        "review_evidence_correct",
        "review_unsupported_claim",
        "review_notes",
    }
    if not rows or set(rows[0]) != required:
        raise ValueError("review overlay schema mismatch")
    result = {}
    for row in rows:
        audit_id = row["audit_id"]
        if audit_id in result:
            raise ValueError("duplicate review overlay ID")
        result[audit_id] = row
    return result


def evaluate(
    *,
    audit: Path,
    private_manifest: Path,
    output: Path,
    reviews: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = json.loads(private_manifest.read_text(encoding="utf-8"))
    with audit.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 300 or len({row["semantic_component"] for row in rows}) != 300:
        raise ValueError("audit membership must remain 300 unique components")
    expected = {
        item["audit_id"]: (item["row_id"], item["semantic_component"])
        for item in manifest["records"]
    }
    observed = {
        row["audit_id"]: (row["row_id"], row["semantic_component"])
        for row in rows
    }
    if expected != observed:
        raise ValueError("audit membership/order keys changed")
    if manifest.get("packet_sha256") not in (None, sha256_file(audit)):
        raise ValueError("frozen audit packet checksum changed")
    review_overlay = _load_reviews(reviews)
    if not set(review_overlay).issubset(observed):
        raise ValueError("review overlay contains an unknown audit ID")

    review_fields = (
        "review_sold_object_correct",
        "review_regulated_substance_correct",
        "review_relation_correct",
        "review_evidence_correct",
        "review_unsupported_claim",
    )
    normalized = []
    invalid = []
    for row in rows:
        review = review_overlay.get(row["audit_id"], row)
        values = {field: review[field].strip().lower() for field in review_fields}
        bad = [field for field, value in values.items() if value not in YES_NO]
        if bad:
            invalid.append({"audit_id": row["audit_id"], "fields": bad})
        normalized.append((row, values))

    completed = [(row, values) for row, values in normalized if all(v in YES_NO for v in values.values())]
    partial_unsupported = sum(
        values["review_unsupported_claim"] == "yes" for _, values in completed
    )
    partial_strict_failures = sum(
        not (
            values["review_sold_object_correct"] == "yes"
            and values["review_regulated_substance_correct"] == "yes"
            and values["review_relation_correct"] == "yes"
            and values["review_evidence_correct"] == "yes"
            and values["review_unsupported_claim"] == "no"
        )
        for _, values in completed
    )
    partial_priority_failures = sum(
        row["focus"] == "transaction_scope_priority"
        and not (
            values["review_sold_object_correct"] == "yes"
            and values["review_relation_correct"] == "yes"
        )
        for row, values in completed
    )
    irreversible = {
        "unsupported_claim_limit_exceeded": partial_unsupported > 3,
        "overall_282_of_300_no_longer_reachable": partial_strict_failures > 18,
        "priority_95_of_100_no_longer_reachable": partial_priority_failures > 5,
    }

    if any(irreversible.values()):
        report = {
            "schema_version": "exp670_causal_audit_score_v1",
            "experiment_id": "670",
            "status": "completed_early_rejection",
            "decision": "NO_GO_REJECT_CAUSAL_TARGETS",
            "rows": 300,
            "reviewed_rows": len(completed),
            "strict_failures_observed": partial_strict_failures,
            "priority_failures_observed": partial_priority_failures,
            "unsupported_claims_observed": partial_unsupported,
            "irreversible_gates": irreversible,
            "incomplete_rows_not_needed_for_decision": 300 - len(completed),
            "sealed_rows_loaded": 0,
            "public_used": False,
        }
    elif invalid:
        report = {
            "schema_version": "exp670_causal_audit_score_v1",
            "experiment_id": "670",
            "status": "blocked_review_required",
            "decision": "WAIT_FOR_COMPLETE_300_ROW_REVIEW",
            "rows": 300,
            "reviewed_rows": len(completed),
            "incomplete_or_invalid_rows": len(invalid),
            "invalid_preview": invalid[:20],
            "sealed_rows_loaded": 0,
            "public_used": False,
        }
    else:
        row_passes = []
        unsupported = 0
        scope_correct = 0
        scope_total = 0
        per_focus: Counter[str] = Counter()
        per_focus_pass: Counter[str] = Counter()
        for row, values in normalized:
            row_pass = (
                values["review_sold_object_correct"] == "yes"
                and values["review_regulated_substance_correct"] == "yes"
                and values["review_relation_correct"] == "yes"
                and values["review_evidence_correct"] == "yes"
                and values["review_unsupported_claim"] == "no"
            )
            row_passes.append(row_pass)
            unsupported += values["review_unsupported_claim"] == "yes"
            per_focus[row["focus"]] += 1
            per_focus_pass[row["focus"]] += row_pass
            if row["focus"] == "transaction_scope_priority":
                scope_total += 1
                scope_correct += (
                    values["review_sold_object_correct"] == "yes"
                    and values["review_relation_correct"] == "yes"
                )
        gates = {
            "overall_at_least_282_of_300": sum(row_passes) >= 282,
            "scope_object_relation_at_least_95_of_100": scope_total == 100
            and scope_correct >= 95,
            "unsupported_claims_at_most_3": unsupported <= 3,
        }
        report = {
            "schema_version": "exp670_causal_audit_score_v1",
            "experiment_id": "670",
            "status": "completed",
            "decision": "GO_BUILD_671" if all(gates.values()) else "NO_GO_REJECT_CAUSAL_TARGETS",
            "rows": 300,
            "strict_passes": sum(row_passes),
            "scope_object_relation_correct": scope_correct,
            "scope_object_relation_total": scope_total,
            "unsupported_claims": unsupported,
            "per_focus": {
                key: {"rows": per_focus[key], "strict_passes": per_focus_pass[key]}
                for key in sorted(per_focus)
            },
            "gates": gates,
            "sealed_rows_loaded": 0,
            "public_used": False,
        }
    report["audit_sha256"] = sha256_file(audit)
    report["private_manifest_sha256"] = sha256_file(private_manifest)
    report["review_overlay_sha256"] = None if reviews is None else sha256_file(reviews)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

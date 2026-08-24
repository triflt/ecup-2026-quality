#!/usr/bin/env python3
"""Score the frozen paired blind review without classification labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "frozen_spec.json").read_text(encoding="utf-8"))
RATING_FIELDS = (
    "review_verdict_consistent",
    "review_evidence_relevant",
    "review_object_relation_correct",
    "review_unsupported_fact",
    "review_visual_decisive",
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


def _public_result(payload: dict[str, Any]) -> dict[str, Any]:
    private_keys = {"audit_id", "row_id", "semantic_component", "mapping"}
    return {key: value for key, value in payload.items() if key not in private_keys}


def evaluate(
    *, candidates: Path, reviews: Path, private_manifest: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = json.loads(private_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "exp672_blind_manifest_v1":
        raise ValueError("private manifest schema mismatch")
    if sha256_file(candidates) != manifest["blind_candidates_sha256"]:
        raise ValueError("blind candidate packet changed after freezing")
    candidate_rows = read_csv(candidates)
    review_rows = read_csv(reviews)
    candidates_by_id = {row["audit_id"]: row for row in candidate_rows}
    reviews_by_id = {row["audit_id"]: row for row in review_rows}
    mapping = manifest["mapping"]
    expected_ids = set(mapping)
    if (
        len(candidate_rows) != 80
        or len(candidates_by_id) != 80
        or len(review_rows) != 80
        or len(reviews_by_id) != 80
        or set(candidates_by_id) != expected_ids
        or set(reviews_by_id) != expected_ids
    ):
        raise ValueError("review, candidate and unblinding scopes differ")
    invalid_values: dict[str, list[str]] = {}
    incomplete: list[str] = []
    for audit_id, review in reviews_by_id.items():
        for field in RATING_FIELDS:
            value = review.get(field, "").strip().lower()
            if not value:
                incomplete.append(audit_id)
            elif value not in {"yes", "no"}:
                invalid_values.setdefault(audit_id, []).append(field)
    if invalid_values:
        raise ValueError(f"invalid yes/no ratings: {invalid_values}")

    result: dict[str, Any] = {
        "schema_version": "exp672_review_score_v1",
        "experiment_id": "672",
        "review_rows": len(review_rows),
        "completed_rows": len(review_rows) - len(set(incomplete)),
        "labels_read": 0,
        "sealed_rows_read": 0,
        "public_used": False,
        "passing_action": SPEC["passing_action"],
    }
    automatic_format = {
        alias: sum(
            candidates_by_id[key]["automatic_contract_valid"].lower() == "true"
            for key, value in mapping.items()
            if value["candidate_alias"] == alias
        )
        for alias in SPEC["candidate_models"]
    }
    if automatic_format["qwen36_27b"] < SPEC["required_candidate_format_count"]:
        result.update(
            {
                "decision": "NO_GO_REJECT_27B_EXPLANATION_SCALE_UP",
                "scores": {
                    alias: {
                        "rows": SPEC["rows"],
                        "automatic_contract_valid": automatic_format[alias],
                    }
                    for alias in SPEC["candidate_models"]
                },
                "gates": {"qwen36_format_and_verdict_40_of_40": False},
                "human_review_skipped": True,
                "human_review_skipped_reason": (
                    "The preregistered automatic 40/40 gate is already impossible; "
                    "human ratings cannot reverse it."
                ),
                "classification_training_authorized": False,
                "full_teacher_extraction_authorized": False,
            }
        )
    elif incomplete:
        result.update(
            {
                "decision": "WAIT_FOR_COMPLETE_80_CANDIDATE_REVIEW",
                "scores": {},
                "gates": {},
            }
        )
    else:
        scores: dict[str, dict[str, Any]] = {}
        for alias in SPEC["candidate_models"]:
            audit_ids = [key for key, value in mapping.items() if value["candidate_alias"] == alias]
            if len(audit_ids) != SPEC["rows"]:
                raise ValueError(f"unblinding map has wrong count for {alias}")
            counts = Counter()
            for key in audit_ids:
                review = reviews_by_id[key]
                for field in RATING_FIELDS:
                    counts[field] += review[field].strip().lower() == "yes"
            scores[alias] = {
                "rows": len(audit_ids),
                "automatic_contract_valid": automatic_format[alias],
                "verdict_consistent": counts["review_verdict_consistent"],
                "evidence_relevant": counts["review_evidence_relevant"],
                "object_relation_correct": counts["review_object_relation_correct"],
                "unsupported_fact": counts["review_unsupported_fact"],
                "visual_decisive": counts["review_visual_decisive"],
            }
        q4 = scores["qwen35_4b"]
        q27 = scores["qwen36_27b"]
        relevance_delta = q27["evidence_relevant"] - q4["evidence_relevant"]
        visual_delta = q27["visual_decisive"] - q4["visual_decisive"]
        gates = {
            "qwen36_format_and_verdict_40_of_40": (
                q27["automatic_contract_valid"] == SPEC["required_candidate_format_count"]
                and q27["verdict_consistent"] == SPEC["required_candidate_format_count"]
            ),
            "qwen36_zero_unsupported": (
                q27["unsupported_fact"] <= SPEC["maximum_candidate_unsupported_count"]
            ),
            "material_relevance_or_visual_gain": (
                relevance_delta >= SPEC["minimum_relevance_count_delta"]
                or visual_delta >= SPEC["minimum_visual_count_delta"]
            ),
        }
        result.update(
            {
                "decision": (
                    SPEC["passing_action"]
                    if all(gates.values())
                    else "NO_GO_REJECT_27B_EXPLANATION_SCALE_UP"
                ),
                "scores": scores,
                "deltas_qwen36_minus_qwen35": {
                    "evidence_relevant_count": relevance_delta,
                    "visual_decisive_count": visual_delta,
                    "object_relation_correct_count": (
                        q27["object_relation_correct"] - q4["object_relation_correct"]
                    ),
                },
                "gates": gates,
                "classification_training_authorized": False,
                "full_teacher_extraction_authorized": False,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_public_result(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = evaluate(**vars(parser.parse_args()))
    print(json.dumps(_public_result(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

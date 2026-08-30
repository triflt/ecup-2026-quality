#!/usr/bin/env python3
"""Fail closed on a completed copy of the frozen experiment-626 packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

EXPERIMENT = Path(__file__).resolve().parent
SPEC = json.loads((EXPERIMENT / "frozen_audit_spec.json").read_text(encoding="utf-8"))
HUMAN_COLUMNS = (
    "evidence_relevant", "object_of_sale_correct", "negation_correct",
    "composition_or_completeness_correct", "verdict_consistent", "unsupported_fact",
    "strict_pass", "critical_unsupported", "scope_or_negation_failure", "review_notes",
)
IMMUTABLE_COLUMNS = (
    "schema_version", "audit_id", "row_id", "semantic_component", "development_fold",
    "category", "source_card", "char_start", "char_end", "exact_span", "concept",
    "verdict", "explanation",
)
COLUMNS = ("schema_version", "sample_sha256") + IMMUTABLE_COLUMNS[1:] + HUMAN_COLUMNS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_hash(rows: list[dict[str, str]]) -> str:
    payload = [[row[column] for column in IMMUTABLE_COLUMNS] for row in rows]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_packet(path: Path) -> list[dict[str, str]]:
    if ".local" not in path.resolve().parts:
        raise ValueError("Human audit files must remain below .local")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("Audit columns/order differ from frozen schema")
        return [dict(row) for row in reader]


def evaluate(*, frozen_packet: Path, completed_copy: Path, frozen_manifest: Path, output: Path) -> dict[str, Any]:
    if frozen_packet.resolve() == completed_copy.resolve():
        raise ValueError("Ratings must be supplied in a separate completed copy")
    if ".local" not in frozen_manifest.resolve().parts:
        raise ValueError("Frozen manifest must remain below .local")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    frozen = read_packet(frozen_packet)
    completed = read_packet(completed_copy)
    manifest = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    if len(frozen) != SPEC["sample_size"] or len(completed) != SPEC["sample_size"]:
        raise ValueError("Frozen and completed packets must each have exactly 300 rows")
    if any(row[field] for row in frozen for field in HUMAN_COLUMNS):
        raise ValueError("Frozen packet already contains ratings")
    if manifest.get("human_ratings_present") is not False or manifest.get("sealed_rows_read") != 0:
        raise ValueError("Frozen manifest violates blind development-only contract")
    expected_hash = manifest.get("sample_sha256")
    if not isinstance(expected_hash, str) or sample_hash(frozen) != expected_hash:
        raise ValueError("Frozen packet does not match frozen sample hash")
    if manifest.get("packet_sha256") != sha256_file(frozen_packet):
        raise ValueError("Frozen packet file hash differs from manifest")
    if [row["row_id"] for row in frozen] != manifest.get("ordered_row_ids"):
        raise ValueError("Frozen row order differs from manifest")
    for index, (source, rated) in enumerate(zip(frozen, completed, strict=True), 1):
        if any(source[field] != rated[field] for field in IMMUTABLE_COLUMNS):
            raise ValueError(f"Completed copy changed immutable content/order at row {index}")
        if rated["sample_sha256"] != expected_hash:
            raise ValueError(f"Completed copy sample hash mismatch at row {index}")
        if rated["audit_id"] != f"E626-{index:03d}":
            raise ValueError(f"Audit ID/order mismatch at row {index}")
        binary = (
            "evidence_relevant", "verdict_consistent", "unsupported_fact", "strict_pass",
            "critical_unsupported", "scope_or_negation_failure",
        )
        if any(rated[field] not in {"0", "1"} for field in binary):
            raise ValueError(f"Missing or invalid required human rating at row {index}")
        scoped = ("object_of_sale_correct", "negation_correct", "composition_or_completeness_correct")
        if any(rated[field] not in {"0", "1", "NA"} for field in scoped):
            raise ValueError(f"Missing or invalid scoped human rating at row {index}")
        critical = int(rated["critical_unsupported"])
        unsupported = int(rated["unsupported_fact"])
        scope_failure = int(rated["scope_or_negation_failure"])
        strict = int(rated["strict_pass"])
        if critical != unsupported:
            raise ValueError(f"Critical unsupported and unsupported_fact disagree at row {index}")
        dimension_failure = (
            rated["evidence_relevant"] == "0"
            or rated["verdict_consistent"] == "0"
            or any(rated[field] == "0" for field in scoped)
            or critical == 1
            or scope_failure == 1
        )
        if strict == int(dimension_failure):
            raise ValueError(f"strict_pass contradicts detailed ratings at row {index}")
        if not strict and not rated["review_notes"].strip():
            raise ValueError(f"Failed row requires review_notes at row {index}")

    counts = Counter()
    for row in completed:
        for field in ("strict_pass", "critical_unsupported", "scope_or_negation_failure"):
            counts[field] += int(row[field])
    passed = (
        counts["strict_pass"] >= SPEC["minimum_strict_pass"]
        and counts["critical_unsupported"] <= SPEC["maximum_critical_unsupported"]
        and counts["scope_or_negation_failure"] <= SPEC["maximum_scope_or_negation_failure"]
    )
    result = {
        "schema_version": "exp626_human_audit_result_v1",
        "experiment_id": "626",
        "status": "accepted" if passed else "rejected_by_gate",
        "decision": "GO" if passed else "NO_GO",
        "review_rows": len(completed),
        "strict_pass": counts["strict_pass"],
        "critical_unsupported": counts["critical_unsupported"],
        "scope_or_negation_failure": counts["scope_or_negation_failure"],
        "sample_sha256": expected_hash,
        "completed_copy_sha256": sha256_file(completed_copy),
        "human_review_required": True,
        "sealed_rows_read": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-packet", type=Path, required=True)
    parser.add_argument("--completed-copy", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(evaluate(
        frozen_packet=args.frozen_packet, completed_copy=args.completed_copy,
        frozen_manifest=args.frozen_manifest, output=args.output,
    ), ensure_ascii=False, indent=2))

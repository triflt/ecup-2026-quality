#!/usr/bin/env python3
"""Freeze a blind development-only packet for experiment 626."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

EXPERIMENT = Path(__file__).resolve().parent
SPEC = json.loads((EXPERIMENT / "frozen_audit_spec.json").read_text(encoding="utf-8"))
FEATURE_COLUMNS = {"id", "category", "name", "description"}
MEMBERSHIP_COLUMNS = {
    "id", "category", "semantic_component", "component_size", "split", "development_fold"
}
PREDICTION_COLUMNS = {
    "id", "category", "fold", "verdict", "evidence", "concept", "explanation",
    "char_start", "char_end",
}
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


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def index_unique(rows: Iterable[dict[str, str]], kind: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id or row_id in result:
            raise ValueError(f"{kind} has empty or duplicate id: {row_id!r}")
        result[row_id] = row
    return result


def require_private_new(path: Path) -> None:
    if ".local" not in path.resolve().parts:
        raise ValueError("Audit packets and manifests must remain below .local")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")


def discover_ids(paths: Iterable[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    ids: set[str] = set()
    sources: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("id", "row_id", "product_id"):
                candidate = value.get(key)
                if candidate is not None and str(candidate).strip():
                    ids.add(str(candidate).strip())
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        before = len(ids)
        if path.suffix.lower() == ".csv":
            _, rows = read_csv(path)
            visit(rows)
        elif path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        visit(json.loads(line))
        else:
            visit(json.loads(path.read_text(encoding="utf-8")))
        sources.append({"path": str(path), "sha256": sha256_file(path), "new_ids": len(ids) - before})
    return ids, sources


def sample_hash(rows: list[dict[str, str]]) -> str:
    payload = [[row[column] for column in IMMUTABLE_COLUMNS] for row in rows]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_text(row: dict[str, str]) -> str:
    return f"Название: {row['name']}\nОписание: {row['description']}"


def build(
    *, features_path: Path, membership_path: Path, prediction_paths: list[Path],
    exp490_manifest: Path, exp622_manifest: Path, prior_manifests: list[Path],
    packet_path: Path, manifest_path: Path,
) -> dict[str, Any]:
    require_private_new(packet_path)
    require_private_new(manifest_path)
    feature_columns, feature_rows = read_csv(features_path)
    member_columns, member_rows = read_csv(membership_path)
    if set(feature_columns) != FEATURE_COLUMNS or set(member_columns) != MEMBERSHIP_COLUMNS:
        raise ValueError("Original feature/membership schema mismatch")
    features = index_unique(feature_rows, "features")
    membership = index_unique(member_rows, "membership")
    if set(features) != set(membership):
        raise ValueError("Feature and membership ID sets differ")
    for row_id, member in membership.items():
        if member["split"] != "development":
            raise ValueError(f"Sealed/non-development membership is forbidden: {row_id}")
        if member["development_fold"] not in {"0", "1", "2", "3", "4"}:
            raise ValueError(f"Invalid development fold: {row_id}")
        if member["category"] != features[row_id]["category"]:
            raise ValueError(f"Category mismatch: {row_id}")

    predictions: dict[str, dict[str, str]] = {}
    for path in prediction_paths:
        columns, rows = read_csv(path)
        if set(columns) != PREDICTION_COLUMNS:
            raise ValueError(f"Prediction schema mismatch: {path}")
        for row in rows:
            row_id = row["id"].strip()
            if row_id in predictions:
                raise ValueError(f"Duplicate prediction id: {row_id}")
            predictions[row_id] = row
    if set(predictions) != set(features):
        raise ValueError("Predictions must cover the exact development ID set")

    exp490_ids, exp490_sources = discover_ids([exp490_manifest])
    exp622_ids, exp622_sources = discover_ids([exp622_manifest])
    if len(exp490_ids) != SPEC["required_exp490_exclusions"]:
        raise ValueError("Experiment 490 exclusion manifest must contain exactly 200 unique IDs")
    if len(exp622_ids) != SPEC["required_exp622_exclusions"]:
        raise ValueError("Experiment 622 exclusion manifest must contain exactly 300 unique IDs")
    prior_ids, prior_sources = discover_ids(prior_manifests)
    excluded = exp490_ids | exp622_ids | prior_ids

    candidates: list[dict[str, str]] = []
    for row_id, prediction in predictions.items():
        feature, member = features[row_id], membership[row_id]
        if row_id in excluded:
            continue
        if prediction["category"] != feature["category"]:
            raise ValueError(f"Prediction category mismatch: {row_id}")
        if prediction["fold"] != member["development_fold"]:
            raise ValueError(f"Prediction fold mismatch: {row_id}")
        if prediction["verdict"] not in {"0", "1"}:
            raise ValueError(f"Invalid verdict: {row_id}")
        if prediction["concept"] not in SPEC["allowed_concepts"]:
            raise ValueError(f"Concept outside frozen ontology: {row_id}")
        card = canonical_text(feature)
        try:
            start, end = int(prediction["char_start"]), int(prediction["char_end"])
        except ValueError as error:
            raise ValueError(f"Invalid offsets: {row_id}") from error
        evidence = prediction["evidence"]
        if evidence == "NO_EVIDENCE":
            if (start, end) != (-1, -1) or prediction["concept"] != "NO_EVIDENCE" or prediction["explanation"] != "NO_EVIDENCE":
                raise ValueError(f"Malformed NO_EVIDENCE payload: {row_id}")
        elif not (0 <= start < end <= len(card)) or card[start:end] != evidence or evidence not in prediction["explanation"]:
            raise ValueError(f"Evidence is not the declared exact card substring: {row_id}")
        rank = stable_hash(SPEC["namespace"], member["semantic_component"], row_id)
        candidates.append({
            "_rank": rank,
            "schema_version": "exp626_human_audit_row_v1",
            "audit_id": "",
            "row_id": row_id,
            "semantic_component": member["semantic_component"],
            "development_fold": member["development_fold"],
            "category": feature["category"],
            "source_card": card,
            "char_start": str(start),
            "char_end": str(end),
            "exact_span": evidence,
            "concept": prediction["concept"],
            "verdict": prediction["verdict"],
            "explanation": prediction["explanation"],
        })
    by_component: dict[str, dict[str, str]] = {}
    for row in candidates:
        previous = by_component.get(row["semantic_component"])
        if previous is None or row["_rank"] < previous["_rank"]:
            by_component[row["semantic_component"]] = row
    selected = sorted(by_component.values(), key=lambda row: row["_rank"])[: SPEC["sample_size"]]
    if len(selected) != SPEC["sample_size"]:
        raise ValueError(f"Only {len(selected)} fresh unique semantic components; need 300")
    for index, row in enumerate(selected, 1):
        row.pop("_rank")
        row["audit_id"] = f"E626-{index:03d}"
        row.update({field: "" for field in HUMAN_COLUMNS})
    frozen_hash = sample_hash(selected)
    for row in selected:
        row["sample_sha256"] = frozen_hash

    packet_path.parent.mkdir(parents=True, exist_ok=True)
    with packet_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "schema_version": "exp626_frozen_sample_manifest_v1",
        "experiment_id": "626",
        "sample_sha256": frozen_hash,
        "packet_sha256": sha256_file(packet_path),
        "sample_rows": len(selected),
        "unique_row_ids": len({row["row_id"] for row in selected}),
        "unique_semantic_components": len({row["semantic_component"] for row in selected}),
        "human_ratings_present": False,
        "sealed_rows_read": 0,
        "excluded_id_count": len(excluded),
        "overlap_with_excluded_ids": len({row["row_id"] for row in selected} & excluded),
        "input_sha256": {
            "features": sha256_file(features_path),
            "membership": sha256_file(membership_path),
            "predictions": {str(path): sha256_file(path) for path in prediction_paths},
        },
        "exclusion_sources": exp490_sources + exp622_sources + prior_sources,
        "ordered_row_ids": [row["row_id"] for row in selected],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--exp490-id-manifest", type=Path, required=True)
    parser.add_argument("--exp622-id-manifest", type=Path, required=True)
    parser.add_argument("--prior-id-manifest", type=Path, action="append", default=[])
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build(
        features_path=args.features, membership_path=args.membership,
        prediction_paths=args.predictions, exp490_manifest=args.exp490_id_manifest,
        exp622_manifest=args.exp622_id_manifest, prior_manifests=args.prior_id_manifest,
        packet_path=args.packet, manifest_path=args.manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

#!/usr/bin/env python3
"""Build a frozen row-blind visual audit of verified OCR evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "frozen_spec.json").read_text(encoding="utf-8"))
CATEGORIES = ("БАД", "Легковоспламеняющиеся")
IMMUTABLE_FIELDS = (
    "audit_id",
    "category",
    "name",
    "description",
    "available_image_count",
    "total_image_count",
    "image_evidence_json",
)
REVIEW_FIELDS = (
    "audit_id",
    "review_visual_critical_span_present",
    "review_ocr_captures_all_critical_text",
    "review_ocr_preserves_scope_relation",
    "review_unsupported_critical_text",
    "review_evidence_relevant",
    "review_notes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(*parts: object) -> str:
    value = "|".join((SPEC["namespace"], *(str(part) for part in parts)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def prior_ids(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        rows = read_csv(path)
    elif path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("sanitized_records", payload.get("records", []))
        else:
            rows = []
    values = {
        str(row.get("row_id", row.get("id")))
        for row in rows
        if isinstance(row, dict) and row.get("row_id", row.get("id")) is not None
    }
    if not values:
        raise ValueError(f"prior audit lacks row IDs: {path}")
    return values


def select_quota(
    rows: list[dict[str, Any]],
    *,
    category: str,
    error: bool,
    fold: int,
    quota: int,
    used_components: set[str],
) -> list[dict[str, Any]]:
    local = [
        row
        for row in rows
        if row["category"] == category
        and row["baseline_error"] is error
        and row["fold"] == fold
        and row["component"] not in used_components
    ]
    by_component: dict[str, dict[str, Any]] = {}
    for row in local:
        current = by_component.get(row["component"])
        if current is None or stable_key(row["id"]) < stable_key(current["id"]):
            by_component[row["component"]] = row
    local = list(by_component.values())
    buckets: dict[tuple[int, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in local:
        buckets[(row["label"], row["semantic_singleton"])].append(row)
    for key, values in buckets.items():
        values.sort(key=lambda row: stable_key(category, error, fold, key, row["component"], row["id"]))
    keys = sorted(buckets, key=lambda key: stable_key(category, error, fold, *key))
    selected: list[dict[str, Any]] = []
    while keys and len(selected) < quota:
        remaining = []
        for key in keys:
            if buckets[key] and len(selected) < quota:
                selected.append(buckets[key].pop(0))
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    if len(selected) != quota:
        raise ValueError(f"insufficient rows for frozen stratum {category}/{error}/{fold}")
    return selected


def build(
    *,
    data: Path,
    folds: Path,
    baseline_predictions: Path,
    image_manifest: Path,
    ocr_availability: Path,
    ocr_available_images: Path,
    aggregate_integrity: Path,
    prior_490: Path,
    prior_reviewed_670: Path,
    prior_672: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    paths = {
        "data": data,
        "folds": folds,
        "baseline_predictions": baseline_predictions,
        "image_manifest": image_manifest,
        "ocr_availability": ocr_availability,
        "ocr_available_images": ocr_available_images,
        "aggregate_integrity": aggregate_integrity,
        "prior_490": prior_490,
        "prior_reviewed_670": prior_reviewed_670,
        "prior_672": prior_672,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != SPEC["expected_sha256"]:
        raise ValueError("input SHA set differs from frozen spec")
    aggregate = json.loads(aggregate_integrity.read_text(encoding="utf-8"))
    if aggregate.get("decision") != "ACCEPT_PARTIAL_FAIL_CLOSED_OCR_DATASET":
        raise ValueError("experiment-660 OCR dataset is not accepted")

    excluded_ids: set[str] = set()
    for path in (prior_490, prior_reviewed_670, prior_672):
        excluded_ids.update(prior_ids(path))
    membership = read_csv(folds)
    components_by_id = {row["id"]: row["semantic_component"] for row in membership}
    excluded_components = {components_by_id[row_id] for row_id in excluded_ids if row_id in components_by_id}
    development = [row for row in membership if row["split"] == "development"]
    component_counts = Counter(row["semantic_component"] for row in development)
    data_rows = {row["id"]: row for row in read_csv(data)}
    predictions = {
        row["id"]: int(row["prediction"])
        for row in read_csv(baseline_predictions)
        if row["source"] == "exact_full140_semantic_v3"
    }

    images_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in image_manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        images_by_id[str(row["id"])].append(
            {"image_index": int(row["image_index"]), "url": str(row["url"])}
        )
    status_by_key = {}
    for line in ocr_availability.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        status_by_key[(str(row["id"]), int(row["image_index"]))] = str(row["status"])
    payload_by_key = {}
    for line in ocr_available_images.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (str(row["id"]), int(row["image_index"]))
        payload_by_key[key] = [
            {
                "text": region["text"],
                "quadrilateral_normalized": region["quadrilateral_normalized"],
            }
            for region in row["regions"]
        ]
    if set(payload_by_key) != {key for key, status in status_by_key.items() if status == "OCR_AVAILABLE"}:
        raise ValueError("available OCR payload keys differ from availability contract")

    candidates: list[dict[str, Any]] = []
    for member in development:
        row_id = member["id"]
        component = member["semantic_component"]
        if component in excluded_components or row_id not in data_rows or row_id not in predictions:
            continue
        images = sorted(images_by_id[row_id], key=lambda row: row["image_index"])
        if not any(status_by_key[(row_id, row["image_index"])] == "OCR_AVAILABLE" for row in images):
            continue
        candidates.append(
            {
                "id": row_id,
                "component": component,
                "fold": int(member["development_fold"]),
                "category": member["category"],
                "label": int(member["label"]),
                "prediction": predictions[row_id],
                "baseline_error": predictions[row_id] != int(member["label"]),
                "semantic_singleton": component_counts[component] == 1,
                "name": data_rows[row_id]["name"],
                "description": data_rows[row_id]["description"],
                "images": images,
            }
        )
    selected: list[dict[str, Any]] = []
    used_components: set[str] = set()
    for category in CATEGORIES:
        for error in (True, False):
            for fold in range(5):
                quota = SPEC["fold_quotas"][
                    "flammable_errors_all_remaining"
                    if category == "Легковоспламеняющиеся" and error
                    else "default"
                ][fold]
                local = select_quota(
                    candidates,
                    category=category,
                    error=error,
                    fold=fold,
                    quota=quota,
                    used_components=used_components,
                )
                selected.extend(local)
                used_components.update(row["component"] for row in local)
    if len(selected) != SPEC["sample_rows"] or len({row["component"] for row in selected}) != len(selected):
        raise ValueError(
            "audit sample size or component uniqueness mismatch: "
            f"rows={len(selected)} unique_components={len({row['component'] for row in selected})}"
        )
    selected.sort(key=lambda row: stable_key(row["component"], row["id"]))

    output_dir.mkdir(parents=True)
    packet = output_dir / "blind_audit_120.csv"
    review = output_dir / "review_form_120.csv"
    mapping = {}
    with packet.open("x", encoding="utf-8", newline="") as pstream, review.open(
        "x", encoding="utf-8", newline=""
    ) as rstream:
        packet_writer = csv.DictWriter(pstream, fieldnames=IMMUTABLE_FIELDS)
        review_writer = csv.DictWriter(rstream, fieldnames=REVIEW_FIELDS)
        packet_writer.writeheader()
        review_writer.writeheader()
        for index, row in enumerate(selected, start=1):
            audit_id = f"E674-{index:03d}"
            evidence = []
            available_count = 0
            for image in row["images"]:
                key = (row["id"], image["image_index"])
                status = status_by_key[key]
                available_count += status == "OCR_AVAILABLE"
                evidence.append(
                    {
                        "image_index": image["image_index"],
                        "url": image["url"],
                        "status": status,
                        "regions": payload_by_key.get(key, []),
                    }
                )
            packet_writer.writerow(
                {
                    "audit_id": audit_id,
                    "category": row["category"],
                    "name": row["name"],
                    "description": row["description"],
                    "available_image_count": available_count,
                    "total_image_count": len(evidence),
                    "image_evidence_json": json.dumps(evidence, ensure_ascii=False),
                }
            )
            review_writer.writerow({field: audit_id if field == "audit_id" else "" for field in REVIEW_FIELDS})
            mapping[audit_id] = {
                "row_id": row["id"],
                "semantic_component": row["component"],
                "development_fold": row["fold"],
                "category": row["category"],
                "label": row["label"],
                "baseline_prediction": row["prediction"],
                "baseline_error": row["baseline_error"],
                "semantic_singleton": row["semantic_singleton"],
            }
    manifest = {
        "schema_version": "exp674_private_manifest_v1",
        "experiment_id": "674",
        "input_sha256": hashes,
        "blind_packet_sha256": sha256_file(packet),
        "review_form_sha256": sha256_file(review),
        "excluded_row_ids": len(excluded_ids),
        "excluded_semantic_components": len(excluded_components),
        "mapping": mapping,
        "sealed_rows_read": 0,
        "public_used": False,
    }
    private_manifest = output_dir / "private_manifest.json"
    private_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    strata = Counter(
        (value["category"], value["baseline_error"], value["development_fold"])
        for value in mapping.values()
    )
    summary = {
        "schema_version": "exp674_build_summary_v1",
        "rows": len(mapping),
        "unique_components": len({value["semantic_component"] for value in mapping.values()}),
        "strata": {"|".join(map(str, key)): count for key, count in sorted(strata.items())},
        "blind_packet_sha256": manifest["blind_packet_sha256"],
        "private_manifest_sha256": sha256_file(private_manifest),
        "excluded_row_ids": len(excluded_ids),
        "excluded_semantic_components": len(excluded_components),
        "sealed_rows_read": 0,
        "public_used": False,
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data",
        "folds",
        "baseline_predictions",
        "image_manifest",
        "ocr_availability",
        "ocr_available_images",
        "aggregate_integrity",
        "prior_490",
        "prior_reviewed_670",
        "prior_672",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    summary = build(**vars(parser.parse_args()))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

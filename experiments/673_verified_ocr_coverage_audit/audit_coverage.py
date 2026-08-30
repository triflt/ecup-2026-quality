#!/usr/bin/env python3
"""Verify experiment-660 OCR artifacts and measure exact-140 error coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "frozen_spec.json").read_text(encoding="utf-8"))
BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"
MARKERS = {
    BAD: re.compile(
        r"\bбад\b|биологически\s+активн|dietary\s+supplement|food\s+supplement|"
        r"не\s+является\s+лекар"
    ),
    FLAMMABLE: re.compile(
        r"бутан|пропан|\bгаз\b|топлив|горюч|легковоспламен|розжиг|"
        r"жидкост.{0,12}(огн|зажиг)|зажигал|спичк|керосин|этанол|метанол|парафин"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalize(value: object) -> str:
    return re.sub(r"[^0-9a-zа-я]+", " ", str(value or "").lower().replace("ё", "е")).strip()


def item_availability(statuses: list[str]) -> str:
    if not statuses or not set(statuses).issubset({"OCR_AVAILABLE", "OCR_UNAVAILABLE"}):
        raise ValueError("missing or invalid image availability")
    available = sum(value == "OCR_AVAILABLE" for value in statuses)
    if available == len(statuses):
        return "all_available"
    if available:
        return "partially_available"
    return "all_unavailable"


def _cohort(mask: list[bool], rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for keep, row in zip(mask, rows, strict=True) if keep]
    status = Counter(row["item_availability"] for row in selected)
    total = len(selected)
    any_available = total - status["all_unavailable"]
    return {
        "rows": total,
        "any_available": any_available,
        "any_available_fraction": any_available / total if total else None,
        "all_available": status["all_available"],
        "partially_available": status["partially_available"],
        "all_unavailable": status["all_unavailable"],
        "has_novel_ocr_region": sum(row["has_novel_ocr_region"] for row in selected),
        "has_broad_policy_marker": sum(row["has_broad_policy_marker"] for row in selected),
        "has_novel_broad_policy_marker": sum(
            row["has_novel_broad_policy_marker"] for row in selected
        ),
    }


def audit(
    *,
    data: Path,
    folds: Path,
    baseline_predictions: Path,
    image_manifest: Path,
    ocr_availability: Path,
    ocr_available_images: Path,
    aggregate_integrity: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    inputs = {
        "data": data,
        "folds": folds,
        "baseline_predictions": baseline_predictions,
        "image_manifest": image_manifest,
        "ocr_availability": ocr_availability,
        "ocr_available_images": ocr_available_images,
        "aggregate_integrity": aggregate_integrity,
    }
    observed_sha = {name: sha256_file(path) for name, path in inputs.items()}
    if observed_sha != SPEC["expected_sha256"]:
        raise ValueError("input SHA set differs from frozen spec")

    data_rows = read_csv(data)
    data_by_id = {row["id"]: row for row in data_rows}
    if len(data_rows) != 12971 or len(data_by_id) != len(data_rows):
        raise ValueError("competition data row contract mismatch")
    membership = read_csv(folds)
    development = [row for row in membership if row["split"] == "development"]
    sealed_rows_read = 0
    if len(development) != SPEC["development_rows"]:
        raise ValueError("development scope mismatch")
    prediction_rows = read_csv(baseline_predictions)
    if not prediction_rows or set(prediction_rows[0]) != {"id", "prediction", "source"}:
        raise ValueError("baseline prediction schema mismatch")
    predictions = {}
    for row in prediction_rows:
        if row["source"] != "exact_full140_semantic_v3" or int(row["prediction"]) not in (0, 1):
            raise ValueError("baseline prediction provenance mismatch")
        predictions[row["id"]] = int(row["prediction"])
    if len(predictions) != SPEC["development_rows"]:
        raise ValueError("baseline prediction scope mismatch")

    with image_manifest.open(encoding="utf-8") as stream:
        image_manifest_rows = [json.loads(line) for line in stream]
    if len(image_manifest_rows) != SPEC["expected_ocr_images"]:
        raise ValueError("source image manifest count mismatch")

    statuses: dict[str, list[str]] = defaultdict(list)
    status_by_key: dict[tuple[int, int], str] = {}
    availability_keys: set[tuple[int, int]] = set()
    with ocr_availability.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (int(row["global_index"]), int(row["image_index"]))
            if key in availability_keys:
                raise ValueError("duplicate OCR availability image key")
            availability_keys.add(key)
            global_index = key[0]
            if not 0 <= global_index < len(image_manifest_rows):
                raise ValueError("OCR global image index is out of range")
            source_image = image_manifest_rows[global_index]
            if (
                str(source_image["id"]) != str(row["id"])
                or int(source_image["image_index"]) != key[1]
                or sha256_text(str(source_image["url"])) != row["source_url_sha256"]
            ):
                raise ValueError("OCR global-index/id alignment mismatch")
            status = str(row["status"])
            status_by_key[key] = status
            reasons = row.get("reasons")
            if status == "OCR_AVAILABLE" and reasons != []:
                raise ValueError("available image unexpectedly has failure reasons")
            if status == "OCR_UNAVAILABLE" and not isinstance(reasons, list):
                raise ValueError("unavailable image lacks explicit failure reasons")
            statuses[str(row["id"])].append(status)
    if len(availability_keys) != SPEC["expected_ocr_images"]:
        raise ValueError("OCR availability image count mismatch")

    ocr_regions: dict[str, list[str]] = defaultdict(list)
    available_keys: set[tuple[int, int]] = set()
    with ocr_available_images.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (int(row["global_index"]), int(row["image_index"]))
            if key in available_keys or key not in availability_keys:
                raise ValueError("duplicate or foreign available OCR image key")
            available_keys.add(key)
            if status_by_key[key] != "OCR_AVAILABLE":
                raise ValueError("OCR payload exists outside OCR_AVAILABLE contract")
            regions = row.get("regions")
            if not isinstance(regions, list):
                raise TypeError("available OCR image lacks regions")
            for region in regions:
                text = region.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("OCR region lacks non-empty text")
                ocr_regions[str(row["id"])].append(text)
    if len(available_keys) != SPEC["expected_ocr_available"]:
        raise ValueError("available OCR payload count mismatch")
    if sum(value == "OCR_UNAVAILABLE" for values in statuses.values() for value in values) != SPEC[
        "expected_ocr_unavailable"
    ]:
        raise ValueError("unavailable OCR image count mismatch")

    aggregate = json.loads(aggregate_integrity.read_text(encoding="utf-8"))
    if (
        aggregate.get("decision") != "ACCEPT_PARTIAL_FAIL_CLOSED_OCR_DATASET"
        or aggregate.get("images") != SPEC["expected_ocr_images"]
        or aggregate.get("available_images") != SPEC["expected_ocr_available"]
        or aggregate.get("unavailable_images") != SPEC["expected_ocr_unavailable"]
    ):
        raise ValueError("aggregate OCR integrity contract mismatch")

    component_counts = Counter(row["semantic_component"] for row in development)
    rows: list[dict[str, Any]] = []
    for member in development:
        row_id = member["id"]
        source = data_by_id[row_id]
        category = member["category"]
        label = int(member["label"])
        prediction = predictions[row_id]
        card_text = normalize(f"{source['name']} {source['description']}")
        novel_region = False
        policy_marker = False
        novel_marker = False
        for region in ocr_regions[row_id]:
            normalized = normalize(region)
            if not normalized:
                continue
            novel = normalized not in card_text
            marker = bool(MARKERS[category].search(normalized))
            novel_region |= novel
            policy_marker |= marker
            novel_marker |= marker and novel
        rows.append(
            {
                "id": row_id,
                "category": category,
                "fold": int(member["development_fold"]),
                "baseline_error": prediction != label,
                "semantic_singleton": component_counts[member["semantic_component"]] == 1,
                "item_availability": item_availability(statuses[row_id]),
                "has_novel_ocr_region": novel_region,
                "has_broad_policy_marker": policy_marker,
                "has_novel_broad_policy_marker": novel_marker,
            }
        )
    if sum(row["baseline_error"] for row in rows) != SPEC["exact140_errors"]:
        raise ValueError("exact-140 error count differs from frozen replay")

    cohorts = {
        "all_development": _cohort([True] * len(rows), rows),
        "baseline_errors": _cohort([row["baseline_error"] for row in rows], rows),
        "baseline_correct": _cohort([not row["baseline_error"] for row in rows], rows),
        "singleton_errors": _cohort(
            [row["baseline_error"] and row["semantic_singleton"] for row in rows], rows
        ),
    }
    by_category = {
        category: {
            "errors": _cohort(
                [row["baseline_error"] and row["category"] == category for row in rows], rows
            ),
            "correct": _cohort(
                [not row["baseline_error"] and row["category"] == category for row in rows], rows
            ),
        }
        for category in (BAD, FLAMMABLE)
    }
    by_fold = {
        str(fold): _cohort(
            [row["baseline_error"] and row["fold"] == fold for row in rows], rows
        )
        for fold in range(5)
    }
    result = {
        "schema_version": "exp673_coverage_audit_v1",
        "experiment_id": "673",
        "status": "INFRASTRUCTURE_ACCEPTED_CRITICAL_SPAN_RECALL_UNMEASURED",
        "input_sha256": observed_sha,
        "ocr_integrity": {
            "images": len(availability_keys),
            "available_images": len(available_keys),
            "unavailable_images": len(availability_keys) - len(available_keys),
            "unavailable_payload_violations": 0,
        },
        "cohorts": cohorts,
        "by_category": by_category,
        "baseline_errors_by_fold": by_fold,
        "critical_span_recall": None,
        "unsupported_critical_span_rate": None,
        "h2_accepted": False,
        "next_action": "BUILD_BLIND_VISUAL_CRITICAL_SPAN_AUDIT_674",
        "sealed_rows_read": sealed_rows_read,
        "public_used": False,
        "gpu_hours": 0.0,
        "notes": (
            "Broad policy-marker counts are descriptive candidate retrieval only; they do not "
            "establish relevance, OCR recall, a decision rule, or a quality gain."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--ocr-availability", type=Path, required=True)
    parser.add_argument("--ocr-available-images", type=Path, required=True)
    parser.add_argument("--aggregate-integrity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = audit(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

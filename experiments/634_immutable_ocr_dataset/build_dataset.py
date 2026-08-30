from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DATASET_VERSION = "competition_train_paddleocr_vl16_spotting_v1"
SOURCE_EXPERIMENT_ID = "633"
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
MODEL_REVISION = "c5630abae1d940eafe0697512a0325494b02ab42"
EXPECTED_ITEMS = 12_971
EXPECTED_IMAGES = 49_456
CONFIDENCE_TYPE = "sequence_geomean_token_probability"
SCHEMA_FILE = Path(__file__).with_name("ocr_image_record.schema.json")
LOC_TOKEN_PATTERN = re.compile(r"<\|LOC_(\d+)\|>")

MANIFEST_KEYS = {"id", "image_index", "url"}
REPORT_KEYS = {
    "schema_version",
    "experiment_id",
    "model_id",
    "model_revision",
    "manifest_sha256",
    "output_sha256",
    "shard_index",
    "num_shards",
    "requested_images",
    "successful_images",
    "failed_images",
    "parseable_images",
    "parseable_fraction_of_successful",
    "detections",
    "elapsed_seconds",
    "images_per_second",
}
SUCCESS_KEYS = {
    "id",
    "image_index",
    "global_index",
    "width",
    "height",
    "raw_generation",
    "sequence_confidence",
    "detections",
    "error",
}
ERROR_KEYS = {"id", "image_index", "global_index", "error"}
DETECTION_KEYS = {
    "text",
    "polygon",
    "normalized_polygon",
    "confidence",
    "confidence_type",
}
REPAIR_REPORT_KEYS = {
    "schema_version",
    "experiment_id",
    "source_experiment_id",
    "source_manifest_sha256",
    "source_shards",
    "model_id",
    "model_revision",
    "source_max_new_tokens",
    "repair_max_new_tokens",
    "all_truncation_candidates",
    "repair_shard_index",
    "num_repair_shards",
    "requested_images",
    "successful_images",
    "failed_images",
    "accepted_images",
    "detections",
    "elapsed_seconds",
    "output_sha256",
    "labels_read",
    "folds_read",
    "public_used",
}
FORBIDDEN_KEYS = {
    "answer",
    "category",
    "class",
    "fold",
    "gold",
    "is_public",
    "label",
    "labels",
    "public",
    "sealed",
    "sealed_membership",
    "split",
    "target",
    "targets",
    "validation_fold",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def is_parseable_generation(raw: object, detections: list[dict[str, Any]]) -> bool:
    """Match experiment 633: a canonical empty-page answer is valid OCR output."""
    return bool(detections) or str(raw).strip() in {"", "</s>"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{context} schema mismatch: missing={missing}, extra={extra}")


def _reject_forbidden_keys(value: Any, context: str = "root") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).strip().casefold()
            if normalized_key in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden label/split key at {context}.{key}")
            _reject_forbidden_keys(nested, f"{context}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_forbidden_keys(nested, f"{context}[{index}]")


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.casefold().split())


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_confidence(value: Any, context: str) -> float | None:
    if value is None:
        return None
    _require(_is_number(value), f"{context} must be a number or null")
    result = float(value)
    _require(math.isfinite(result) and 0.0 <= result <= 1.0, f"{context} outside [0, 1]")
    return result


def _polygon_area(points: tuple[tuple[int, int], ...]) -> float:
    return (
        abs(
            sum(
                points[index][0] * points[(index + 1) % 4][1]
                - points[(index + 1) % 4][0] * points[index][1]
                for index in range(4)
            )
        )
        / 2.0
    )


def _orientation(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    return (value > 0) - (value < 0)


def _strictly_intersects(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
    d: tuple[int, int],
) -> bool:
    return _orientation(a, b, c) != _orientation(a, b, d) and _orientation(c, d, a) != _orientation(
        c, d, b
    )


def validate_quadrilateral(value: Any, context: str) -> tuple[tuple[int, int], ...]:
    _require(isinstance(value, list) and len(value) == 4, f"{context} must contain four points")
    points: list[tuple[int, int]] = []
    for point_index, point in enumerate(value):
        _require(
            isinstance(point, list) and len(point) == 2, f"{context}[{point_index}] must be [x, y]"
        )
        x, y = point
        _require(
            type(x) is int and type(y) is int, f"{context}[{point_index}] must contain integers"
        )
        _require(0 <= x <= 1000 and 0 <= y <= 1000, f"{context}[{point_index}] outside 0..1000")
        points.append((x, y))
    result = tuple(points)
    _require(len(set(result)) >= 3 and _polygon_area(result) > 0, f"{context} is degenerate")
    _require(
        not _strictly_intersects(result[0], result[1], result[2], result[3])
        and not _strictly_intersects(result[1], result[2], result[3], result[0]),
        f"{context} is self-intersecting",
    )
    return result


def load_manifest(path: Path, *, expected_items: int, expected_images: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as stream:
        for global_index, line in enumerate(stream):
            row = json.loads(line)
            _require(isinstance(row, dict), f"manifest line {global_index + 1} is not an object")
            _reject_forbidden_keys(row, f"manifest[{global_index}]")
            _exact_keys(row, MANIFEST_KEYS, f"manifest[{global_index}]")
            item_id = row["id"]
            image_index = row["image_index"]
            url = row["url"]
            _require(isinstance(item_id, str) and item_id, f"invalid manifest id at {global_index}")
            _require(
                type(image_index) is int and image_index >= 0,
                f"invalid image_index at {global_index}",
            )
            _require(
                isinstance(url, str) and url.startswith("https://"),
                f"invalid URL at {global_index}",
            )
            key = (item_id, image_index)
            _require(key not in seen_keys, f"duplicate manifest image key: {key}")
            seen_keys.add(key)
            rows.append(
                {
                    "id": item_id,
                    "image_index": image_index,
                    "url": url,
                    "global_index": global_index,
                }
            )
    _require(
        len(rows) == expected_images, f"manifest images={len(rows)}, expected={expected_images}"
    )
    item_count = len({row["id"] for row in rows})
    _require(
        item_count == expected_items, f"manifest items={item_count}, expected={expected_items}"
    )
    return rows


def _validate_report(
    report: dict[str, Any],
    *,
    report_path: Path,
    spotting_path: Path,
    manifest_sha256: str,
) -> None:
    _reject_forbidden_keys(report, "shard_report")
    _exact_keys(report, REPORT_KEYS, f"report {report_path.name}")
    _require(report["schema_version"] == 1, "unsupported exp633 report schema")
    _require(report["experiment_id"] == SOURCE_EXPERIMENT_ID, "wrong source experiment")
    _require(report["model_id"] == MODEL_ID, "wrong OCR model")
    _require(report["model_revision"] == MODEL_REVISION, "wrong OCR model revision")
    _require(report["manifest_sha256"] == manifest_sha256, "shard manifest checksum mismatch")
    _require(
        report["output_sha256"] == sha256_file(spotting_path), "shard output checksum mismatch"
    )
    for key in (
        "shard_index",
        "num_shards",
        "requested_images",
        "successful_images",
        "failed_images",
        "parseable_images",
        "detections",
    ):
        _require(type(report[key]) is int and report[key] >= 0, f"invalid report field {key}")
    _require(report["num_shards"] > 0, "num_shards must be positive")
    _require(report["shard_index"] < report["num_shards"], "invalid shard_index")
    _require(
        report["successful_images"] + report["failed_images"] == report["requested_images"],
        "inconsistent report image counts",
    )


def _canonical_region(
    detection: dict[str, Any],
    *,
    detection_index: int,
    global_index: int,
    width: int,
    height: int,
    sequence_confidence: float | None,
) -> tuple[tuple[str, tuple[tuple[int, int], ...]], dict[str, Any]]:
    _reject_forbidden_keys(detection, f"detection[{detection_index}]")
    _exact_keys(detection, DETECTION_KEYS, f"detection[{detection_index}]")
    text = detection["text"]
    _require(
        isinstance(text, str) and text == text.strip() and bool(text), "invalid detection text"
    )
    normalized_text = normalize_text(text)
    _require(bool(normalized_text), "empty normalized detection text")
    quad = validate_quadrilateral(
        detection["normalized_polygon"], f"detection[{detection_index}].normalized_polygon"
    )
    pixel = detection["polygon"]
    _require(isinstance(pixel, list) and len(pixel) == 4, "pixel polygon must contain four points")
    expected_pixel = [[round(x * width / 1000), round(y * height / 1000)] for x, y in quad]
    _require(
        pixel == expected_pixel, "pixel polygon disagrees with normalized polygon and image size"
    )
    _require(detection["confidence_type"] == CONFIDENCE_TYPE, "unexpected confidence semantics")
    detection_confidence = _validate_confidence(detection["confidence"], "detection confidence")
    _require(
        detection_confidence == sequence_confidence,
        "region confidence differs from sequence confidence",
    )
    key = (normalized_text, quad)
    region = {
        "region_id": f"ocr-{global_index:05d}-{detection_index:04d}",
        "text": text,
        "normalized_text": normalized_text,
        "quadrilateral_normalized": [list(point) for point in quad],
        "confidence_ref": "generation_confidence",
        "source_occurrences": [{"detection_index": detection_index, "text": text}],
        "technical_duplicate_count": 0,
    }
    return key, region


def canonicalize_success_record(
    source: dict[str, Any], manifest_row: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    _reject_forbidden_keys(source, f"source[{manifest_row['global_index']}]")
    _exact_keys(source, SUCCESS_KEYS, f"source[{manifest_row['global_index']}]")
    _require(source["error"] is None, "successful record has non-null error")
    for key in ("id", "image_index", "global_index"):
        _require(source[key] == manifest_row[key], f"source {key} differs from manifest")
    width = source["width"]
    height = source["height"]
    _require(type(width) is int and width > 0, "invalid image width")
    _require(type(height) is int and height > 0, "invalid image height")
    raw = source["raw_generation"]
    _require(isinstance(raw, str), "raw_generation must be a string")
    confidence = _validate_confidence(source["sequence_confidence"], "sequence_confidence")
    detections = source["detections"]
    _require(isinstance(detections, list), "detections must be a list")

    loc_tokens = LOC_TOKEN_PATTERN.findall(raw)
    _require(len(loc_tokens) % 8 == 0, "malformed location-token count")
    stripped_raw = raw.strip()
    _require(
        stripped_raw in {"", "</s>"} or stripped_raw.endswith("</s>"),
        "OCR generation did not terminate with EOS",
    )
    _require(
        len(detections) == len(loc_tokens) // 8,
        "parsed detections disagree with location-token blocks",
    )

    deduplicated: dict[tuple[str, tuple[tuple[int, int], ...]], dict[str, Any]] = {}
    duplicate_count = 0
    for detection_index, detection in enumerate(detections):
        _require(isinstance(detection, dict), f"detection[{detection_index}] is not an object")
        key, region = _canonical_region(
            detection,
            detection_index=detection_index,
            global_index=int(source["global_index"]),
            width=width,
            height=height,
            sequence_confidence=confidence,
        )
        if key in deduplicated:
            existing = deduplicated[key]
            existing["source_occurrences"].extend(region["source_occurrences"])
            existing["technical_duplicate_count"] += 1
            duplicate_count += 1
        else:
            deduplicated[key] = region

    regions = list(deduplicated.values())
    record = {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "id": str(source["id"]),
        "image_index": int(source["image_index"]),
        "global_index": int(source["global_index"]),
        "source_url_sha256": sha256_text(str(manifest_row["url"])),
        "width": width,
        "height": height,
        "raw_generation": raw,
        "generation_confidence": {
            "value": confidence,
            "type": CONFIDENCE_TYPE,
            "scope": "complete_generated_sequence",
            "calibrated": False,
            "region_specific": False,
        },
        "parse_status": "regions_parsed" if regions else "no_location_tokens",
        "regions": regions,
    }
    return record, duplicate_count


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            _require(isinstance(value, dict), f"{path.name}:{line_number} is not an object")
            yield line_number, value


def collect_shards(
    shard_dirs: list[Path],
    *,
    manifest_rows: list[dict[str, Any]],
    manifest_sha256: str,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    _require(bool(shard_dirs), "at least one --shard-dir is required")
    by_global_index: dict[int, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    seen_shard_indices: set[int] = set()
    expected_num_shards: int | None = None
    for shard_dir in shard_dirs:
        report_path = shard_dir / "report.json"
        spotting_path = shard_dir / "spotting.jsonl"
        _require(
            report_path.is_file() and spotting_path.is_file(),
            "each shard directory needs report.json and spotting.jsonl",
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _require(isinstance(report, dict), "shard report is not an object")
        _validate_report(
            report,
            report_path=report_path,
            spotting_path=spotting_path,
            manifest_sha256=manifest_sha256,
        )
        shard_index = int(report["shard_index"])
        num_shards = int(report["num_shards"])
        if expected_num_shards is None:
            expected_num_shards = num_shards
        _require(num_shards == expected_num_shards, "shards disagree on num_shards")
        _require(shard_index not in seen_shard_indices, f"duplicate shard index {shard_index}")
        seen_shard_indices.add(shard_index)

        row_count = 0
        successful = 0
        failed = 0
        detection_count = 0
        parseable_images = 0
        for line_number, row in _iter_jsonl(spotting_path):
            _reject_forbidden_keys(row, f"shard[{shard_index}][{line_number}]")
            global_index = row.get("global_index")
            _require(type(global_index) is int, "missing or invalid global_index")
            _require(0 <= global_index < len(manifest_rows), "global_index outside manifest")
            _require(global_index % num_shards == shard_index, "row assigned to wrong shard")
            _require(
                global_index not in by_global_index,
                f"duplicate OCR image global_index={global_index}",
            )
            expected = manifest_rows[global_index]
            if row.get("error") is None:
                _exact_keys(row, SUCCESS_KEYS, f"shard[{shard_index}][{line_number}]")
                successful += 1
                detections = row["detections"]
                _require(isinstance(detections, list), "detections must be a list")
                detection_count += len(detections)
                parseable_images += is_parseable_generation(row["raw_generation"], detections)
            else:
                _exact_keys(row, ERROR_KEYS, f"shard[{shard_index}][{line_number}]")
                failed += 1
            for key in ("id", "image_index", "global_index"):
                _require(row[key] == expected[key], f"shard row {key} differs from manifest")
            by_global_index[global_index] = row
            row_count += 1

        _require(row_count == report["requested_images"], "shard row count differs from report")
        _require(
            successful == report["successful_images"], "shard successful count differs from report"
        )
        _require(failed == report["failed_images"], "shard failed count differs from report")
        _require(
            detection_count == report["detections"], "shard detection count differs from report"
        )
        _require(
            parseable_images == report["parseable_images"],
            "shard parseable count differs from report",
        )
        provenance.append(
            {
                "shard_index": shard_index,
                "num_shards": num_shards,
                "spotting_sha256": sha256_file(spotting_path),
                "report_sha256": sha256_file(report_path),
                "rows": row_count,
                "successful_images": successful,
                "failed_images": failed,
                "detections": detection_count,
            }
        )

    _require(expected_num_shards is not None, "no shard metadata found")
    _require(
        seen_shard_indices == set(range(expected_num_shards)),
        f"incomplete shard indices: got={sorted(seen_shard_indices)}, expected=0..{expected_num_shards - 1}",
    )
    missing = sorted(set(range(len(manifest_rows))) - set(by_global_index))
    _require(not missing, f"missing OCR rows: count={len(missing)}, first={missing[:5]}")
    return by_global_index, sorted(provenance, key=lambda row: int(row["shard_index"]))


def repair_reasons(source: dict[str, Any]) -> list[str]:
    if source.get("error") is not None:
        raise ValueError("processing error remains in source OCR")
    raw = source["raw_generation"]
    detections = source["detections"]
    _require(isinstance(raw, str), "source raw_generation must be a string")
    _require(isinstance(detections, list), "source detections must be a list")
    stripped = raw.strip()
    reasons: list[str] = []
    if len(LOC_TOKEN_PATTERN.findall(raw)) % 8:
        reasons.append("partial_location_block")
    if not detections and stripped not in {"", "</s>"}:
        reasons.append("noncanonical_unparseable_generation")
    if stripped not in {"", "</s>"} and not stripped.endswith("</s>"):
        reasons.append("generation_reached_limit_without_eos")
    return reasons


def apply_repair_overlay(
    shard_rows: dict[int, dict[str, Any]],
    repair_dirs: list[Path],
    *,
    manifest_rows: list[dict[str, Any]],
    manifest_sha256: str,
    source_provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [index for index in sorted(shard_rows) if repair_reasons(shard_rows[index])]
    _require(bool(candidates), "repair overlay supplied but no source row requires repair")
    reports: list[dict[str, Any]] = []
    repaired: dict[int, dict[str, Any]] = {}
    expected_num_shards: int | None = None
    seen_repair_shards: set[int] = set()
    for repair_dir in repair_dirs:
        report_path = repair_dir / "report.json"
        spotting_path = repair_dir / "spotting.jsonl"
        _require(
            report_path.is_file() and spotting_path.is_file(),
            "each repair directory needs report.json and spotting.jsonl",
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _require(isinstance(report, dict), "repair report is not an object")
        _exact_keys(report, REPAIR_REPORT_KEYS, "repair report")
        _reject_forbidden_keys(
            {key: value for key, value in report.items() if key != "public_used"},
            "repair report",
        )
        _require(report["schema_version"] == 1, "unsupported repair schema")
        _require(report["experiment_id"] == "655", "wrong repair experiment")
        _require(report["source_experiment_id"] == SOURCE_EXPERIMENT_ID, "wrong repair source")
        _require(report["source_manifest_sha256"] == manifest_sha256, "repair manifest mismatch")
        _require(report["source_shards"] == source_provenance, "repair source provenance mismatch")
        _require(report["model_id"] == MODEL_ID, "wrong repair model")
        _require(report["model_revision"] == MODEL_REVISION, "wrong repair model revision")
        _require(report["source_max_new_tokens"] == 512, "wrong source generation limit")
        _require(report["repair_max_new_tokens"] > 512, "repair limit did not increase")
        _require(
            report["labels_read"] == 0
            and report["folds_read"] == 0
            and report["public_used"] is False,
            "repair overlay is not label-blind",
        )
        _require(
            report["all_truncation_candidates"] == len(candidates),
            "repair candidate count mismatch",
        )
        repair_shard = report["repair_shard_index"]
        num_repair_shards = report["num_repair_shards"]
        _require(type(repair_shard) is int and type(num_repair_shards) is int, "invalid repair shard")
        _require(0 <= repair_shard < num_repair_shards, "repair shard outside range")
        if expected_num_shards is None:
            expected_num_shards = num_repair_shards
        _require(num_repair_shards == expected_num_shards, "repair reports disagree on shard count")
        _require(repair_shard not in seen_repair_shards, "duplicate repair shard")
        seen_repair_shards.add(repair_shard)
        _require(
            report["failed_images"] == 0
            and report["successful_images"] == report["requested_images"]
            and report["accepted_images"] == report["requested_images"],
            "repair shard did not accept every requested image",
        )
        _require(report["output_sha256"] == sha256_file(spotting_path), "repair output checksum mismatch")
        expected_indices = candidates[repair_shard::num_repair_shards]
        observed_indices: list[int] = []
        for line_number, row in _iter_jsonl(spotting_path):
            _reject_forbidden_keys(row, f"repair[{repair_shard}][{line_number}]")
            _exact_keys(row, SUCCESS_KEYS, f"repair[{repair_shard}][{line_number}]")
            _require(row["error"] is None, "accepted repair row has an error")
            global_index = row["global_index"]
            _require(type(global_index) is int, "repair global_index is invalid")
            _require(global_index not in repaired, "duplicate repaired global_index")
            _require(global_index in shard_rows, "repair global_index outside source")
            expected_identity = manifest_rows[global_index]
            for key in ("id", "image_index", "global_index"):
                _require(row[key] == expected_identity[key], f"repair {key} differs from manifest")
            _require(not repair_reasons(row), "repair output remains truncated or unparseable")
            repaired[global_index] = row
            observed_indices.append(global_index)
        _require(observed_indices == expected_indices, "repair shard rows differ from frozen assignment")
        _require(len(observed_indices) == report["requested_images"], "repair row count mismatch")
        reports.append(report)
    _require(expected_num_shards is not None, "no repair reports found")
    _require(
        seen_repair_shards == set(range(expected_num_shards)),
        "repair shard set is incomplete",
    )
    _require(set(repaired) == set(candidates), "repair overlay does not cover every candidate")
    shard_rows.update(repaired)
    _require(
        not [index for index in sorted(shard_rows) if repair_reasons(shard_rows[index])],
        "truncated OCR remains after repair overlay",
    )
    return {
        "experiment_id": "655",
        "candidate_images": len(candidates),
        "repaired_images": len(repaired),
        "repair_shards": expected_num_shards,
        "repair_max_new_tokens": sorted({report["repair_max_new_tokens"] for report in reports}),
        "reports_sha256": [sha256_file(path / "report.json") for path in repair_dirs],
    }


def _bundle_digest(file_hashes: dict[str, str]) -> str:
    payload = "".join(f"{name}\0{file_hashes[name]}\n" for name in sorted(file_hashes))
    return sha256_text(payload)


def build_dataset(
    *,
    manifest_path: Path,
    shard_dirs: list[Path],
    output_dir: Path,
    repair_dirs: list[Path] | None = None,
    expected_items: int = EXPECTED_ITEMS,
    expected_images: int = EXPECTED_IMAGES,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output directory: {output_dir}")
    _require(SCHEMA_FILE.is_file(), f"missing schema: {SCHEMA_FILE}")
    manifest_sha256 = sha256_file(manifest_path)
    manifest_rows = load_manifest(
        manifest_path,
        expected_items=expected_items,
        expected_images=expected_images,
    )
    shard_rows, shard_provenance = collect_shards(
        shard_dirs,
        manifest_rows=manifest_rows,
        manifest_sha256=manifest_sha256,
    )
    repair_candidates = [
        index for index in sorted(shard_rows) if repair_reasons(shard_rows[index])
    ]
    repair_provenance = None
    if repair_candidates:
        _require(bool(repair_dirs), f"OCR repair required for {len(repair_candidates)} images")
        repair_provenance = apply_repair_overlay(
            shard_rows,
            list(repair_dirs or []),
            manifest_rows=manifest_rows,
            manifest_sha256=manifest_sha256,
            source_provenance=shard_provenance,
        )
    else:
        _require(not repair_dirs, "repair overlay supplied but source is already complete")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        data_path = temporary / "ocr_images.jsonl"
        total_regions = 0
        technical_duplicates = 0
        parse_status = Counter()
        confidence_null = 0
        with data_path.open("w", encoding="utf-8") as output:
            for manifest_row in manifest_rows:
                source = shard_rows[int(manifest_row["global_index"])]
                if source.get("error") is not None:
                    raise ValueError(
                        f"processing error remains at global_index={manifest_row['global_index']}: {source['error']}"
                    )
                record, duplicate_count = canonicalize_success_record(source, manifest_row)
                total_regions += len(record["regions"])
                technical_duplicates += duplicate_count
                parse_status[str(record["parse_status"])] += 1
                confidence_null += record["generation_confidence"]["value"] is None
                output.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )

        copied_schema = temporary / SCHEMA_FILE.name
        shutil.copyfile(SCHEMA_FILE, copied_schema)
        merge_report = {
            "schema_version": 1,
            "experiment_id": "634",
            "dataset_version": DATASET_VERSION,
            "status": "accepted",
            "scope": {
                "items": expected_items,
                "images": expected_images,
                "manifest_coverage": 1.0,
                "missing_images": 0,
                "duplicate_images": 0,
                "processing_errors": 0,
            },
            "parseability": {
                "regions_parsed_images": parse_status["regions_parsed"],
                "no_location_token_images": parse_status["no_location_tokens"],
                "malformed_location_images": 0,
                "location_output_parseable_fraction": 1.0,
            },
            "regions": {
                "canonical_regions": total_regions,
                "technical_duplicates_merged": technical_duplicates,
                "out_of_bounds_regions": 0,
                "cross_image_duplicates_removed": 0,
            },
            "confidence": {
                "type": CONFIDENCE_TYPE,
                "scope": "complete_generated_sequence",
                "calibrated": False,
                "region_specific": False,
                "null_values": confidence_null,
            },
            "privacy": {
                "labels_read": 0,
                "categories_read": 0,
                "folds_read": 0,
                "sealed_membership_used": False,
                "public_used": False,
            },
        }
        merge_report_path = temporary / "merge_report.json"
        merge_report_path.write_text(
            json.dumps(merge_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        provenance = {
            "schema_version": 1,
            "dataset_version": DATASET_VERSION,
            "source_data_version": "competition_train_v1",
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "source_manifest": {
                "filename": manifest_path.name,
                "sha256": manifest_sha256,
                "rows": expected_images,
                "items": expected_items,
            },
            "ocr_model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "builder": {
                "experiment_id": "634",
                "filename": Path(__file__).name,
                "sha256": sha256_file(Path(__file__)),
            },
            "shards": shard_provenance,
            "repair_overlay": repair_provenance,
            "deduplication": {
                "scope": "within_same_id_and_image_index_only",
                "key": "nfkc_casefold_whitespace_text_plus_exact_normalized_quadrilateral",
                "cross_image_deduplication": False,
                "source_occurrences_preserved": True,
            },
            "privacy": {
                "labels_read": 0,
                "categories_read": 0,
                "folds_read": 0,
                "sealed_membership_used": False,
                "public_used": False,
            },
        }
        provenance_path = temporary / "provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        file_hashes = {
            data_path.name: sha256_file(data_path),
            copied_schema.name: sha256_file(copied_schema),
            merge_report_path.name: sha256_file(merge_report_path),
            provenance_path.name: sha256_file(provenance_path),
        }
        dataset_manifest = {
            "schema_version": 1,
            "dataset_version": DATASET_VERSION,
            "immutable": True,
            "dataset_file": data_path.name,
            "dataset_sha256": file_hashes[data_path.name],
            "bundle_sha256": _bundle_digest(file_hashes),
            "files": file_hashes,
            "records": expected_images,
            "items": expected_items,
            "labels_read": 0,
            "categories_read": 0,
            "sealed_membership_used": False,
            "public_used": False,
        }
        (temporary / "dataset_manifest.json").write_text(
            json.dumps(dataset_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_bundle(output_dir, expected_items=expected_items, expected_images=expected_images)


def verify_bundle(
    dataset_dir: Path,
    *,
    expected_items: int = EXPECTED_ITEMS,
    expected_images: int = EXPECTED_IMAGES,
) -> dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    _require(manifest_path.is_file(), "dataset_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict), "dataset manifest is not an object")
    expected_manifest_keys = {
        "schema_version",
        "dataset_version",
        "immutable",
        "dataset_file",
        "dataset_sha256",
        "bundle_sha256",
        "files",
        "records",
        "items",
        "labels_read",
        "categories_read",
        "sealed_membership_used",
        "public_used",
    }
    _exact_keys(manifest, expected_manifest_keys, "dataset_manifest")
    _reject_forbidden_keys(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"sealed_membership_used", "public_used"}
        },
        "dataset_manifest",
    )
    _require(manifest["schema_version"] == 1, "unsupported dataset manifest schema")
    _require(manifest["dataset_version"] == DATASET_VERSION, "wrong dataset version")
    _require(manifest["immutable"] is True, "dataset is not immutable")
    _require(
        manifest["records"] == expected_images and manifest["items"] == expected_items,
        "wrong dataset scope",
    )
    _require(
        manifest["labels_read"] == 0 and manifest["categories_read"] == 0,
        "dataset is not label-blind",
    )
    _require(
        manifest["sealed_membership_used"] is False and manifest["public_used"] is False,
        "sealed/Public use is forbidden",
    )
    files = manifest["files"]
    _require(isinstance(files, dict) and bool(files), "missing bundle file hashes")
    _require(
        set(files)
        == {
            "ocr_images.jsonl",
            "ocr_image_record.schema.json",
            "merge_report.json",
            "provenance.json",
        },
        "unexpected immutable bundle file set",
    )
    for filename, expected_sha in files.items():
        _require(Path(filename).name == filename, "bundle filenames must be flat")
        path = dataset_dir / filename
        _require(path.is_file(), f"missing bundle file: {filename}")
        _require(sha256_file(path) == expected_sha, f"bundle checksum mismatch: {filename}")
    _require(_bundle_digest(files) == manifest["bundle_sha256"], "bundle digest mismatch")
    _require(
        files[manifest["dataset_file"]] == manifest["dataset_sha256"], "dataset checksum mismatch"
    )

    merge_report = json.loads((dataset_dir / "merge_report.json").read_text(encoding="utf-8"))
    _require(isinstance(merge_report, dict), "merge report is not an object")
    _require(merge_report.get("experiment_id") == "634", "wrong merge experiment")
    _require(merge_report.get("dataset_version") == DATASET_VERSION, "wrong merge dataset version")
    _require(merge_report.get("status") == "accepted", "merge report is not accepted")
    scope = merge_report.get("scope")
    _require(isinstance(scope, dict), "merge scope is missing")
    _require(
        scope.get("items") == expected_items and scope.get("images") == expected_images,
        "merge scope mismatch",
    )
    _require(scope.get("manifest_coverage") == 1.0, "incomplete manifest coverage")
    _require(
        scope.get("missing_images") == 0
        and scope.get("duplicate_images") == 0
        and scope.get("processing_errors") == 0,
        "merge integrity gate failed",
    )
    _require(
        merge_report.get("parseability", {}).get("malformed_location_images") == 0,
        "malformed OCR remains",
    )
    _require(
        merge_report.get("regions", {}).get("out_of_bounds_regions") == 0,
        "out-of-bounds OCR remains",
    )
    report_privacy = merge_report.get("privacy")
    _require(
        isinstance(report_privacy, dict)
        and report_privacy.get("labels_read") == 0
        and report_privacy.get("categories_read") == 0
        and report_privacy.get("folds_read") == 0
        and report_privacy.get("sealed_membership_used") is False
        and report_privacy.get("public_used") is False,
        "merge report is not label-blind",
    )

    provenance = json.loads((dataset_dir / "provenance.json").read_text(encoding="utf-8"))
    _require(isinstance(provenance, dict), "provenance is not an object")
    _require(
        provenance.get("dataset_version") == DATASET_VERSION, "wrong provenance dataset version"
    )
    _require(
        provenance.get("source_experiment_id") == SOURCE_EXPERIMENT_ID, "wrong provenance source"
    )
    _require(
        provenance.get("ocr_model") == {"id": MODEL_ID, "revision": MODEL_REVISION},
        "wrong OCR provenance",
    )
    source_manifest = provenance.get("source_manifest")
    _require(isinstance(source_manifest, dict), "source manifest provenance is missing")
    _require(
        source_manifest.get("rows") == expected_images
        and source_manifest.get("items") == expected_items
        and isinstance(source_manifest.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", source_manifest["sha256"]) is not None,
        "source manifest provenance mismatch",
    )
    shards = provenance.get("shards")
    _require(isinstance(shards, list) and bool(shards), "shard provenance is missing")
    shard_indices = [shard.get("shard_index") for shard in shards if isinstance(shard, dict)]
    _require(
        shard_indices == list(range(len(shards))), "shard provenance is incomplete or unordered"
    )
    _require(
        sum(int(shard["rows"]) for shard in shards) == expected_images,
        "shard provenance rows mismatch",
    )
    for shard in shards:
        _require(
            shard.get("failed_images") == 0
            and isinstance(shard.get("spotting_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", shard["spotting_sha256"]) is not None
            and isinstance(shard.get("report_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", shard["report_sha256"]) is not None,
            "invalid shard provenance",
        )
    provenance_privacy = provenance.get("privacy")
    _require(
        isinstance(provenance_privacy, dict)
        and provenance_privacy.get("labels_read") == 0
        and provenance_privacy.get("categories_read") == 0
        and provenance_privacy.get("folds_read") == 0
        and provenance_privacy.get("sealed_membership_used") is False
        and provenance_privacy.get("public_used") is False,
        "provenance is not label-blind",
    )

    data_path = dataset_dir / str(manifest["dataset_file"])
    seen_global: set[int] = set()
    seen_keys: set[tuple[str, int]] = set()
    item_ids: set[str] = set()
    rows = 0
    for line_number, record in _iter_jsonl(data_path):
        _reject_forbidden_keys(record, f"dataset[{line_number}]")
        required = {
            "schema_version",
            "dataset_version",
            "id",
            "image_index",
            "global_index",
            "source_url_sha256",
            "width",
            "height",
            "raw_generation",
            "generation_confidence",
            "parse_status",
            "regions",
        }
        _exact_keys(record, required, f"dataset[{line_number}]")
        _require(
            record["schema_version"] == 1 and record["dataset_version"] == DATASET_VERSION,
            "record contract mismatch",
        )
        global_index = record["global_index"]
        image_index = record["image_index"]
        item_id = record["id"]
        _require(
            type(global_index) is int and global_index == rows,
            "dataset rows are not in canonical global order",
        )
        _require(type(image_index) is int and image_index >= 0, "invalid image_index")
        _require(isinstance(item_id, str) and item_id, "invalid item id")
        _require(
            isinstance(record["source_url_sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", record["source_url_sha256"]) is not None,
            "invalid source URL checksum",
        )
        _require(type(record["width"]) is int and record["width"] > 0, "invalid record width")
        _require(type(record["height"]) is int and record["height"] > 0, "invalid record height")
        _require(isinstance(record["raw_generation"], str), "raw_generation must be a string")
        confidence = record["generation_confidence"]
        _require(isinstance(confidence, dict), "generation_confidence is not an object")
        _exact_keys(
            confidence,
            {"value", "type", "scope", "calibrated", "region_specific"},
            "generation_confidence",
        )
        _validate_confidence(confidence["value"], "generation_confidence.value")
        _require(confidence["type"] == CONFIDENCE_TYPE, "invalid confidence type")
        _require(confidence["scope"] == "complete_generated_sequence", "invalid confidence scope")
        _require(confidence["calibrated"] is False, "confidence must be marked uncalibrated")
        _require(confidence["region_specific"] is False, "confidence is not region-specific")
        key = (item_id, image_index)
        _require(
            global_index not in seen_global and key not in seen_keys, "duplicate dataset image"
        )
        seen_global.add(global_index)
        seen_keys.add(key)
        item_ids.add(item_id)
        _require(isinstance(record["regions"], list), "regions must be a list")
        _require(
            record["parse_status"]
            == ("regions_parsed" if record["regions"] else "no_location_tokens"),
            "parse_status disagrees with regions",
        )
        region_keys: set[tuple[str, tuple[tuple[int, int], ...]]] = set()
        for region_index, region in enumerate(record["regions"]):
            _require(isinstance(region, dict), "region is not an object")
            _exact_keys(
                region,
                {
                    "region_id",
                    "text",
                    "normalized_text",
                    "quadrilateral_normalized",
                    "confidence_ref",
                    "source_occurrences",
                    "technical_duplicate_count",
                },
                f"dataset[{line_number}].regions[{region_index}]",
            )
            quad = validate_quadrilateral(
                region.get("quadrilateral_normalized"),
                f"dataset[{line_number}].regions[{region_index}]",
            )
            _require(
                isinstance(region["text"], str) and region["text"].strip() == region["text"],
                "invalid region text",
            )
            _require(
                region["normalized_text"] == normalize_text(region["text"]),
                "invalid normalized region text",
            )
            _require(
                region.get("confidence_ref") == "generation_confidence",
                "invalid confidence reference",
            )
            occurrences = region.get("source_occurrences")
            _require(
                isinstance(occurrences, list) and bool(occurrences), "missing source occurrences"
            )
            _require(
                region.get("technical_duplicate_count") == len(occurrences) - 1,
                "wrong duplicate count",
            )
            detection_indices: list[int] = []
            for occurrence in occurrences:
                _require(isinstance(occurrence, dict), "source occurrence is not an object")
                _exact_keys(occurrence, {"detection_index", "text"}, "source occurrence")
                _require(
                    type(occurrence["detection_index"]) is int
                    and occurrence["detection_index"] >= 0,
                    "invalid source detection index",
                )
                _require(
                    isinstance(occurrence["text"], str) and bool(occurrence["text"].strip()),
                    "invalid source occurrence text",
                )
                _require(
                    normalize_text(occurrence["text"]) == region["normalized_text"],
                    "source occurrence text changed deduplication key",
                )
                detection_indices.append(occurrence["detection_index"])
            _require(
                detection_indices == sorted(set(detection_indices)),
                "source detection indices must be unique and sorted",
            )
            _require(
                region["region_id"] == f"ocr-{global_index:05d}-{detection_indices[0]:04d}",
                "region id does not point to its first source occurrence",
            )
            region_key = (str(region["normalized_text"]), quad)
            _require(
                region_key not in region_keys,
                "technical duplicate remained in canonical image record",
            )
            region_keys.add(region_key)
        rows += 1
    _require(rows == expected_images, f"dataset rows={rows}, expected={expected_images}")
    _require(
        len(item_ids) == expected_items, f"dataset items={len(item_ids)}, expected={expected_items}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and audit all exp633 OCR shards.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--shard-dir", type=Path, action="append", default=[])
    parser.add_argument("--repair-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--expected-items", type=int, default=EXPECTED_ITEMS, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-images", type=int, default=EXPECTED_IMAGES, help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.verify_only:
        if args.manifest is not None or args.shard_dir or args.repair_dir:
            raise ValueError("--verify-only accepts only --output-dir")
        result = verify_bundle(
            args.output_dir,
            expected_items=args.expected_items,
            expected_images=args.expected_images,
        )
    else:
        if args.manifest is None or not args.shard_dir:
            raise ValueError("build mode requires --manifest and at least one --shard-dir")
        result = build_dataset(
            manifest_path=args.manifest,
            shard_dirs=args.shard_dir,
            output_dir=args.output_dir,
            repair_dirs=args.repair_dir,
            expected_items=args.expected_items,
            expected_images=args.expected_images,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

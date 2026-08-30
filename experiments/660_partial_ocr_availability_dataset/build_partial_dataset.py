from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

EXPERIMENT_ID = "660"
DATASET_VERSION = "competition_train_paddleocr_vl16_partial_v1"
EXPECTED_ITEMS = 12_971
EXPECTED_IMAGES = 49_456
BUILDER_634 = Path(__file__).parents[1] / "634_immutable_ocr_dataset" / "build_dataset.py"


def load_strict_builder(path: Path = BUILDER_634) -> ModuleType:
    spec = importlib.util.spec_from_file_location("strict_ocr_builder_634", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load strict OCR builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_source(
    builder: ModuleType,
    source: dict[str, Any],
    manifest_row: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any] | None]:
    if source.get("error") is not None:
        return "OCR_UNAVAILABLE", ["source_processing_error"], None
    reasons = list(builder.repair_reasons(source))
    if reasons:
        return "OCR_UNAVAILABLE", sorted(set(reasons)), None
    try:
        record, _ = builder.canonicalize_success_record(source, manifest_row)
    except (KeyError, TypeError, ValueError):
        return "OCR_UNAVAILABLE", ["strict_schema_or_geometry_failure"], None
    record["dataset_version"] = DATASET_VERSION
    return "OCR_AVAILABLE", [], record


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unzip_member(archive: Path, member: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/unzip", "-p", str(archive), member],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def collect_archives(
    builder: ModuleType,
    archives: list[Path],
    *,
    manifest_rows: list[dict[str, Any]],
    manifest_sha256: str,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    if not archives:
        raise ValueError("at least one source archive is required")
    by_global_index: dict[int, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    seen_shards: set[int] = set()
    expected_num_shards: int | None = None
    for archive in archives:
        subprocess.run(
            ["/usr/bin/unzip", "-tq", str(archive)],
            check=True,
            capture_output=True,
        )
        report_payload = _unzip_member(archive, "report.json")
        report = json.loads(report_payload)
        if not isinstance(report, dict):
            raise ValueError("source report is not an object")
        builder._reject_forbidden_keys(report, "source_report")
        builder._exact_keys(report, builder.REPORT_KEYS, "source_report")
        if report["schema_version"] != 1:
            raise ValueError("unsupported source report schema")
        if report["experiment_id"] != "633":
            raise ValueError("wrong source experiment")
        if report["model_id"] != builder.MODEL_ID or report["model_revision"] != builder.MODEL_REVISION:
            raise ValueError("wrong source model identity")
        if report["manifest_sha256"] != manifest_sha256:
            raise ValueError("source manifest checksum mismatch")
        shard_index = int(report["shard_index"])
        num_shards = int(report["num_shards"])
        if not 0 <= shard_index < num_shards:
            raise ValueError("source shard index outside range")
        if expected_num_shards is None:
            expected_num_shards = num_shards
        if num_shards != expected_num_shards or shard_index in seen_shards:
            raise ValueError("duplicate or inconsistent source shard")
        seen_shards.add(shard_index)

        command = ["/usr/bin/unzip", "-p", str(archive), "spotting.jsonl"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdout is not None
        digest = hashlib.sha256()
        row_count = successful = failed = detections = parseable = 0
        for raw_line in process.stdout:
            digest.update(raw_line)
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError("source row is not an object")
            builder._reject_forbidden_keys(row, f"source[{shard_index}][{row_count}]")
            global_index = row.get("global_index")
            if type(global_index) is not int or not 0 <= global_index < len(manifest_rows):
                raise ValueError("invalid source global_index")
            if global_index % num_shards != shard_index or global_index in by_global_index:
                raise ValueError("source row has wrong or duplicate shard assignment")
            expected = manifest_rows[global_index]
            for key in ("id", "image_index", "global_index"):
                if row.get(key) != expected[key]:
                    raise ValueError(f"source identity mismatch: {key}")
            if row.get("error") is None:
                builder._exact_keys(row, builder.SUCCESS_KEYS, "source_success")
                successful += 1
                row_detections = row["detections"]
                if not isinstance(row_detections, list):
                    raise ValueError("source detections is not a list")
                detections += len(row_detections)
                parseable += builder.is_parseable_generation(row["raw_generation"], row_detections)
            else:
                builder._exact_keys(row, builder.ERROR_KEYS, "source_error")
                failed += 1
            by_global_index[global_index] = row
            row_count += 1
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"cannot stream source archive: exit={return_code}, stderr_bytes={len(stderr)}")
        output_sha256 = digest.hexdigest()
        if output_sha256 != report["output_sha256"]:
            raise ValueError("source spotting checksum mismatch")
        if row_count != report["requested_images"]:
            raise ValueError("source row count differs from report")
        if successful != report["successful_images"] or failed != report["failed_images"]:
            raise ValueError("source success counts differ from report")
        if detections != report["detections"] or parseable != report["parseable_images"]:
            raise ValueError("source parse counts differ from report")
        provenance.append(
            {
                "shard_index": shard_index,
                "num_shards": num_shards,
                "spotting_sha256": output_sha256,
                "report_sha256": _sha256_bytes(report_payload),
                "rows": row_count,
                "successful_images": successful,
                "failed_images": failed,
                "detections": detections,
                "archive_crc_passed": True,
            }
        )
    if expected_num_shards is None or seen_shards != set(range(expected_num_shards)):
        raise ValueError("source archive set is incomplete")
    if set(by_global_index) != set(range(len(manifest_rows))):
        raise ValueError("source archive rows do not cover the manifest")
    return by_global_index, sorted(provenance, key=lambda row: int(row["shard_index"]))


def build(
    *,
    manifest_path: Path,
    shard_dirs: list[Path] | None,
    shard_archives: list[Path] | None,
    output_dir: Path,
    expected_items: int = EXPECTED_ITEMS,
    expected_images: int = EXPECTED_IMAGES,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output_dir}")
    builder = load_strict_builder()
    manifest_sha256 = builder.sha256_file(manifest_path)
    manifest_rows = builder.load_manifest(
        manifest_path,
        expected_items=expected_items,
        expected_images=expected_images,
    )
    if bool(shard_dirs) == bool(shard_archives):
        raise ValueError("choose exactly one source mode: shard directories or shard archives")
    if shard_archives:
        source_rows, source_provenance = collect_archives(
            builder,
            list(shard_archives),
            manifest_rows=manifest_rows,
            manifest_sha256=manifest_sha256,
        )
    else:
        source_rows, source_provenance = builder.collect_shards(
            list(shard_dirs or []),
            manifest_rows=manifest_rows,
            manifest_sha256=manifest_sha256,
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    available_regions = 0
    item_total: Counter[str] = Counter()
    item_available: Counter[str] = Counter()
    try:
        availability_path = temporary / "ocr_availability.jsonl"
        available_path = temporary / "ocr_available_images.jsonl"
        with availability_path.open("w", encoding="utf-8") as availability_output, available_path.open(
            "w", encoding="utf-8"
        ) as available_output:
            for manifest_row in manifest_rows:
                global_index = int(manifest_row["global_index"])
                item_id = str(manifest_row["id"])
                status, reasons, record = classify_source(
                    builder, source_rows[global_index], manifest_row
                )
                status_counts[status] += 1
                reason_counts.update(reasons)
                item_total[item_id] += 1
                if record is not None:
                    item_available[item_id] += 1
                    available_regions += len(record["regions"])
                    available_output.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                availability = {
                    "schema_version": 1,
                    "dataset_version": DATASET_VERSION,
                    "id": item_id,
                    "image_index": int(manifest_row["image_index"]),
                    "global_index": global_index,
                    "source_url_sha256": builder.sha256_text(str(manifest_row["url"])),
                    "status": status,
                    "reasons": reasons,
                }
                availability_output.write(
                    json.dumps(
                        availability, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )

        if sum(status_counts.values()) != expected_images:
            raise ValueError("availability mask does not cover every image")
        if len(item_total) != expected_items:
            raise ValueError("availability mask does not cover every item")

        item_path = temporary / "ocr_item_availability.jsonl"
        all_images_available_items = 0
        no_images_available_items = 0
        with item_path.open("w", encoding="utf-8") as output:
            for item_id in sorted(item_total):
                available = item_available[item_id]
                total = item_total[item_id]
                all_images_available_items += available == total
                no_images_available_items += available == 0
                output.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "dataset_version": DATASET_VERSION,
                            "id": item_id,
                            "images": total,
                            "available_images": available,
                            "unavailable_images": total - available,
                            "has_any_ocr_available": available > 0,
                            "all_images_ocr_available": available == total,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        report = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "dataset_version": DATASET_VERSION,
            "status": "accepted",
            "source_experiment_id": "633",
            "source_manifest_sha256": manifest_sha256,
            "source_shards": len(source_provenance),
            "images": expected_images,
            "items": expected_items,
            "available_images": status_counts["OCR_AVAILABLE"],
            "unavailable_images": status_counts["OCR_UNAVAILABLE"],
            "availability_fraction": status_counts["OCR_AVAILABLE"] / expected_images,
            "available_regions": available_regions,
            "unavailable_reasons": dict(sorted(reason_counts.items())),
            "items_with_all_images_available": all_images_available_items,
            "items_with_no_images_available": no_images_available_items,
            "availability_sha256": builder.sha256_file(availability_path),
            "available_records_sha256": builder.sha256_file(available_path),
            "item_availability_sha256": builder.sha256_file(item_path),
            "source_provenance_sha256": builder.sha256_text(
                json.dumps(source_provenance, sort_keys=True, separators=(",", ":"))
            ),
            "privacy": {
                "labels_read": 0,
                "categories_read": 0,
                "folds_read": 0,
                "sealed_membership_used": False,
                "public_used": False,
            },
            "paddleocr_vl_submission_runtime": False,
            "decision": "ACCEPT_PARTIAL_FAIL_CLOSED_OCR_DATASET",
        }
        report_path = temporary / "aggregate_integrity.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_dir)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--manifest", type=Path, required=True)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--shard-dir", action="append", type=Path)
    source.add_argument("--shard-archive", action="append", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--expected-items", type=int, default=EXPECTED_ITEMS)
    result.add_argument("--expected-images", type=int, default=EXPECTED_IMAGES)
    return result


def main() -> None:
    args = parser().parse_args()
    report = build(
        manifest_path=args.manifest,
        shard_dirs=args.shard_dir,
        shard_archives=args.shard_archive,
        output_dir=args.output_dir,
        expected_items=args.expected_items,
        expected_images=args.expected_images,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

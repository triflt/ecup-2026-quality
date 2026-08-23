from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/634_immutable_ocr_dataset"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILD = load_module("exp634_build", EXP / "build_dataset.py")
COVERAGE = load_module("exp634_coverage", EXP / "evaluate_category_coverage.py")

LOC = "<|LOC_100|><|LOC_200|><|LOC_300|><|LOC_200|><|LOC_300|><|LOC_400|><|LOC_100|><|LOC_400|>"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detection(text: str = "Газ", confidence: float = 0.8) -> dict[str, Any]:
    return {
        "text": text,
        "polygon": [[20, 20], [60, 20], [60, 40], [20, 40]],
        "normalized_polygon": [[100, 200], [300, 200], [300, 400], [100, 400]],
        "confidence": confidence,
        "confidence_type": BUILD.CONFIDENCE_TYPE,
    }


def success_row(
    item_id: str,
    image_index: int,
    global_index: int,
    *,
    detections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    detections = [] if detections is None else detections
    return {
        "id": item_id,
        "image_index": image_index,
        "global_index": global_index,
        "width": 200,
        "height": 100,
        "raw_generation": "\n".join(f"{row['text']}{LOC}" for row in detections),
        "sequence_confidence": 0.8,
        "detections": detections,
        "error": None,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_inputs(tmp_path: Path) -> tuple[Path, list[Path]]:
    manifest = tmp_path / "all_images.jsonl"
    manifest_rows = [
        {"id": "a", "image_index": 0, "url": "https://example.invalid/a0"},
        {"id": "a", "image_index": 1, "url": "https://example.invalid/a1"},
        {"id": "b", "image_index": 0, "url": "https://example.invalid/b0"},
    ]
    write_jsonl(manifest, manifest_rows)
    rows_by_shard = {
        0: [
            success_row("a", 0, 0, detections=[detection("Газ"), detection(" ГАЗ ".strip())]),
            success_row("b", 0, 2, detections=[detection()]),
        ],
        1: [success_row("a", 1, 1)],
    }
    shard_dirs: list[Path] = []
    for shard_index, rows in rows_by_shard.items():
        shard_dir = tmp_path / f"shard-{shard_index}"
        shard_dir.mkdir()
        spotting = shard_dir / "spotting.jsonl"
        write_jsonl(spotting, rows)
        successful = len(rows)
        parseable = sum(
            bool(row["detections"]) or str(row["raw_generation"]).strip() in {"", "</s>"}
            for row in rows
        )
        report = {
            "schema_version": 1,
            "experiment_id": "633",
            "model_id": BUILD.MODEL_ID,
            "model_revision": BUILD.MODEL_REVISION,
            "manifest_sha256": sha256_file(manifest),
            "output_sha256": sha256_file(spotting),
            "shard_index": shard_index,
            "num_shards": 2,
            "requested_images": len(rows),
            "successful_images": successful,
            "failed_images": 0,
            "parseable_images": parseable,
            "parseable_fraction_of_successful": parseable / successful,
            "detections": sum(len(row["detections"]) for row in rows),
            "elapsed_seconds": 1.0,
            "images_per_second": float(len(rows)),
        }
        (shard_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        shard_dirs.append(shard_dir)
    return manifest, shard_dirs


def test_merge_is_complete_immutable_and_preserves_duplicate_provenance(tmp_path: Path) -> None:
    manifest, shards = make_inputs(tmp_path)
    output = tmp_path / "dataset"
    bundle = BUILD.build_dataset(
        manifest_path=manifest,
        shard_dirs=list(reversed(shards)),
        output_dir=output,
        expected_items=2,
        expected_images=3,
    )
    assert bundle["dataset_version"] == BUILD.DATASET_VERSION
    rows = [json.loads(line) for line in (output / "ocr_images.jsonl").read_text().splitlines()]
    assert [row["global_index"] for row in rows] == [0, 1, 2]
    assert rows[0]["id"] == "a" and rows[0]["image_index"] == 0
    assert len(rows[0]["regions"]) == 1
    region = rows[0]["regions"][0]
    assert region["technical_duplicate_count"] == 1
    assert region["source_occurrences"] == [
        {"detection_index": 0, "text": "Газ"},
        {"detection_index": 1, "text": "ГАЗ"},
    ]
    assert region["quadrilateral_normalized"] == [[100, 200], [300, 200], [300, 400], [100, 400]]
    assert rows[2]["regions"][0]["region_id"] != region["region_id"]
    assert rows[0]["generation_confidence"] == {
        "value": 0.8,
        "type": "sequence_geomean_token_probability",
        "scope": "complete_generated_sequence",
        "calibrated": False,
        "region_specific": False,
    }
    report = json.loads((output / "merge_report.json").read_text())
    assert report["scope"]["manifest_coverage"] == 1.0
    assert report["regions"]["technical_duplicates_merged"] == 1
    assert report["privacy"]["labels_read"] == 0
    with pytest.raises(FileExistsError, match="immutable"):
        BUILD.build_dataset(
            manifest_path=manifest,
            shard_dirs=shards,
            output_dir=output,
            expected_items=2,
            expected_images=3,
        )


def test_merge_rejects_out_of_bounds_box_and_label_fields(tmp_path: Path) -> None:
    manifest, shards = make_inputs(tmp_path)
    spotting = shards[0] / "spotting.jsonl"
    rows = [json.loads(line) for line in spotting.read_text().splitlines()]
    rows[0]["detections"][0]["normalized_polygon"][0][0] = 1001
    write_jsonl(spotting, rows)
    report_path = shards[0] / "report.json"
    report = json.loads(report_path.read_text())
    report["output_sha256"] = sha256_file(spotting)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="outside 0..1000"):
        BUILD.build_dataset(
            manifest_path=manifest,
            shard_dirs=shards,
            output_dir=tmp_path / "bad-box",
            expected_items=2,
            expected_images=3,
        )

    manifest_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    manifest_rows[0]["label"] = 1
    write_jsonl(manifest, manifest_rows)
    with pytest.raises(ValueError, match="forbidden"):
        BUILD.load_manifest(manifest, expected_items=2, expected_images=3)


def test_merge_rejects_missing_or_duplicate_shards_and_processing_errors(tmp_path: Path) -> None:
    manifest, shards = make_inputs(tmp_path)
    with pytest.raises(ValueError, match="incomplete shard"):
        BUILD.build_dataset(
            manifest_path=manifest,
            shard_dirs=[shards[0]],
            output_dir=tmp_path / "missing",
            expected_items=2,
            expected_images=3,
        )
    with pytest.raises(ValueError, match="duplicate shard"):
        BUILD.build_dataset(
            manifest_path=manifest,
            shard_dirs=[shards[0], shards[0]],
            output_dir=tmp_path / "duplicate",
            expected_items=2,
            expected_images=3,
        )

    spotting = shards[1] / "spotting.jsonl"
    error_row = {"id": "a", "image_index": 1, "global_index": 1, "error": "download failed"}
    write_jsonl(spotting, [error_row])
    report_path = shards[1] / "report.json"
    report = json.loads(report_path.read_text())
    report.update(
        {
            "output_sha256": sha256_file(spotting),
            "successful_images": 0,
            "failed_images": 1,
            "parseable_images": 0,
            "parseable_fraction_of_successful": 0.0,
            "detections": 0,
        }
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="processing error remains"):
        BUILD.build_dataset(
            manifest_path=manifest,
            shard_dirs=shards,
            output_dir=tmp_path / "error",
            expected_items=2,
            expected_images=3,
        )


def test_verify_detects_tampering(tmp_path: Path) -> None:
    manifest, shards = make_inputs(tmp_path)
    output = tmp_path / "dataset"
    BUILD.build_dataset(
        manifest_path=manifest,
        shard_dirs=shards,
        output_dir=output,
        expected_items=2,
        expected_images=3,
    )
    with (output / "ocr_images.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        BUILD.verify_bundle(output, expected_items=2, expected_images=3)


def test_category_coverage_is_external_aggregate_only(tmp_path: Path) -> None:
    manifest, shards = make_inputs(tmp_path)
    output = tmp_path / "dataset"
    BUILD.build_dataset(
        manifest_path=manifest,
        shard_dirs=shards,
        output_dir=output,
        expected_items=2,
        expected_images=3,
    )
    categories = tmp_path / "categories.csv"
    categories.write_text("id,category\na,БАД\nb,ЛВ\n", encoding="utf-8")
    report = COVERAGE.evaluate(output, categories, expected_items=2, expected_images=3)
    assert set(report["by_category"]) == {"БАД", "ЛВ"}
    assert report["by_category"]["БАД"]["images"] == 2
    assert report["by_category"]["ЛВ"]["images"] == 1
    assert report["report_scope"] == "aggregate_category_coverage_only"
    assert report["privacy"]["row_level_output"] is False
    assert "rows" not in report

    categories.write_text("id,category,label\na,БАД,1\nb,ЛВ,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly id,category"):
        COVERAGE.load_categories(categories)


def test_schema_and_registry_proposal_are_strict_and_promotion_is_gated() -> None:
    schema = json.loads((EXP / "ocr_image_record.schema.json").read_text())
    assert schema["additionalProperties"] is False
    assert schema["properties"]["generation_confidence"]["properties"]["region_specific"] == {
        "const": False
    }
    point = schema["$defs"]["point"]
    assert point["prefixItems"][0]["maximum"] == 1000
    assert point["prefixItems"][1]["minimum"] == 0

    proposal = tomllib.loads((EXP / "dataset_registry_proposal.toml").read_text())
    assert proposal["status"] == "proposal_pending_full_artifact"
    assert proposal["entry"]["version"] == BUILD.DATASET_VERSION
    assert proposal["entry"]["sha256_source"] == "dataset_manifest.json:dataset_sha256"
    assert proposal["entry"]["immutable"] is True
    assert proposal["entry"]["label_blind"] is True
    assert proposal["entry"]["contains_public_data"] is False
    assert proposal["entry"]["contains_sealed_membership"] is False
    assert proposal["promotion_requirements"]["manifest_coverage"] == 1.0
    assert proposal["promotion_requirements"]["processing_errors"] == 0

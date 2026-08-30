from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

MODULE_PATH = Path(__file__).with_name("build_dataset.py")
SPEC = importlib.util.spec_from_file_location("exp634_build_dataset", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load {MODULE_PATH}")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)

CATEGORY_COLUMNS = ["id", "category"]


def load_categories(path: Path) -> dict[str, str]:
    categories: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CATEGORY_COLUMNS:
            raise ValueError(
                "category manifest must contain exactly id,category; labels, folds, split, Public and sealed membership are forbidden"
            )
        for line_number, row in enumerate(reader, start=2):
            item_id = str(row["id"])
            category = str(row["category"])
            if not item_id or not category:
                raise ValueError(f"empty id/category at line {line_number}")
            if item_id in categories:
                raise ValueError(f"duplicate category id: {item_id}")
            categories[item_id] = category
    if not categories:
        raise ValueError("empty category manifest")
    return categories


def evaluate(
    dataset_dir: Path,
    category_manifest: Path,
    *,
    expected_items: int = BUILD.EXPECTED_ITEMS,
    expected_images: int = BUILD.EXPECTED_IMAGES,
) -> dict[str, Any]:
    bundle = BUILD.verify_bundle(
        dataset_dir,
        expected_items=expected_items,
        expected_images=expected_images,
    )
    categories = load_categories(category_manifest)
    data_path = dataset_dir / str(bundle["dataset_file"])
    observed_ids: set[str] = set()
    per_category: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "items": set(),
            "images": 0,
            "images_with_regions": 0,
            "images_without_location_tokens": 0,
            "regions": 0,
            "technical_duplicates_merged": 0,
            "null_generation_confidence": 0,
        }
    )
    for _, row in BUILD._iter_jsonl(data_path):
        item_id = str(row["id"])
        if item_id not in categories:
            raise ValueError(f"dataset id missing from category manifest: {item_id}")
        observed_ids.add(item_id)
        stats = per_category[categories[item_id]]
        stats["items"].add(item_id)
        stats["images"] += 1
        stats["images_with_regions"] += bool(row["regions"])
        stats["images_without_location_tokens"] += row["parse_status"] == "no_location_tokens"
        stats["regions"] += len(row["regions"])
        stats["technical_duplicates_merged"] += sum(
            int(region["technical_duplicate_count"]) for region in row["regions"]
        )
        stats["null_generation_confidence"] += row["generation_confidence"]["value"] is None
    extra = sorted(set(categories) - observed_ids)
    if extra:
        raise ValueError(
            f"category manifest has ids outside dataset: count={len(extra)}, first={extra[:5]}"
        )

    output_categories: dict[str, dict[str, Any]] = {}
    for category in sorted(per_category):
        source = per_category[category]
        images = int(source["images"])
        output_categories[category] = {
            "items": len(source["items"]),
            "images": images,
            "images_with_regions": int(source["images_with_regions"]),
            "image_region_coverage": float(source["images_with_regions"] / images),
            "images_without_location_tokens": int(source["images_without_location_tokens"]),
            "canonical_regions": int(source["regions"]),
            "mean_regions_per_image": float(source["regions"] / images),
            "technical_duplicates_merged": int(source["technical_duplicates_merged"]),
            "null_generation_confidence": int(source["null_generation_confidence"]),
        }
    return {
        "schema_version": 1,
        "experiment_id": "634",
        "dataset_version": BUILD.DATASET_VERSION,
        "dataset_sha256": bundle["dataset_sha256"],
        "report_scope": "aggregate_category_coverage_only",
        "by_category": output_categories,
        "privacy": {
            "labels_read": 0,
            "folds_read": 0,
            "sealed_membership_used": False,
            "public_used": False,
            "row_level_output": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate OCR coverage by category without labels."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--category-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite category coverage report")
    report = evaluate(args.dataset_dir, args.category_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

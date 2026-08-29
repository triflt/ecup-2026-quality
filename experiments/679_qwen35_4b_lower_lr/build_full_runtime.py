"""Build the single post-validation full-data runtime for experiment 679."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from grid_contract import clean_text

DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"
REGISTRY_SHA256 = "16b9c47999c6c1e97b1317182adc356931db60a1156ec237fa496fa48c5387ae"
SELECTOR_SHA256 = "5d7467c48fc8a5a73f947f5aa1300071c77ba699b29c12250caf1bcd3176d7ac"
IMAGE_MANIFEST_SHA256 = "d6193215ce2d6145440bd484ea225e77fe10784c4c2c06b7c6b0b85dc8efc7d9"
EXPECTED_ROWS = 12971
EXPECTED_TRAIN_OCCURRENCES = 5590
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def hard_random(
    indices: np.ndarray, scores: np.ndarray, count: int, rng: np.random.Generator
) -> list[int]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= count:
        return indices.tolist()
    hard_count = count // 2
    order = np.lexsort((indices, scores[indices]))
    hard = indices[order[:hard_count]]
    remaining = np.setdiff1d(indices, hard, assume_unique=False)
    random_part = rng.choice(remaining, size=count - hard_count, replace=False)
    return np.concatenate([hard, random_part]).tolist()


def select_full_training_indices(
    *, labels: np.ndarray, categories: np.ndarray, fused_scores: np.ndarray
) -> list[int]:
    threshold_map = {"БАД": 0.24864045896205267, "Легковоспламеняющиеся": 0.9591804083988902}
    thresholds = np.asarray([threshold_map[str(category)] for category in categories])
    uncertainty = np.abs(fused_scores.astype(np.float32) - thresholds)
    rng = np.random.default_rng(SEED)
    records: list[int] = []
    bad = np.flatnonzero(categories == "БАД")
    bad_pos = bad[labels[bad] == 1]
    bad_neg = bad[labels[bad] == 0]
    bad_count = min(1500, len(bad_neg), len(bad_pos))
    records.extend(hard_random(bad_pos, uncertainty, bad_count, rng))
    records.extend(hard_random(bad_neg, uncertainty, bad_count, rng))
    flammable = np.flatnonzero(categories == "Легковоспламеняющиеся")
    flammable_pos = flammable[labels[flammable] == 1]
    flammable_neg = flammable[labels[flammable] == 0]
    records.extend(np.repeat(flammable_pos, 5).tolist())
    records.extend(hard_random(flammable_neg, uncertainty, min(1600, len(flammable_neg)), rng))
    random.Random(SEED).shuffle(records)
    if len(records) != EXPECTED_TRAIN_OCCURRENCES:
        raise ValueError("full-data training occurrence count drifted")
    return records


def verify_full_gate(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not (
        report.get("experiment_id") == "679"
        and report.get("stage") == "full"
        and report.get("passed") is True
        and report.get("decision") == "ACCEPT_FOR_REFIT"
        and report.get("bad_route_byte_identical") is True
        and report.get("public_used") is False
        and report.get("sealed_rows") == 0
    ):
        raise ValueError("full production report does not authorize refit")
    return report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite full-data runtime")
    verify_full_gate(args.full_production_report)
    expected_hashes = {
        "data": (args.data, DATA_SHA256),
        "registry": (args.registry, REGISTRY_SHA256),
        "selector": (args.selector, SELECTOR_SHA256),
        "image_manifest": (args.image_manifest, IMAGE_MANIFEST_SHA256),
    }
    mismatch = {
        name: {"expected": expected, "actual": sha256_file(path)}
        for name, (path, expected) in expected_hashes.items()
        if sha256_file(path) != expected
    }
    if mismatch:
        raise ValueError(f"full-data frozen input checksum mismatch: {mismatch}")
    data = pd.read_csv(args.data, dtype={"id": str})
    registry = pd.read_csv(args.registry, dtype={"id": str, "semantic_component": str})
    if len(data) != EXPECTED_ROWS or len(registry) != EXPECTED_ROWS:
        raise ValueError("full-data row count mismatch")
    if data["id"].duplicated().any() or registry["id"].duplicated().any():
        raise ValueError("full-data IDs must be unique")
    registry = registry.set_index("id").loc[data["id"]].reset_index()
    if not np.array_equal(data["label"].astype(np.int8), registry["label"].astype(np.int8)):
        raise ValueError("data labels differ from immutable registry")
    with np.load(args.selector, allow_pickle=True) as selector:
        if selector["ids"].astype(str).tolist() != data["id"].astype(str).tolist():
            raise ValueError("selector IDs differ from full data")
        if not np.array_equal(selector["labels"].astype(np.int8), data["label"].astype(np.int8)):
            raise ValueError("selector labels differ from full data")
        if not np.array_equal(selector["categories"].astype(str), data["category"].astype(str)):
            raise ValueError("selector categories differ from full data")
        fused_scores = selector["fused_scores"].astype(np.float32)
    with gzip.open(args.image_manifest, "rt", encoding="utf-8", newline="") as stream:
        image_rows = list(csv.DictReader(stream, delimiter="\t"))
    if not image_rows or set(image_rows[0]) != {"id", "image_url"}:
        raise ValueError("full image manifest schema mismatch")
    image_by_id = {str(row["id"]): str(row["image_url"]) for row in image_rows}
    if len(image_by_id) != EXPECTED_ROWS or set(image_by_id) != set(data["id"]):
        raise ValueError("full image manifest coverage mismatch")
    labels = data["label"].to_numpy(np.int8)
    categories = data["category"].astype(str).to_numpy()
    selected = select_full_training_indices(
        labels=labels, categories=categories, fused_scores=fused_scores
    )
    common: dict[str, dict[str, Any]] = {}
    for global_index, (row, registry_row) in enumerate(
        zip(data.to_dict("records"), registry.to_dict("records"), strict=True)
    ):
        row_id = str(row["id"])
        common[row_id] = {
            "global_index": global_index,
            "id": row_id,
            "fold": -1,
            "semantic_component": str(registry_row["semantic_component"]),
            "category": str(row["category"]),
            "name": clean_text(row.get("name", ""), 320),
            "description": clean_text(row.get("description", ""), 1800),
            "image_url": image_by_id[row_id],
            "ocr_images": [],
        }
    train_rows: list[dict[str, Any]] = []
    for occurrence_index, row_index in enumerate(selected):
        source = data.iloc[row_index]
        train_rows.append(
            {
                **common[str(source["id"])],
                "occurrence_index": occurrence_index,
                "label": int(source["label"]),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    validation_path.write_text("", encoding="utf-8")
    split_counts = registry["split"].astype(str).value_counts().to_dict()
    report = {
        "schema_version": 1,
        "experiment_id": "679",
        "scope": "single_post_validation_full_competition_train_refit",
        "full_production_report_sha256": sha256_file(args.full_production_report),
        "rows": EXPECTED_ROWS,
        "train_occurrences": len(train_rows),
        "train_unique_ids": len({row["id"] for row in train_rows}),
        "validation_rows": 0,
        "internal_split_rows_used_for_training": split_counts,
        "internal_holdout_feedback_used_for_selection": False,
        "seed": SEED,
        "flammable_positive_repeat": 5,
        "bad_per_class_limit": 1500,
        "flammable_negative_limit": 1600,
        "selected_multiset_sha256": canonical_sha256(
            sorted(Counter(row["id"] for row in train_rows).items())
        ),
        "input_sha256": {
            name: expected for name, (_path, expected) in expected_hashes.items()
        },
        "output_sha256": {
            "train.jsonl": sha256_file(train_path),
            "validation.jsonl": sha256_file(validation_path),
        },
        "uses_27b": False,
        "public_used": False,
        "decision": "GO_SINGLE_REFIT",
    }
    report["contract_sha256"] = canonical_sha256(report)
    (args.output_dir / "runtime_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--full-production-report", type=Path, required=True)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--selector", type=Path, required=True)
    result.add_argument("--image-manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    print(json.dumps(build(parsed), ensure_ascii=False, indent=2, sort_keys=True))

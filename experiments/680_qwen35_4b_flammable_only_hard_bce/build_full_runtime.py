"""Build the single post-validation flammable-only full-data runtime."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT_PATH = ROOT / "experiments/679_qwen35_4b_lower_lr/build_full_runtime.py"
PARENT_SHA256 = "91d4d163cd25d948d820f270e8470835e0663c6fc5de78440d5e9874f69a7b6c"
FLAMMABLE = "Легковоспламеняющиеся"
EXPECTED_TRAIN_OCCURRENCES = 2590


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_parent() -> Any:
    if sha256_file(PARENT_PATH) != PARENT_SHA256:
        raise ValueError("frozen full-data parent builder checksum mismatch")
    spec = importlib.util.spec_from_file_location("exp679_full_for_680", PARENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen full-data parent builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_full_gate(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not (
        report.get("experiment_id") == "680"
        and report.get("stage") == "full"
        and (
            report.get("passed") is True
            or report.get("ablation_submission_eligible") is True
        )
        and report.get("decision") == "ACCEPT_FOR_REFIT"
        and report.get("public_used") is False
        and report.get("sealed_rows") == 0
        and (
            all(report.get("gates", {}).values())
            or all(report.get("ablation_gates", {}).values())
        )
        and report.get("gates", {}).get("bad_route_byte_identical") is True
        and set(report.get("fold_flammable_average_precision", {})) == {"0", "1", "2", "3", "4"}
    ):
        raise ValueError("full experiment-680 report does not authorize refit")
    return report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite full-data runtime")
    verify_full_gate(args.full_report)
    parent = load_parent()
    expected_hashes = {
        "data": (args.data, parent.DATA_SHA256),
        "registry": (args.registry, parent.REGISTRY_SHA256),
        "selector": (args.selector, parent.SELECTOR_SHA256),
        "image_manifest": (args.image_manifest, parent.IMAGE_MANIFEST_SHA256),
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
    if len(data) != parent.EXPECTED_ROWS or len(registry) != parent.EXPECTED_ROWS:
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
    if len(image_by_id) != parent.EXPECTED_ROWS or set(image_by_id) != set(data["id"]):
        raise ValueError("full image manifest coverage mismatch")

    labels = data["label"].to_numpy(np.int8)
    categories = data["category"].astype(str).to_numpy()
    parent_selected = parent.select_full_training_indices(
        labels=labels,
        categories=categories,
        fused_scores=fused_scores,
    )
    selected = [index for index in parent_selected if categories[index] == FLAMMABLE]
    if len(selected) != EXPECTED_TRAIN_OCCURRENCES:
        raise ValueError("flammable-only full occurrence count drifted")
    selected_labels = Counter(int(labels[index]) for index in selected)
    if selected_labels != Counter({0: 1600, 1: 990}):
        raise ValueError("flammable-only full class multiset drifted")

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
            "name": parent.clean_text(row.get("name", ""), 320),
            "description": parent.clean_text(row.get("description", ""), 1800),
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
    view_by_id = {str(row["id"]): row for row in train_rows}
    model_input_view = [
        {
            key: row[key]
            for key in (
                "global_index",
                "id",
                "fold",
                "category",
                "name",
                "description",
                "image_url",
            )
        }
        for row in sorted(view_by_id.values(), key=lambda item: int(item["global_index"]))
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    validation_path.write_text("", encoding="utf-8")
    report = {
        "schema_version": 1,
        "experiment_id": "680",
        "scope": "single_post_validation_flammable_only_full_competition_train_refit",
        "changed_factor": "remove_bad_training_occurrences",
        "source_full_builder_sha256": PARENT_SHA256,
        "full_report_sha256": sha256_file(args.full_report),
        "rows": parent.EXPECTED_ROWS,
        "train_occurrences": len(train_rows),
        "train_unique_ids": len({row["id"] for row in train_rows}),
        "validation_rows": 0,
        "seed": parent.SEED,
        "bad_train_occurrences": 0,
        "flammable_train_occurrences": len(train_rows),
        "flammable_positive_repeat": 5,
        "flammable_negative_limit": 1600,
        "model_input_view_sha256": parent.canonical_sha256(model_input_view),
        "selected_multiset_sha256": parent.canonical_sha256(
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
    report["contract_sha256"] = parent.canonical_sha256(report)
    (args.output_dir / "runtime_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--full-report", type=Path, required=True)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--selector", type=Path, required=True)
    result.add_argument("--image-manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    print(json.dumps(build(parsed), ensure_ascii=False, indent=2, sort_keys=True))

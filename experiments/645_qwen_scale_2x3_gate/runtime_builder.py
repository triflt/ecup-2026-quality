from __future__ import annotations

import csv
import gzip
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from grid_contract import (
    CELL_SPECS,
    CONCEPTS,
    DATA_VERSION,
    EVALUATION_VERSION,
    GRID_CONTRACT_SHA256,
    NO_EVIDENCE,
    canonical_sha256,
    clean_text,
    resolve_grounding,
    sha256_file,
)

EXPECTED_DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"
EXPECTED_FOLDS_SHA256 = "16b9c47999c6c1e97b1317182adc356931db60a1156ec237fa496fa48c5387ae"
EXPECTED_SELECTOR_SHA256 = "107ef2bc83d77bc674c0734e8c1b62fc323eba9f9731d69b5e2353796e97c3be"
EXPECTED_IMAGE_MANIFEST_SHA256 = "409e10a4035c5964a08bf2343603a1557148ec15e22674043c442a8c1bcea178"
EXPECTED_EVIDENCE_SHA256 = "bae8b85912a62778a785126a5d3077a6449084f7d3daa10814e86a7277c699a6"

CONCEPT_BY_EVIDENCE = {
    "BAD_EXPLICIT_MARKING": "OBJECT_OF_SALE",
    "BAD_EXPLICIT_SUPPLEMENT_MARKING": "OBJECT_OF_SALE",
    "BAD_EXPLICIT_NEGATION": "NEGATION",
    "FL_STANDALONE_IGNITION_SOURCE": "FUEL_OR_IGNITION",
    "FL_PYROTECHNIC_PRODUCT": "FUEL_OR_IGNITION",
    "FL_COMBUSTIBLE_PRODUCT": "FUEL_OR_IGNITION",
    "FL_INCLUDED_QUALIFYING_ITEM": "COMPLETENESS",
    "FL_EXPLICIT_HAZARD_MARKING": "COMPOSITION",
    "FL_FUEL_EXCLUDED": "NEGATION",
    "FL_EMPTY_CONTAINER": "COMPLETENESS",
    "FL_COMPATIBILITY_ONLY": "OBJECT_OF_SALE",
    "FL_INTEGRATED_IGNITION_ONLY": "COMPOSITION",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _hard_random(
    indices: np.ndarray,
    scores: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= count:
        return indices.tolist()
    hard_count = count // 2
    # The stable secondary index removes platform-dependent tied boundaries.
    order = np.lexsort((indices, scores[indices]))
    hard = indices[order[:hard_count]]
    remaining = np.setdiff1d(indices, hard, assume_unique=False)
    random_part = rng.choice(remaining, size=count - hard_count, replace=False)
    return np.concatenate([hard, random_part]).tolist()


def select_training_indices(
    *,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    fused_scores: np.ndarray,
    outer_fold: int,
    seed: int = 42,
) -> list[int]:
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be 0..4")
    threshold_map = {"БАД": 0.24864045896205267, "Легковоспламеняющиеся": 0.9591804083988902}
    thresholds = np.asarray([threshold_map[str(category)] for category in categories])
    uncertainty = np.abs(fused_scores.astype(np.float32) - thresholds)
    train_mask = folds.astype(np.int8) != outer_fold
    rng = np.random.default_rng(seed)
    records: list[int] = []

    bad = np.flatnonzero(train_mask & (categories == "БАД"))
    bad_pos = bad[labels[bad] == 1]
    bad_neg = bad[labels[bad] == 0]
    bad_count = min(1500, len(bad_neg), len(bad_pos))
    records.extend(_hard_random(bad_pos, uncertainty, bad_count, rng))
    records.extend(_hard_random(bad_neg, uncertainty, bad_count, rng))

    flammable = np.flatnonzero(train_mask & (categories == "Легковоспламеняющиеся"))
    flammable_pos = flammable[labels[flammable] == 1]
    flammable_neg = flammable[labels[flammable] == 0]
    records.extend(np.repeat(flammable_pos, 5).tolist())
    records.extend(_hard_random(flammable_neg, uncertainty, min(1600, len(flammable_neg)), rng))
    random.Random(seed).shuffle(records)
    return records


def load_ocr(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for path in paths:
        for row in _read_jsonl(path):
            row_id = str(row["id"])
            image_index = int(row["image_index"])
            key = (row_id, image_index)
            if key in seen:
                raise ValueError(f"duplicate OCR image record: {key}")
            seen.add(key)
            if row.get("error") is not None:
                detections: list[dict[str, Any]] = []
            else:
                detections = list(row.get("detections", []))
                for detection in detections:
                    if not str(detection.get("text", "")).strip():
                        raise ValueError("OCR detection has empty text")
                    polygon = detection.get("polygon")
                    if not isinstance(polygon, list) or len(polygon) != 4:
                        raise ValueError("OCR detection lacks a four-point polygon")
            by_id.setdefault(row_id, []).append(
                {"image_index": image_index, "detections": detections}
            )
    for images in by_id.values():
        images.sort(key=lambda item: int(item["image_index"]))
    return by_id


def _evidence_target(
    *,
    row: dict[str, Any],
    candidate: dict[str, Any] | None,
) -> dict[str, str]:
    fallback = {"quote": NO_EVIDENCE, "concept": NO_EVIDENCE}
    if not candidate:
        return fallback
    concept = CONCEPT_BY_EVIDENCE.get(str(candidate.get("concept", "")))
    if concept is None and candidate.get("concept") in CONCEPTS:
        concept = str(candidate["concept"])
    if concept is None:
        return fallback
    quote = str(candidate.get("exact_surface_span", candidate.get("quote", "")))
    if not quote or len(quote) > 240:
        return fallback
    grounding = resolve_grounding(row, quote)
    if not grounding["grounded"] or grounding["source"] == "none":
        return fallback
    requested_source = str(candidate.get("source", ""))
    if requested_source == "ocr" and grounding["source"] != "ocr":
        return fallback
    return {"quote": quote, "concept": concept}


def build_runtime(
    *,
    experiment_id: str,
    outer_fold: int,
    data_path: Path,
    folds_path: Path,
    selector_path: Path,
    image_manifest_path: Path,
    evidence_manifest_path: Path,
    ocr_paths: list[Path],
    output_dir: Path,
    expected_ocr_sha256: list[str] | None = None,
    enforce_frozen_hashes: bool = True,
) -> dict[str, Any]:
    spec = CELL_SPECS[experiment_id]
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be 0..4")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty runtime directory")
    frozen = {
        "data": (data_path, EXPECTED_DATA_SHA256),
        "folds": (folds_path, EXPECTED_FOLDS_SHA256),
        "selector": (selector_path, EXPECTED_SELECTOR_SHA256),
        "image_manifest": (image_manifest_path, EXPECTED_IMAGE_MANIFEST_SHA256),
        "evidence_manifest": (evidence_manifest_path, EXPECTED_EVIDENCE_SHA256),
    }
    if enforce_frozen_hashes:
        mismatch = {
            name: {"expected": expected, "actual": sha256_file(path)}
            for name, (path, expected) in frozen.items()
            if sha256_file(path) != expected
        }
        if mismatch:
            raise ValueError(f"frozen input checksum mismatch: {mismatch}")
    if spec.objective == "grounded_evidence" and not ocr_paths:
        raise ValueError("grounded-evidence cells require the completed OCR sidecar")
    if ocr_paths and expected_ocr_sha256 is None:
        raise ValueError("OCR shard checksums must be declared before runtime construction")
    actual_ocr_sha = [sha256_file(path) for path in ocr_paths]
    if expected_ocr_sha256 is not None and actual_ocr_sha != expected_ocr_sha256:
        raise ValueError("OCR shard checksums differ from the predeclared contract")

    data = pd.read_csv(data_path, dtype={"id": str})
    folds_frame = pd.read_csv(folds_path, dtype={"id": str, "semantic_component": str})
    required = {
        "id",
        "category",
        "label",
        "semantic_component",
        "split",
        "development_fold",
    }
    if not required.issubset(folds_frame.columns):
        raise ValueError("semantic-family registry schema mismatch")
    if data["id"].duplicated().any() or folds_frame["id"].duplicated().any():
        raise ValueError("duplicate IDs are forbidden")
    if set(data["id"]) != set(folds_frame["id"]):
        raise ValueError("data and semantic-family registry IDs differ")
    aligned = folds_frame.set_index("id").loc[data["id"]].reset_index()
    if not np.array_equal(data["label"].astype(np.int8), aligned["label"].astype(np.int8)):
        raise ValueError("data and semantic-family labels differ")
    development_mask = aligned["split"].astype(str).eq("development").to_numpy()
    development = data.loc[development_mask].copy().reset_index(drop=True)
    registry = aligned.loc[development_mask].copy().reset_index(drop=True)
    if len(development) != 11118:
        raise ValueError("semantic-family development scope must contain 11118 rows")
    if registry.groupby("semantic_component")["development_fold"].nunique().max() != 1:
        raise ValueError("semantic component crosses folds")

    with gzip.open(image_manifest_path, "rt", encoding="utf-8", newline="") as stream:
        image_rows = list(csv.DictReader(stream, delimiter="\t"))
    if not image_rows or set(image_rows[0]) != {"id", "image_url"}:
        raise ValueError("first-image manifest schema must be exactly id,image_url")
    image_by_id = {str(item["id"]): str(item["image_url"]) for item in image_rows}
    if len(image_by_id) != len(image_rows) or set(image_by_id) != set(development["id"]):
        raise ValueError("first-image manifest must exactly cover development")

    evidence_rows = _read_jsonl(evidence_manifest_path)
    evidence_by_id = {str(item["id"]): item for item in evidence_rows}
    if len(evidence_by_id) != len(evidence_rows) or set(evidence_by_id) != set(development["id"]):
        raise ValueError("evidence manifest must exactly cover development")
    if any(set(item) & {"label", "target", "gold", "y_true"} for item in evidence_rows):
        raise ValueError("evidence manifest itself must remain label-free")
    ocr_by_id = load_ocr(ocr_paths)
    unknown_ocr_ids = set(ocr_by_id) - set(development["id"])
    if unknown_ocr_ids:
        raise ValueError("OCR contains IDs outside development")
    if spec.objective == "grounded_evidence" and set(ocr_by_id) != set(development["id"]):
        raise ValueError("grounded-evidence OCR must cover every development item")

    selector = np.load(selector_path, allow_pickle=False)
    if not {"ids", "fold_ids"}.issubset(selector.files):
        raise ValueError("selector lacks IDs/folds")
    ids = development["id"].astype(str).tolist()
    if selector["ids"].astype(str).tolist() != ids:
        raise ValueError("selector IDs/order differ from development")
    local_folds = registry["development_fold"].to_numpy(np.int8)
    if not np.array_equal(selector["fold_ids"].astype(np.int8), local_folds):
        raise ValueError("selector folds differ from registry")
    score_key = "fused_scores" if "fused_scores" in selector.files else "fused"
    if score_key not in selector.files:
        raise ValueError("selector lacks fused donor-only scores")
    selected = select_training_indices(
        labels=development["label"].to_numpy(np.int8),
        categories=development["category"].astype(str).to_numpy(),
        folds=local_folds,
        fused_scores=selector[score_key].astype(np.float32),
        outer_fold=outer_fold,
    )

    common_rows_unsorted: list[dict[str, Any]] = []
    for row, registry_row in zip(
        development.to_dict("records"), registry.to_dict("records"), strict=True
    ):
        row_id = str(row["id"])
        common_rows_unsorted.append(
            {
                "id": row_id,
                "fold": int(registry_row["development_fold"]),
                "semantic_component": str(registry_row["semantic_component"]),
                "category": str(row["category"]),
                "name": clean_text(row.get("name", ""), 320),
                "description": clean_text(row.get("description", ""), 1800),
                "image_url": image_by_id[row_id],
                "ocr_images": ocr_by_id.get(row_id, []),
            }
        )
    common_rows = sorted(common_rows_unsorted, key=lambda item: str(item["id"]))
    for global_index, row in enumerate(common_rows):
        row["global_index"] = global_index
    common_by_id = {str(row["id"]): row for row in common_rows}

    train_rows: list[dict[str, Any]] = []
    for occurrence_index, row_index in enumerate(selected):
        selected_id = str(development.iloc[row_index]["id"])
        row = common_by_id[selected_id]
        if row["fold"] == outer_fold:
            raise ValueError("outer validation entered selected training")
        label = int(development.iloc[row_index]["label"])
        record = {**row, "occurrence_index": occurrence_index, "label": label}
        if spec.objective == "grounded_evidence":
            candidate = evidence_by_id[row["id"]].get(f"candidate_for_{label}")
            if candidate is not None and candidate.get("verdict") != label:
                raise ValueError("evidence candidate verdict differs from donor label")
            record["evidence_target"] = _evidence_target(row=row, candidate=candidate)
        train_rows.append(record)
    validation_rows = [row for row in common_rows if row["fold"] == outer_fold]
    if any("label" in row or "evidence_target" in row for row in validation_rows):
        raise ValueError("validation runtime contains supervision")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)
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
        for row in common_rows
    ]
    report = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "objective": spec.objective,
        "data_version": DATA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "grid_contract_sha256": GRID_CONTRACT_SHA256,
        "outer_fold": outer_fold,
        "train_occurrences": len(train_rows),
        "train_unique_ids": len({row["id"] for row in train_rows}),
        "validation_rows": len(validation_rows),
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "outer_validation_occurrences": 0,
        "model_input_fields": [
            "category",
            "name",
            "description",
            "first_image",
        ],
        "model_input_view_sha256": canonical_sha256(model_input_view),
        "selected_multiset_sha256": canonical_sha256(
            sorted(Counter(row["id"] for row in train_rows).items())
        ),
        "input_sha256": {name: sha256_file(path) for name, (path, _expected) in frozen.items()}
        | {"ocr_shards": actual_ocr_sha},
        "output_sha256": {
            "train.jsonl": sha256_file(train_path),
            "validation.jsonl": sha256_file(validation_path),
        },
        "decision": "GO",
    }
    report["contract_sha256"] = canonical_sha256(report)
    (output_dir / "runtime_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

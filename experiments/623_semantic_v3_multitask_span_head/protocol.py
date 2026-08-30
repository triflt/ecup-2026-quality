from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEVELOPMENT_FOLDS = (0, 1, 2, 3, 4)
CONCEPTS = (
    "OBJECT_OF_SALE",
    "COMPOSITION",
    "COMPLETENESS",
    "FUEL_OR_IGNITION",
    "NEGATION",
)
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
FORBIDDEN_LABEL_COLUMNS = {"label", "target", "gold", "y_true"}
FROZEN_INPUT_SHA256 = {
    "data": "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510",
    "folds": "16b9c47999c6c1e97b1317182adc356931db60a1156ec237fa496fa48c5387ae",
    "evidence_manifest": "bae8b85912a62778a785126a5d3077a6449084f7d3daa10814e86a7277c699a6",
    "selector": "107ef2bc83d77bc674c0734e8c1b62fc323eba9f9731d69b5e2353796e97c3be",
    "image_manifest": "409e10a4035c5964a08bf2343603a1557148ec15e22674043c442a8c1bcea178",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_text(name: object, description: object) -> tuple[str, dict[str, int]]:
    name_text = "" if pd.isna(name) else str(name)
    description_text = "" if pd.isna(description) else str(description)
    name_prefix = "Название: "
    description_prefix = "\nОписание: "
    text = name_prefix + name_text + description_prefix + description_text
    return text, {
        "name": len(name_prefix),
        "description": len(name_prefix) + len(name_text) + len(description_prefix),
    }


def rationale_target(row: dict[str, Any], candidate: dict[str, Any] | None) -> dict[str, Any]:
    text, bases = canonical_text(row.get("name", ""), row.get("description", ""))
    no_evidence = {
        "has_evidence": False,
        "char_start": -1,
        "char_end": -1,
        "exact_span": "",
        "concept": None,
        "quality_weight": 0.0,
    }
    if not candidate:
        return no_evidence
    source = str(candidate.get("source", ""))
    if source not in bases:
        return no_evidence
    concept = CONCEPT_BY_EVIDENCE.get(str(candidate.get("concept", "")))
    if concept is None:
        return no_evidence
    try:
        local_start = int(candidate["surface_start"])
        local_end = int(candidate["surface_end"])
    except (KeyError, TypeError, ValueError):
        return no_evidence
    start, end = bases[source] + local_start, bases[source] + local_end
    exact = str(candidate.get("exact_surface_span", ""))
    if not (0 <= start < end <= len(text)) or text[start:end] != exact:
        return no_evidence
    return {
        "has_evidence": True,
        "char_start": start,
        "char_end": end,
        "exact_span": exact,
        "concept": concept,
        "quality_weight": 1.0,
    }


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if set(item) & FORBIDDEN_LABEL_COLUMNS:
                raise ValueError("evidence manifest must remain label-free")
            row_id = str(item["id"])
            if row_id in rows:
                raise ValueError("evidence manifest contains duplicate ids")
            rows[row_id] = item
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_fold_runtime(
    *,
    data_path: Path,
    folds_path: Path,
    evidence_manifest_path: Path,
    selector_path: Path,
    image_manifest_path: Path,
    outer_fold: int,
    output_dir: Path,
    enforce_frozen_hashes: bool = True,
) -> dict[str, Any]:
    if outer_fold not in DEVELOPMENT_FOLDS:
        raise ValueError("outer_fold must be 0..4")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty output directory")
    inputs = {
        "data": data_path,
        "folds": folds_path,
        "evidence_manifest": evidence_manifest_path,
        "selector": selector_path,
        "image_manifest": image_manifest_path,
    }
    if enforce_frozen_hashes:
        mismatches = {
            name: {"expected": FROZEN_INPUT_SHA256[name], "actual": sha256_file(path)}
            for name, path in inputs.items()
            if sha256_file(path) != FROZEN_INPUT_SHA256[name]
        }
        if mismatches:
            raise ValueError(f"frozen parent/runtime input checksum mismatch: {mismatches}")
    data = pd.read_csv(data_path, dtype={"id": str})
    folds = pd.read_csv(folds_path, dtype={"id": str, "semantic_component": str})
    required_fold_columns = {
        "id",
        "category",
        "label",
        "semantic_component",
        "split",
        "development_fold",
    }
    if not required_fold_columns.issubset(folds.columns):
        raise ValueError("fold registry schema mismatch")
    if data["id"].duplicated().any() or folds["id"].duplicated().any():
        raise ValueError("duplicate ids are forbidden")
    if set(data["id"]) != set(folds["id"]):
        raise ValueError("data and fold registry ids differ")
    aligned = folds.set_index("id").loc[data["id"]]
    if (
        not data["label"]
        .astype(int)
        .reset_index(drop=True)
        .equals(aligned["label"].astype(int).reset_index(drop=True))
    ):
        raise ValueError("data and frozen registry labels differ")
    development_mask = aligned["split"].eq("development").to_numpy()
    development = data.loc[development_mask].copy().reset_index(drop=True)
    registry = aligned.loc[development_mask].reset_index(drop=True)
    local_folds = registry["development_fold"].astype(int)
    if set(local_folds) != set(DEVELOPMENT_FOLDS):
        raise ValueError("development folds must be exactly 0..4")
    if registry.groupby("semantic_component")["development_fold"].nunique().max() != 1:
        raise ValueError("semantic component crosses development folds")
    evidence = _load_manifest(evidence_manifest_path)
    if set(evidence) != set(development["id"].astype(str)):
        raise ValueError("evidence manifest must contain exactly development ids")
    registry_by_id = {
        str(row["id"]): row
        for row in registry.assign(id=development["id"].astype(str)).to_dict("records")
    }
    for row_id, manifest_row in evidence.items():
        registry_row = registry_by_id[row_id]
        expected_metadata = {
            "category": str(registry_row["category"]),
            "development_fold": int(registry_row["development_fold"]),
            "semantic_component": str(registry_row["semantic_component"]),
        }
        mismatches = {
            key: {"expected": expected, "actual": manifest_row.get(key)}
            for key, expected in expected_metadata.items()
            if manifest_row.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"evidence manifest metadata mismatch for {row_id}: {mismatches}")

    selector = np.load(selector_path, allow_pickle=False)
    required_selector = {"ids", "fold_ids"}
    if not required_selector.issubset(selector.files):
        raise ValueError("selector lacks ids or fold_ids")
    selector_ids = selector["ids"].astype(str).tolist()
    development_ids = development["id"].astype(str).tolist()
    if selector_ids != development_ids:
        raise ValueError("selector ids/order differ from the physical development data")
    if not np.array_equal(selector["fold_ids"].astype(np.int8), local_folds.to_numpy(np.int8)):
        raise ValueError("selector folds differ from the frozen registry")
    if "fused_scores" not in selector.files and "fused" not in selector.files:
        raise ValueError("selector lacks frozen donor-only scores")
    for key in selector.files:
        value = selector[key]
        if value.ndim and value.shape[0] != len(development):
            raise ValueError(f"selector array {key} is not development-scoped")

    with gzip.open(image_manifest_path, "rt", encoding="utf-8", newline="") as stream:
        image_rows = list(csv.DictReader(stream, delimiter="\t"))
    if not image_rows or set(image_rows[0]) != {"id", "image_url"}:
        raise ValueError("image manifest schema must be exactly id,image_url")
    image_ids = [str(row["id"]) for row in image_rows]
    if len(image_ids) != len(set(image_ids)) or set(image_ids) != set(development_ids):
        raise ValueError("image manifest must contain exactly the development ids")
    image_by_id = {str(row["id"]): str(row["image_url"]) for row in image_rows}
    if any(not image_by_id[item_id].strip() for item_id in development_ids):
        raise ValueError("image manifest contains an empty URL")

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for row_index, (row, fold) in enumerate(
        zip(development.to_dict("records"), local_folds, strict=True)
    ):
        row_id = str(row["id"])
        common = {
            "id": row_id,
            "category": str(row["category"]),
            "name": "" if pd.isna(row.get("name")) else str(row.get("name", "")),
            "description": ""
            if pd.isna(row.get("description"))
            else str(row.get("description", "")),
            "development_fold": int(fold),
            "row_index": row_index,
        }
        if int(fold) == outer_fold:
            validation_rows.append(common)
            continue
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("labels must be binary")
        candidate = evidence[row_id].get(f"candidate_for_{label}")
        if candidate is not None and candidate.get("verdict") != label:
            raise ValueError("evidence candidate verdict differs from donor label")
        train_rows.append({**common, "label": label, "rationale": rationale_target(row, candidate)})

    train_ids = {row["id"] for row in train_rows}
    validation_ids = {row["id"] for row in validation_rows}
    if train_ids & validation_ids:
        raise ValueError("outer validation entered training")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    selector_output_path = output_dir / "development_selector_oof.npz"
    image_output_path = output_dir / "development_image_manifest.tsv.gz"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)
    shutil.copyfile(selector_path, selector_output_path)
    shutil.copyfile(image_manifest_path, image_output_path)
    validation_keys = set().union(*(row.keys() for row in validation_rows))
    if validation_keys & FORBIDDEN_LABEL_COLUMNS or "rationale" in validation_keys:
        raise ValueError("validation runtime contains supervision")
    report = {
        "experiment_id": "623",
        "outer_fold": outer_fold,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_evidence_rows": sum(row["rationale"]["has_evidence"] for row in train_rows),
        "validation_label_columns": [],
        "validation_rationale_columns": [],
        "train_validation_overlap": 0,
        "sealed_rows_written": 0,
        "input_sha256": {
            "data": sha256_file(data_path),
            "folds": sha256_file(folds_path),
            "evidence_manifest": sha256_file(evidence_manifest_path),
            "selector": sha256_file(selector_path),
            "image_manifest": sha256_file(image_manifest_path),
        },
        "output_sha256": {
            "train.jsonl": sha256_file(train_path),
            "validation.jsonl": sha256_file(validation_path),
            "development_selector_oof.npz": sha256_file(selector_output_path),
            "development_image_manifest.tsv.gz": sha256_file(image_output_path),
        },
        "decision": "GO",
        "frozen_input_hashes_enforced": enforce_frozen_hashes,
    }
    (output_dir / "runtime_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def validate_label_free_predictions(
    *, prediction_path: Path,
    validation_path: Path,
    outer_fold: int,
) -> dict[str, Any]:
    validation = read_jsonl(validation_path)
    expected_ids = [str(row["id"]) for row in validation]
    expected_categories = [str(row["category"]) for row in validation]
    predictions = pd.read_csv(prediction_path, dtype={"id": str})
    expected_columns = {
        "id", "category", "fold", "lora_score", "verdict_probability", "evidence",
        "concept", "explanation", "char_start", "char_end",
    }
    if set(predictions.columns) != expected_columns:
        raise ValueError("prediction schema mismatch or supervision leakage")
    if predictions["id"].astype(str).tolist() != expected_ids:
        raise ValueError("prediction ids/order differ from label-free validation")
    if predictions["category"].astype(str).tolist() != expected_categories:
        raise ValueError("prediction categories differ from label-free validation")
    if not predictions["fold"].astype(int).eq(outer_fold).all():
        raise ValueError("prediction fold mismatch")
    raw_scores = pd.to_numeric(predictions["lora_score"], errors="coerce").to_numpy()
    probabilities = pd.to_numeric(
        predictions["verdict_probability"], errors="coerce"
    ).to_numpy()
    if not np.isfinite(raw_scores).all():
        raise ValueError("raw digit-logit scores must be finite")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("verdict probabilities must be finite and in [0, 1]")
    row_by_id = {str(row["id"]): row for row in validation}
    for prediction in predictions.to_dict("records"):
        evidence = str(prediction["evidence"])
        if evidence == "NO_EVIDENCE":
            if (
                str(prediction["concept"]) != "NO_EVIDENCE"
                or str(prediction["explanation"]) != "NO_EVIDENCE"
                or int(prediction["char_start"]) != -1
                or int(prediction["char_end"]) != -1
            ):
                raise ValueError("NO_EVIDENCE prediction has an evidence payload")
            continue
        source = row_by_id[str(prediction["id"])]
        text, _ = canonical_text(source["name"], source["description"])
        start, end = int(prediction["char_start"]), int(prediction["char_end"])
        if not (0 <= start < end <= len(text)) or text[start:end] != evidence:
            raise ValueError("prediction evidence is not the declared exact substring")
        if str(prediction["concept"]) not in CONCEPTS:
            raise ValueError("prediction concept is outside the closed ontology")
        if evidence not in str(prediction["explanation"]):
            raise ValueError("closed explanation omits its exact evidence")
    return {
        "rows": len(predictions),
        "prediction_ids_sha256": hashlib.sha256(
            json.dumps(expected_ids, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        "predictions_sha256": sha256_file(prediction_path),
        "validation_labels_read": 0,
        "decision": "GO",
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]

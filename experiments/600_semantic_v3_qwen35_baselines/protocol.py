from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_ID = "600"
PROTOCOL_VERSION = "semantic_family_v3"
SELECTOR_SOURCE_PROTOCOL = "semantic_v3_robust_base_strict_nested_v2"
FOLDS_SHA256 = "16b9c47999c6c1e97b1317182adc356931db60a1156ec237fa496fa48c5387ae"
DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"
IMAGE_MANIFEST_SHA256 = "d6193215ce2d6145440bd484ea225e77fe10784c4c2c06b7c6b0b85dc8efc7d9"
DEVELOPMENT_ROWS = 11_118
SEALED_ROWS = 1_853
DEVELOPMENT_IDS_SHA256 = "4fd6adf6a20d782972fa63b3569bf13a7e36e5bf282b2a7bd351a0230ccffa8e"
DEVELOPMENT_FOLDS = (0, 1, 2, 3, 4)
COMPONENTS = ("original", "specialist")
EXPECTED_TRAINING_MODE = "hard"
EXPECTED_SEED = 42
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
EPOCHS = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def id_sequence_sha256(values: list[str]) -> str:
    return canonical_sha256([str(value) for value in values])


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def read_folds(path: Path, *, enforce_frozen_hash: bool = True) -> pd.DataFrame:
    if enforce_frozen_hash and sha256_file(path) != FOLDS_SHA256:
        raise ValueError("semantic-family-v3 folds checksum mismatch")
    frame = pd.read_csv(path, dtype={"id": str, "semantic_component": str})
    expected_columns = {
        "id",
        "category",
        "label",
        "semantic_component",
        "component_size",
        "split",
        "development_fold",
    }
    if set(frame.columns) != expected_columns:
        raise ValueError("semantic-family-v3 folds schema mismatch")
    if frame["id"].duplicated().any():
        raise ValueError("semantic-family-v3 folds contain duplicate ids")
    split_values = set(frame["split"].astype(str))
    if split_values != {"development", "sealed_holdout"}:
        raise ValueError(f"unexpected split values: {sorted(split_values)}")
    development = frame["split"].eq("development")
    local_folds = frame.loc[development, "development_fold"].astype(int)
    if set(local_folds) != set(DEVELOPMENT_FOLDS):
        raise ValueError("development folds must be exactly 0..4")
    sealed_folds = frame.loc[~development, "development_fold"].astype(int)
    if not sealed_folds.eq(-1).all():
        raise ValueError("sealed rows must have development_fold=-1")
    if enforce_frozen_hash and (int(development.sum()), int((~development).sum())) != (
        DEVELOPMENT_ROWS,
        SEALED_ROWS,
    ):
        raise ValueError("semantic-family-v3 split counts mismatch")
    if frame.groupby("semantic_component")["split"].nunique().max() != 1:
        raise ValueError("a semantic component crosses development and sealed splits")
    if (
        frame.loc[development].groupby("semantic_component")["development_fold"].nunique().max()
        != 1
    ):
        raise ValueError("a semantic component crosses development folds")
    return frame


def _validate_selector_provenance(
    path: Path,
    *,
    expected_ids: list[str],
    selector_path: Path,
) -> dict[str, Any]:
    provenance = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": "development_only",
        "rows": len(expected_ids),
        "ids_sha256": id_sequence_sha256(expected_ids),
        "sealed_rows_used_for_fit": False,
        "sealed_labels_used": False,
        "sealed_rows_used_for_selection": False,
        "sealed_rows_used_for_thresholds": False,
        "selector_source_protocol": SELECTOR_SOURCE_PROTOCOL,
    }
    mismatches = {
        key: {"expected": value, "actual": provenance.get(key)}
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(f"development-only selector provenance mismatch: {mismatches}")
    if provenance.get("selector_npz_sha256") != sha256_file(selector_path):
        raise ValueError("selector NPZ checksum differs from provenance")
    if not provenance.get("frozen_before_qwen_training"):
        raise ValueError("selector scores were not frozen before Qwen training")
    return provenance


def _load_development_selector(
    path: Path,
    *,
    expected_ids: list[str],
    expected_folds: np.ndarray,
) -> dict[str, np.ndarray]:
    source = np.load(path, allow_pickle=True)
    required = {"ids", "fold_ids"}
    if not required.issubset(source.files):
        raise ValueError("selector NPZ lacks ids or fold_ids")
    if "fused_scores" not in source.files and "fused" not in source.files:
        raise ValueError("selector NPZ lacks frozen fused scores")
    ids = source["ids"].astype(str)
    if not np.array_equal(ids, np.asarray(expected_ids, dtype=str)):
        raise ValueError("selector NPZ must contain exactly development ids in data order")
    fold_ids = source["fold_ids"].astype(np.int8)
    if not np.array_equal(fold_ids, expected_folds.astype(np.int8)):
        raise ValueError("selector NPZ folds differ from semantic-family-v3 development folds")
    arrays: dict[str, np.ndarray] = {}
    for key in source.files:
        value = source[key]
        if value.ndim and value.shape[0] != len(expected_ids):
            raise ValueError(f"selector array {key} has a non-development leading dimension")
        arrays[key] = value
    return arrays


def prepare_development_inputs(
    *,
    data_path: Path,
    folds_path: Path,
    selector_path: Path,
    selector_provenance_path: Path,
    output_dir: Path,
    component: str,
    outer_fold: int,
    enforce_frozen_hash: bool = True,
) -> dict[str, Any]:
    if component not in COMPONENTS:
        raise ValueError(f"unknown component: {component}")
    if outer_fold not in DEVELOPMENT_FOLDS:
        raise ValueError(f"outer fold must be one of {DEVELOPMENT_FOLDS}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite protocol input directory")

    if enforce_frozen_hash and sha256_file(data_path) != DATA_SHA256:
        raise ValueError("source data checksum mismatch")
    data = pd.read_csv(data_path, dtype={"id": str})
    folds = read_folds(folds_path, enforce_frozen_hash=enforce_frozen_hash)
    if data["id"].duplicated().any():
        raise ValueError("source data contains duplicate ids")
    if set(data["id"]) != set(folds["id"]):
        raise ValueError("source data ids differ from semantic-family-v3 folds")
    fold_by_id = folds.set_index("id")
    aligned = fold_by_id.loc[data["id"]]
    if not np.array_equal(data["category"].astype(str), aligned["category"].astype(str)):
        raise ValueError("data categories differ from frozen folds")
    if not np.array_equal(data["label"].astype(int), aligned["label"].astype(int)):
        raise ValueError("data labels differ from frozen folds")

    development_mask = aligned["split"].eq("development").to_numpy()
    development = data.loc[development_mask].reset_index(drop=True)
    development_ids = development["id"].astype(str).tolist()
    development_folds = aligned.loc[development_mask, "development_fold"].to_numpy(dtype=np.int8)
    sealed_ids = set(aligned.index[~development_mask].astype(str))
    if sealed_ids.intersection(development_ids):
        raise ValueError("sealed ids survived physical development filtering")
    if enforce_frozen_hash and len(development) != DEVELOPMENT_ROWS:
        raise ValueError("physical development dataset has an unexpected row count")

    provenance = _validate_selector_provenance(
        selector_provenance_path,
        expected_ids=development_ids,
        selector_path=selector_path,
    )
    selector_arrays = _load_development_selector(
        selector_path,
        expected_ids=development_ids,
        expected_folds=development_folds,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_data_path = output_dir / "development_data.csv"
    filtered_selector_path = output_dir / "development_selector_oof.npz"
    mapping_path = output_dir / "id_mapping.csv"
    audit_path = output_dir / "protocol_input_audit.json"
    development.to_csv(filtered_data_path, index=False)
    selector_arrays = dict(selector_arrays)
    selector_arrays["ids"] = np.asarray(development_ids)
    selector_arrays["fold_ids"] = development_folds
    np.savez_compressed(filtered_selector_path, **selector_arrays)

    original_positions = np.flatnonzero(development_mask)
    with mapping_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "local_index",
                "id",
                "original_index",
                "development_fold",
                "outer_role",
            ],
        )
        writer.writeheader()
        for local_index, (item_id, original_index, fold) in enumerate(
            zip(development_ids, original_positions, development_folds, strict=True)
        ):
            writer.writerow(
                {
                    "local_index": local_index,
                    "id": item_id,
                    "original_index": int(original_index),
                    "development_fold": int(fold),
                    "outer_role": "validation" if int(fold) == outer_fold else "train",
                }
            )

    validation_mask = development_folds == outer_fold
    audit = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "component": component,
        "outer_fold": outer_fold,
        "source_rows": len(data),
        "development_rows": len(development),
        "outer_train_rows": int((~validation_mask).sum()),
        "outer_validation_rows": int(validation_mask.sum()),
        "sealed_rows_in_source": len(sealed_ids),
        "sealed_rows_in_physical_dataset": 0,
        "sealed_rows_in_selector_npz": 0,
        "sealed_rows_in_mapping": 0,
        "development_ids_sha256": id_sequence_sha256(development_ids),
        "outer_train_ids_sha256": id_sequence_sha256(
            [
                item_id
                for item_id, keep in zip(development_ids, ~validation_mask, strict=True)
                if keep
            ]
        ),
        "outer_validation_ids_sha256": id_sequence_sha256(
            [
                item_id
                for item_id, keep in zip(development_ids, validation_mask, strict=True)
                if keep
            ]
        ),
        "folds_sha256": sha256_file(folds_path),
        "data_sha256": sha256_file(data_path),
        "selector_source_sha256": sha256_file(selector_path),
        "selector_provenance_sha256": sha256_file(selector_provenance_path),
        "selector_provenance_audit_sha256": canonical_sha256(provenance),
        "physical_data_sha256": sha256_file(filtered_data_path),
        "physical_selector_sha256": sha256_file(filtered_selector_path),
        "mapping_sha256": sha256_file(mapping_path),
        "sealed_holdout_used_for_train": False,
        "sealed_holdout_used_for_selection": False,
        "sealed_holdout_used_for_threshold": False,
        "sealed_holdout_used_for_evaluation": False,
        "decision": "GO",
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "data": filtered_data_path,
        "selector": filtered_selector_path,
        "mapping": mapping_path,
        "audit": audit_path,
        "audit_payload": audit,
    }


def prepare_runtime_inputs(
    *,
    data_path: Path,
    folds_path: Path,
    image_manifest_path: Path,
    robust_base_path: Path,
    robust_report_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite runtime inputs")
    if sha256_file(data_path) != DATA_SHA256:
        raise ValueError("source data checksum mismatch")
    if sha256_file(image_manifest_path) != IMAGE_MANIFEST_SHA256:
        raise ValueError("first-image manifest checksum mismatch")
    data = pd.read_csv(data_path, dtype={"id": str})
    folds = read_folds(folds_path)
    aligned = folds.set_index("id").loc[data["id"]]
    development_mask = aligned["split"].eq("development").to_numpy()
    development = data.loc[development_mask].reset_index(drop=True)
    development_folds = aligned.loc[development_mask].reset_index()
    development_ids = development["id"].astype(str).tolist()
    if id_sequence_sha256(development_ids) != DEVELOPMENT_IDS_SHA256:
        raise ValueError("development ID sequence mismatch")

    report = json.loads(robust_report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise ValueError("robust-base report is not completed")
    if report.get("version") != SELECTOR_SOURCE_PROTOCOL:
        raise ValueError("robust-base protocol is not strict nested v2")
    if report.get("development_rows") != DEVELOPMENT_ROWS:
        raise ValueError("robust-base row count mismatch")
    if report.get("sealed_rows_seen_by_models_or_metrics") != 0:
        raise ValueError("robust base used sealed rows")
    if report.get("input_sha256", {}).get("data") != DATA_SHA256:
        raise ValueError("robust-base data provenance mismatch")
    if report.get("input_sha256", {}).get("folds") != FOLDS_SHA256:
        raise ValueError("robust-base folds provenance mismatch")
    if report.get("output_sha256", {}).get("robust_base_semantic_v3.npz") != sha256_file(
        robust_base_path
    ):
        raise ValueError("robust-base output checksum mismatch")
    robust = np.load(robust_base_path, allow_pickle=False)
    if "protocol_version" not in robust.files or str(robust["protocol_version"].item()) != (
        SELECTOR_SOURCE_PROTOCOL
    ):
        raise ValueError("robust-base NPZ lacks strict nested v2 protocol marker")
    expected_arrays = {
        "ids": np.asarray(development_ids),
        "labels": development["label"].to_numpy(np.int8),
        "categories": development["category"].astype(str).to_numpy(),
        "folds": development_folds["development_fold"].to_numpy(np.int8),
    }
    for key, expected in expected_arrays.items():
        if key not in robust.files or not np.array_equal(
            robust[key].astype(expected.dtype), expected
        ):
            raise ValueError(f"robust-base {key} mismatch")
    scores = robust["robust_base_score"].astype(np.float32)
    if scores.shape != (DEVELOPMENT_ROWS,) or not np.isfinite(scores).all():
        raise ValueError("robust-base scores are incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    data_out = output_dir / "development_data.csv"
    folds_out = output_dir / "development_folds.csv"
    selector_out = output_dir / "development_selector_oof.npz"
    provenance_out = output_dir / "development_selector_provenance.json"
    manifest_out = output_dir / "development_first_image_manifest.tsv.gz"
    mapping_out = output_dir / "development_id_map.csv"
    development.to_csv(data_out, index=False)
    development_folds.to_csv(folds_out, index=False)
    pd.DataFrame(
        {
            "local_index": np.arange(DEVELOPMENT_ROWS, dtype=np.int32),
            "id": development_ids,
            "original_index": np.flatnonzero(development_mask),
            "development_fold": expected_arrays["folds"],
        }
    ).to_csv(mapping_out, index=False)
    write_deterministic_npz(
        selector_out,
        {
            "ids": np.asarray(development_ids),
            "fold_ids": expected_arrays["folds"],
            "fused_scores": scores,
        },
    )
    wanted = set(development_ids)
    written: list[str] = []
    with gzip.open(image_manifest_path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames != ["id", "image_url"]:
            raise ValueError("first-image manifest schema mismatch")
        raw = io.BytesIO()
        with (
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
            io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_stream,
        ):
            writer = csv.DictWriter(text_stream, fieldnames=reader.fieldnames, delimiter="\t")
            writer.writeheader()
            for row in reader:
                if str(row["id"]) in wanted:
                    writer.writerow(row)
                    written.append(str(row["id"]))
        manifest_out.write_bytes(raw.getvalue())
    if written != development_ids:
        raise ValueError("development first-image manifest ID/order mismatch")
    provenance = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": "development_only",
        "rows": DEVELOPMENT_ROWS,
        "ids_sha256": DEVELOPMENT_IDS_SHA256,
        "selector_npz_sha256": sha256_file(selector_out),
        "selector_source": "semantic_v3_nested_robust_base",
        "selector_source_protocol": SELECTOR_SOURCE_PROTOCOL,
        "selector_source_sha256": sha256_file(robust_base_path),
        "sealed_rows_used_for_fit": False,
        "sealed_labels_used": False,
        "sealed_rows_used_for_selection": False,
        "sealed_rows_used_for_thresholds": False,
        "frozen_before_qwen_training": True,
    }
    provenance_out.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = [data_out, folds_out, selector_out, provenance_out, manifest_out, mapping_out]
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "selector_source_protocol": SELECTOR_SOURCE_PROTOCOL,
        "source_rows": len(data),
        "development_rows": DEVELOPMENT_ROWS,
        "sealed_rows_in_source": SEALED_ROWS,
        "sealed_rows_in_runtime_inputs": 0,
        "development_ids_sha256": DEVELOPMENT_IDS_SHA256,
        "output_sha256": {path.name: sha256_file(path) for path in outputs},
        "decision": "GO",
    }
    (output_dir / "zero_sealed_runtime_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def materialize_runtime_fold(
    *, runtime_dir: Path, output_dir: Path, component: str, outer_fold: int
) -> dict[str, Any]:
    required = {
        "data": runtime_dir / "development_data.csv",
        "folds": runtime_dir / "development_folds.csv",
        "selector": runtime_dir / "development_selector_oof.npz",
        "provenance": runtime_dir / "development_selector_provenance.json",
        "manifest": runtime_dir / "development_first_image_manifest.tsv.gz",
        "mapping": runtime_dir / "development_id_map.csv",
        "audit": runtime_dir / "zero_sealed_runtime_audit.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"runtime inputs missing: {missing}")
    audit = json.loads(required["audit"].read_text(encoding="utf-8"))
    if audit.get("decision") != "GO" or audit.get("sealed_rows_in_runtime_inputs") != 0:
        raise ValueError("runtime audit does not prove zero sealed rows")
    for name, expected in audit.get("output_sha256", {}).items():
        if sha256_file(runtime_dir / name) != expected:
            raise ValueError(f"runtime input checksum mismatch: {name}")
    data = pd.read_csv(required["data"], dtype={"id": str})
    folds = pd.read_csv(required["folds"], dtype={"id": str})
    ids = data["id"].astype(str).tolist()
    if len(ids) != DEVELOPMENT_ROWS or id_sequence_sha256(ids) != DEVELOPMENT_IDS_SHA256:
        raise ValueError("runtime development IDs mismatch")
    if not np.array_equal(data["id"].astype(str), folds["id"].astype(str)):
        raise ValueError("runtime data/folds ID mismatch")
    local_folds = folds["development_fold"].to_numpy(np.int8)
    _validate_selector_provenance(
        required["provenance"], expected_ids=ids, selector_path=required["selector"]
    )
    _load_development_selector(required["selector"], expected_ids=ids, expected_folds=local_folds)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite protocol inputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("data", "selector", "manifest", "audit"):
        shutil.copyfile(required[key], output_dir / required[key].name)
    source_mapping = pd.read_csv(required["mapping"], dtype={"id": str})
    source_mapping["outer_role"] = np.where(local_folds == outer_fold, "validation", "train")
    mapping_path = output_dir / "id_mapping.csv"
    source_mapping.to_csv(mapping_path, index=False)
    runtime_audit = {
        **audit,
        "component": component,
        "outer_fold": outer_fold,
        "outer_train_rows": int((local_folds != outer_fold).sum()),
        "outer_validation_rows": int((local_folds == outer_fold).sum()),
        "sealed_rows_in_mapping": 0,
        "mapping_sha256": sha256_file(mapping_path),
    }
    protocol_audit = output_dir / "protocol_input_audit.json"
    protocol_audit.write_text(
        json.dumps(runtime_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "data": output_dir / required["data"].name,
        "selector": output_dir / required["selector"].name,
        "manifest": output_dir / required["manifest"].name,
        "mapping": mapping_path,
        "audit": protocol_audit,
    }


def expected_step_policy(training_records: int) -> dict[str, int]:
    batches = math.ceil(training_records / BATCH_SIZE)
    optimizer_steps = math.ceil(batches * EPOCHS / GRADIENT_ACCUMULATION)
    return {"batches": batches, "optimizer_steps": optimizer_steps}


def build_static_protocol_manifest(data_path: Path, folds_path: Path) -> dict[str, Any]:
    data = pd.read_csv(data_path, dtype={"id": str})
    folds = read_folds(folds_path)
    aligned = folds.set_index("id").loc[data["id"]]
    development = aligned["split"].eq("development")
    fold_counts = Counter(aligned.loc[development, "development_fold"].astype(int))
    jobs = []
    for component in COMPONENTS:
        for fold in DEVELOPMENT_FOLDS:
            validation_rows = int(fold_counts[fold])
            jobs.append(
                {
                    "component": component,
                    "fold": fold,
                    "outer_train_rows": int(development.sum()) - validation_rows,
                    "outer_validation_rows": validation_rows,
                    "sealed_rows": 0,
                }
            )
    development_ids = data.loc[development.to_numpy(), "id"].astype(str).tolist()
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "prepared_blocked_on_development_only_selector_scores",
        "folds_sha256": sha256_file(folds_path),
        "data_sha256": sha256_file(data_path),
        "source_rows": len(data),
        "development_rows": int(development.sum()),
        "sealed_holdout_rows": int((~development).sum()),
        "development_ids_sha256": id_sequence_sha256(development_ids),
        "jobs": jobs,
        "required_jobs": len(jobs),
        "blockers": ["A frozen development-only selector-score NPZ and its provenance are absent."],
    }

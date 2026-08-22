from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_ID = "620"
VALIDATION_VERSION = "semantic_family_v3"
PROTOCOL_VERSION = "semantic_v3_evidence_disagreement_router_v2"
DEVELOPMENT_FOLDS = (0, 1, 2, 3, 4)
EXPECTED_DEVELOPMENT_ROWS = 11_118
EXPECTED_DEVELOPMENT_IDS_SHA256 = "4fd6adf6a20d782972fa63b3569bf13a7e36e5bf282b2a7bd351a0230ccffa8e"
EXPECTED_FOLDS_SHA256 = "16b9c47999c6c1e97b1317182adc356931db60a1156ec237fa496fa48c5387ae"

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDS = ROOT / "validation" / "semantic_family_v3" / "folds.csv"
DEFAULT_SPEC = Path(__file__).with_name("frozen_router_spec.json")
DEFAULT_RUNTIME_CONTRACT = Path(__file__).with_name("runtime_contract.template.json")
DEFAULT_BACKBONE_DECISION = Path(__file__).with_name("backbone_compatibility_decision.json")
DEFAULT_600_METRICS = (
    ROOT / "experiments" / "600_semantic_v3_qwen35_baselines" / "results" / "metrics.json"
)
DEFAULT_601_METRICS = (
    ROOT / "experiments" / "601_semantic_v3_visual_base_baselines" / "results" / "metrics.json"
)

SAFE_FOLD_COLUMNS = (
    "id",
    "category",
    "semantic_component",
    "component_size",
    "split",
    "development_fold",
)
FORBIDDEN_RUNTIME_ARRAYS = frozenset({"labels", "gold", "targets", "sealed_ids"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def id_sequence_sha256(values: list[str] | np.ndarray) -> str:
    return canonical_sha256([str(value) for value in values])


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_development_registry(
    path: Path,
    *,
    enforce_frozen: bool = True,
    expected_rows: int = EXPECTED_DEVELOPMENT_ROWS,
) -> pd.DataFrame:
    """Read semantic-v3 membership without ever loading its label column."""

    if enforce_frozen and sha256_file(path) != EXPECTED_FOLDS_SHA256:
        raise ValueError("semantic-v3 folds checksum mismatch")
    frame = pd.read_csv(
        path,
        usecols=list(SAFE_FOLD_COLUMNS),
        dtype={"id": str, "semantic_component": str},
    )
    if tuple(frame.columns) != SAFE_FOLD_COLUMNS:
        raise ValueError("semantic-v3 safe registry schema/order mismatch")
    if frame["id"].duplicated().any():
        raise ValueError("semantic-v3 registry contains duplicate ids")
    development = frame.loc[frame["split"].eq("development")].reset_index(drop=True)
    if len(development) != expected_rows:
        raise ValueError(f"development row count mismatch: {len(development)} != {expected_rows}")
    folds = development["development_fold"].astype(int)
    if set(folds) != set(DEVELOPMENT_FOLDS):
        raise ValueError("development folds must be exactly 0..4")
    if development.groupby("semantic_component")["development_fold"].nunique().max() != 1:
        raise ValueError("a semantic component crosses development folds")
    if enforce_frozen:
        ids_hash = id_sequence_sha256(development["id"].astype(str).tolist())
        if ids_hash != EXPECTED_DEVELOPMENT_IDS_SHA256:
            raise ValueError("semantic-v3 development ID sequence mismatch")
    return development


def require_complete_metrics(path: Path, *, experiment_id: str) -> dict[str, Any]:
    metrics = load_json(path)
    if str(metrics.get("experiment_id")) != experiment_id:
        raise ValueError(f"metrics are not for experiment {experiment_id}")
    if metrics.get("status") != "complete":
        raise RuntimeError(f"experiment {experiment_id} is not complete: {metrics.get('status')!r}")
    if metrics.get("sealed_holdout_used") is not False:
        raise ValueError(f"experiment {experiment_id} did not prove sealed exclusion")
    return metrics


def require_accepted_backbone_decision(path: Path) -> dict[str, Any]:
    decision = load_json(path)
    required_text = ("model_id", "revision", "license", "label_free_probe_report_sha256")
    if decision.get("status") != "accepted" or decision.get("compatibility_passed") is not True:
        raise RuntimeError("backbone compatibility decision is not accepted")
    if any(not isinstance(decision.get(key), str) or not decision[key] for key in required_text):
        raise ValueError("accepted backbone decision lacks immutable identity/probe fields")
    files = decision.get("files_sha256")
    if (
        not isinstance(files, dict)
        or not files
        or any(not isinstance(value, str) or len(value) != 64 for value in files.values())
    ):
        raise ValueError("accepted backbone decision lacks file checksums")
    return decision


def _require_arrays(bundle: np.lib.npyio.NpzFile, required: set[str], *, name: str) -> None:
    missing = required - set(bundle.files)
    if missing:
        raise ValueError(f"{name} bundle lacks arrays: {sorted(missing)}")


def validate_feature_bundle(
    path: Path,
    *,
    registry: pd.DataFrame,
    backbone_decision_path: Path,
    runtime_contract_path: Path = DEFAULT_RUNTIME_CONTRACT,
) -> dict[str, Any]:
    decision = require_accepted_backbone_decision(backbone_decision_path)
    contract = load_json(runtime_contract_path)["feature_bundle"]
    with np.load(path, allow_pickle=False) as bundle:
        _require_arrays(bundle, set(contract["required_arrays"]), name="feature")
        forbidden = FORBIDDEN_RUNTIME_ARRAYS & set(bundle.files)
        if forbidden:
            raise ValueError(f"feature bundle contains forbidden arrays: {sorted(forbidden)}")
        ids = bundle["ids"].astype(str)
        expected_ids = registry["id"].astype(str).to_numpy()
        if not np.array_equal(ids, expected_ids):
            raise ValueError("feature bundle IDs/order differ from development registry")
        scores = bundle["hypothesis_scores"]
        hypothesis_ids = bundle["hypothesis_ids"].astype(str)
        if scores.ndim != 2 or scores.shape != (len(ids), len(hypothesis_ids)):
            raise ValueError("hypothesis score matrix shape mismatch")
        if not np.isfinite(scores).all():
            raise ValueError("hypothesis scores contain non-finite values")
        if str(bundle["protocol_version"]) != PROTOCOL_VERSION:
            raise ValueError("feature protocol version mismatch")
        decision_hash = canonical_sha256(decision)
        if str(bundle["backbone_decision_sha256"]) != decision_hash:
            raise ValueError("feature bundle backbone decision checksum mismatch")
    return {
        "rows": len(registry),
        "feature_bundle_sha256": sha256_file(path),
        "backbone_decision_sha256": canonical_sha256(decision),
        "sealed_rows_loaded": 0,
        "labels_loaded": False,
    }


def validate_visual_component_bundle(
    path: Path,
    *,
    contract_path: Path,
    registry: pd.DataFrame,
) -> dict[str, np.ndarray]:
    contract = load_json(contract_path)
    if contract.get("status") != "complete" or contract.get("sealed_rows_in_outputs") != 0:
        raise RuntimeError("experiment-601 component contract is not complete and development-only")
    if contract.get("output_sha256", {}).get(path.name) != sha256_file(path):
        raise ValueError("experiment-601 component checksum mismatch")
    required = {
        "ids",
        "categories",
        "folds",
        "semantic_components",
        "robust_base_rank",
        "qwen3vl_rank",
    }
    with np.load(path, allow_pickle=False) as bundle:
        _require_arrays(bundle, required, name="visual")
        ids = bundle["ids"].astype(str)
        expected_ids = registry["id"].astype(str).to_numpy()
        if not np.array_equal(ids, expected_ids):
            raise ValueError("experiment-601 IDs/order mismatch")
        expected_categories = registry["category"].astype(str).to_numpy()
        expected_folds = registry["development_fold"].to_numpy(np.int8)
        expected_components = registry["semantic_component"].astype(str).to_numpy()
        for key, expected in (
            ("categories", expected_categories),
            ("folds", expected_folds),
            ("semantic_components", expected_components),
        ):
            if not np.array_equal(bundle[key].astype(expected.dtype), expected):
                raise ValueError(f"experiment-601 {key} mismatch")
        robust = bundle["robust_base_rank"].astype(np.float32)
        visual = bundle["qwen3vl_rank"].astype(np.float32)
        if not np.isfinite(robust).all() or not np.isfinite(visual).all():
            raise ValueError("experiment-601 ranks contain non-finite values")
    return {
        "ids": ids,
        "categories": expected_categories,
        "folds": expected_folds,
        "semantic_components": expected_components,
        "robust_base_rank": robust,
        "qwen3vl_rank": visual,
    }


def validate_qwen35_component_bundle(
    path: Path,
    *,
    contract_path: Path,
    registry: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Validate the future experiment-600 bundle without opening label arrays."""

    contract = load_json(contract_path)
    expected_contract = {
        "status": "complete",
        "validation_version": VALIDATION_VERSION,
        "development_rows": len(registry),
        "sealed_rows_in_outputs": 0,
        "completed_folds": list(DEVELOPMENT_FOLDS),
        "components": ["original", "specialist"],
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"experiment-600 component contract mismatch for {key}")
    if contract.get("output_sha256", {}).get(path.name) != sha256_file(path):
        raise ValueError("experiment-600 component checksum mismatch")
    required = {
        "ids",
        "categories",
        "folds",
        "semantic_components",
        "qwen35_original_rank",
        "qwen35_specialist_rank",
    }
    with np.load(path, allow_pickle=False) as bundle:
        _require_arrays(bundle, required, name="Qwen3.5")
        ids = bundle["ids"].astype(str)
        expected_ids = registry["id"].astype(str).to_numpy()
        if not np.array_equal(ids, expected_ids):
            raise ValueError("experiment-600 IDs/order mismatch")
        expected_categories = registry["category"].astype(str).to_numpy()
        expected_folds = registry["development_fold"].to_numpy(np.int8)
        expected_components = registry["semantic_component"].astype(str).to_numpy()
        for key, expected in (
            ("categories", expected_categories),
            ("folds", expected_folds),
            ("semantic_components", expected_components),
        ):
            if not np.array_equal(bundle[key].astype(expected.dtype), expected):
                raise ValueError(f"experiment-600 {key} mismatch")
        original = bundle["qwen35_original_rank"].astype(np.float32)
        specialist = bundle["qwen35_specialist_rank"].astype(np.float32)
        if not np.isfinite(original).all() or not np.isfinite(specialist).all():
            raise ValueError("experiment-600 ranks contain non-finite values")
    return {
        "ids": ids,
        "categories": expected_categories,
        "folds": expected_folds,
        "semantic_components": expected_components,
        "qwen35_original_rank": original,
        "qwen35_specialist_rank": specialist,
    }

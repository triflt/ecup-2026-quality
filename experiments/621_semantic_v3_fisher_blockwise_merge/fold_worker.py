"""Local fold worker for the 621 adapter merge.

The worker is intentionally model-runtime agnostic: a GPU runtime may produce
the train-only Fisher report, while this process performs the auditable adapter
merge and serialization.  It never reads labels and never accepts sealed rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from merge_utils import (
    AdapterMetadata,
    compute_blockwise_alpha,
    effective_lora_delta,
    factorize_delta,
    merge_effective_deltas,
    validate_compatible_metadata,
)
from safetensors.numpy import load_file, save_file

DEVELOPMENT_FOLDS = (0, 1, 2, 3, 4)
SAFE_MEMBERSHIP_COLUMNS = (
    "id",
    "category",
    "semantic_component",
    "component_size",
    "split",
    "development_fold",
)
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def id_sequence_sha256(values: Iterable[str]) -> str:
    return canonical_sha256([str(value) for value in values])


def _safe_membership(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(SAFE_MEMBERSHIP_COLUMNS).issubset(
            reader.fieldnames
        ):
            raise ValueError(f"{path} lacks safe semantic-v3 membership columns")
        rows: list[dict[str, str]] = []
        for raw in reader:
            rows.append({column: str(raw[column]) for column in SAFE_MEMBERSHIP_COLUMNS})
    return rows


def _safe_data_ids(path: Path) -> list[dict[str, str]]:
    """Read only IDs/categories from a scoped development data file."""

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(f"{path} lacks an id column")
        rows = []
        for raw in reader:
            if "split" in reader.fieldnames and raw["split"] != "development":
                raise ValueError("sealed or non-development rows are forbidden")
            rows.append({"id": str(raw["id"]), "category": str(raw.get("category", ""))})
    return rows


def load_fold_membership(
    data_path: Path,
    folds_path: Path,
    *,
    outer_fold: int,
) -> tuple[list[str], list[str]]:
    """Return train/validation IDs after a development-only membership audit."""

    if outer_fold not in DEVELOPMENT_FOLDS:
        raise ValueError(f"outer_fold must be one of {DEVELOPMENT_FOLDS}")
    data = _safe_data_ids(data_path)
    folds = _safe_membership(folds_path)
    if any(row["split"] not in {"development", "sealed_holdout"} for row in folds):
        raise ValueError("membership contains an unexpected split")
    development_folds = [row for row in folds if row["split"] == "development"]
    if [row["id"] for row in data] != [row["id"] for row in development_folds]:
        raise ValueError("data must be a scoped development-only ID sequence")
    if any(
        row["category"] and row["category"] != fold_row["category"]
        for row, fold_row in zip(data, development_folds, strict=True)
    ):
        raise ValueError("data and semantic-v3 membership categories differ")
    if any(int(row["development_fold"]) not in DEVELOPMENT_FOLDS for row in development_folds):
        raise ValueError("membership contains an invalid development fold")
    if any(
        row["split"] == "sealed_holdout" and int(row["development_fold"]) != -1 for row in folds
    ):
        raise ValueError("sealed membership rows must have fold=-1")
    components: dict[str, int] = {}
    for row in development_folds:
        component = row["semantic_component"]
        fold = int(row["development_fold"])
        previous = components.setdefault(component, fold)
        if previous != fold:
            raise ValueError("semantic components cross development folds")
    train = [row["id"] for row in development_folds if int(row["development_fold"]) != outer_fold]
    validation = [row["id"] for row in development_folds if int(row["development_fold"]) == outer_fold]
    if not train or not validation:
        raise ValueError("both train and validation memberships must be non-empty")
    return train, validation


def _read_config(path: Path, *, base_model_id: str, base_model_revision: str) -> AdapterMetadata:
    config_path = path / ADAPTER_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise ValueError("only PEFT LoRA adapters are supported")
    target_modules = config.get("target_modules")
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError("adapter config lacks target_modules")
    if not isinstance(rank, int) or rank <= 0 or not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError("adapter config has invalid rank or lora_alpha")
    identity = canonical_sha256(
        {
            key: (
                sorted(str(value) for value in target_modules)
                if key == "target_modules"
                else config.get(key)
            )
            for key in (
                "peft_type",
                "base_model_name_or_path",
                "target_modules",
                "r",
                "lora_alpha",
                "use_rslora",
                "fan_in_fan_out",
                "bias",
            )
        }
    )
    return AdapterMetadata(
        base_model_id=base_model_id,
        base_model_revision=base_model_revision,
        target_modules=tuple(sorted(str(value) for value in target_modules)),
        rank=rank,
        lora_alpha=float(alpha),
        use_rslora=bool(config.get("use_rslora", False)),
        config_identity_sha256=identity,
    )


def _load_config_payload(path: Path) -> dict[str, Any]:
    config_path = path / ADAPTER_CONFIG
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("adapter config must be a JSON object")
    return payload


def load_adapter(
    path: Path, *, base_model_id: str, base_model_revision: str
) -> tuple[AdapterMetadata, dict[str, np.ndarray]]:
    """Load a PEFT adapter directory without serializing its source path."""

    if not path.is_dir():
        raise ValueError("adapter input must be an extracted directory")
    metadata = _read_config(
        path, base_model_id=base_model_id, base_model_revision=base_model_revision
    )
    weights_path = path / ADAPTER_WEIGHTS
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    state = {str(key): np.asarray(value) for key, value in load_file(str(weights_path)).items()}
    if not state:
        raise ValueError("adapter weights are empty")
    if any(not np.isfinite(value).all() for value in state.values()):
        raise ValueError("adapter weights contain non-finite values")
    return metadata, state


def _split_lora_state(
    state: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    suffix_a = ".lora_A.weight"
    suffix_b = ".lora_B.weight"
    names_a = {key[: -len(suffix_a)] for key in state if key.endswith(suffix_a)}
    names_b = {key[: -len(suffix_b)] for key in state if key.endswith(suffix_b)}
    if names_a != names_b:
        raise ValueError("adapter has unmatched LoRA A/B tensors")
    return (
        {name: state[f"{name}{suffix_a}"] for name in names_a},
        {name: state[f"{name}{suffix_b}"] for name in names_b},
    )


def estimate_fisher_from_gradients(
    gradient_batches: Iterable[Mapping[str, np.ndarray]],
    *,
    train_rows: int,
    outer_fold: int,
    train_ids_sha256: str,
) -> dict[str, Any]:
    """Aggregate squared gradients into a train-only per-module Fisher report."""

    if train_rows <= 0 or outer_fold not in DEVELOPMENT_FOLDS:
        raise ValueError("invalid Fisher provenance")
    sums: dict[str, float] = {}
    batches = 0
    for batch in gradient_batches:
        if not batch:
            raise ValueError("empty gradient batch")
        batches += 1
        for module, gradient in batch.items():
            values = np.asarray(gradient, dtype=np.float64)
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"invalid gradient for {module}")
            sums[module] = sums.get(module, 0.0) + float(np.mean(values * values))
    if batches == 0 or not sums:
        raise ValueError("no gradients were supplied")
    fisher = {module: value / batches for module, value in sorted(sums.items())}
    return {
        "protocol": "621_train_only_fisher_v1",
        "outer_fold": outer_fold,
        "train_rows": train_rows,
        "train_ids_sha256": train_ids_sha256,
        "gradient_batches": batches,
        "validation_rows": 0,
        "sealed_rows": 0,
        "validation_labels_loaded": False,
        "sealed_labels_loaded": False,
        "fisher": fisher,
    }


def read_fisher_report(
    path: Path,
    *,
    outer_fold: int,
    train_ids: list[str],
    modules: set[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": "621_train_only_fisher_v1",
        "outer_fold": outer_fold,
        "train_rows": len(train_ids),
        "train_ids_sha256": id_sequence_sha256(train_ids),
        "validation_rows": 0,
        "sealed_rows": 0,
        "validation_labels_loaded": False,
        "sealed_labels_loaded": False,
    }
    mismatches = {
        key: (expected, report.get(key))
        for key, expected in required.items()
        if report.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Fisher provenance mismatch: {mismatches}")
    fisher = report.get("fisher")
    if not isinstance(fisher, dict) or set(fisher) != modules:
        raise ValueError("Fisher report modules differ from adapter modules")
    values: dict[str, float] = {}
    for module, value in fisher.items():
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid Fisher value for {module}")
        values[module] = value
    return values, report


def merge_fold_adapters(
    *,
    original_dir: Path,
    specialist_dir: Path,
    data_path: Path,
    folds_path: Path,
    fisher_report_path: Path,
    output_dir: Path,
    outer_fold: int,
    base_model_id: str,
    base_model_revision: str,
    max_alpha: float = 0.5,
    max_relative_error: float = 0.05,
) -> dict[str, Any]:
    train_ids, validation_ids = load_fold_membership(data_path, folds_path, outer_fold=outer_fold)
    original_meta, original_state = load_adapter(
        original_dir, base_model_id=base_model_id, base_model_revision=base_model_revision
    )
    specialist_meta, specialist_state = load_adapter(
        specialist_dir, base_model_id=base_model_id, base_model_revision=base_model_revision
    )
    validate_compatible_metadata(original_meta, specialist_meta)
    original_a, original_b = _split_lora_state(original_state)
    specialist_a, specialist_b = _split_lora_state(specialist_state)
    modules = set(original_a)
    if modules != set(specialist_a):
        raise ValueError("adapter target tensor modules differ")
    original_delta = {
        module: effective_lora_delta(
            original_a[module],
            original_b[module],
            lora_alpha=original_meta.lora_alpha,
            rank=original_meta.rank,
            use_rslora=original_meta.use_rslora,
        )
        for module in sorted(modules)
    }
    specialist_delta = {
        module: effective_lora_delta(
            specialist_a[module],
            specialist_b[module],
            lora_alpha=specialist_meta.lora_alpha,
            rank=specialist_meta.rank,
            use_rslora=specialist_meta.use_rslora,
        )
        for module in sorted(modules)
    }
    fisher_original, fisher_report = read_fisher_report(
        fisher_report_path,
        outer_fold=outer_fold,
        train_ids=train_ids,
        modules=modules,
    )
    fisher_specialist = fisher_report.get("fisher_specialist")
    if not isinstance(fisher_specialist, dict):
        raise TypeError("Fisher report must contain fisher_specialist")
    if set(fisher_specialist) != modules:
        raise ValueError("specialist Fisher modules differ from adapter modules")
    alpha = compute_blockwise_alpha(
        fisher_original,
        {module: float(fisher_specialist[module]) for module in modules},
        max_alpha=max_alpha,
    )
    merged = merge_effective_deltas(original_delta, specialist_delta, alpha)
    output_state: dict[str, np.ndarray] = {}
    reconstruction_errors: dict[str, float] = {}
    for module in sorted(merged):
        a, b, error = factorize_delta(
            merged[module],
            rank=original_meta.rank,
            lora_alpha=original_meta.lora_alpha,
            use_rslora=original_meta.use_rslora,
        )
        if error > max_relative_error:
            raise ValueError(f"SVD reconstruction error exceeds gate for {module}: {error}")
        reconstruction_errors[module] = error
        output_state[f"{module}.lora_A.weight"] = a.astype(np.float32)
        output_state[f"{module}.lora_B.weight"] = b.astype(np.float32)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite merged adapter output")
    output_dir.mkdir(parents=True, exist_ok=False)
    # Preserve the complete PEFT config so defaults such as task_type,
    # dropout, modules_to_save, and future PEFT fields remain loadable.
    config = _load_config_payload(original_dir)
    config["base_model_name_or_path"] = base_model_id
    config["revision"] = base_model_revision
    config["inference_mode"] = True
    config["target_modules"] = list(original_meta.target_modules)
    (output_dir / ADAPTER_CONFIG).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    save_file(output_state, str(output_dir / ADAPTER_WEIGHTS), metadata={"format": "pt"})
    manifest = {
        "protocol": "621_train_only_fisher_v1",
        "outer_fold": outer_fold,
        "train_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "sealed_rows": 0,
        "validation_labels_loaded": False,
        "sealed_labels_loaded": False,
        "train_ids_sha256": id_sequence_sha256(train_ids),
        "validation_ids_sha256": id_sequence_sha256(validation_ids),
        "original_weights_sha256": sha256_file(original_dir / ADAPTER_WEIGHTS),
        "specialist_weights_sha256": sha256_file(specialist_dir / ADAPTER_WEIGHTS),
        "original_config_identity_sha256": original_meta.config_identity_sha256,
        "specialist_config_identity_sha256": specialist_meta.config_identity_sha256,
        "merged_weights_sha256": sha256_file(output_dir / ADAPTER_WEIGHTS),
        "coefficients": alpha,
        "reconstruction_errors": reconstruction_errors,
        "decision": "READY_FOR_EXTERNAL_FOLD_SCORING",
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge one semantic-v3 fold's two PEFT adapters.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--outer-fold", required=True, type=int, choices=DEVELOPMENT_FOLDS)
    parser.add_argument("--original-adapter", required=True, type=Path)
    parser.add_argument("--specialist-adapter", required=True, type=Path)
    parser.add_argument("--fisher-report", required=True, type=Path)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--base-model-revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = merge_fold_adapters(
        original_dir=args.original_adapter,
        specialist_dir=args.specialist_adapter,
        data_path=args.data,
        folds_path=args.folds,
        fisher_report_path=args.fisher_report,
        output_dir=args.output_dir,
        outer_fold=args.outer_fold,
        base_model_id=args.base_model_id,
        base_model_revision=args.base_model_revision,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

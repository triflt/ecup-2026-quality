from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from semantic_v3_contract import (
    DEFAULT_FOLDS,
    DEVELOPMENT_FOLDS,
    ROBUST_PROTOCOL_VERSION,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def fold_category_ranks(
    values: np.ndarray, folds: np.ndarray, categories: np.ndarray
) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float32)
    for fold in DEVELOPMENT_FOLDS:
        for category in sorted(np.unique(categories)):
            mask = (folds == fold) & (categories == category)
            result[mask] = rank01(values[mask])
    return result


def assemble(
    *,
    folds_path: Path,
    robust_base_path: Path,
    prediction_paths: list[Path],
    report_paths: list[Path],
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    if len(prediction_paths) != 5 or len(report_paths) != 5:
        raise ValueError("exactly five Qwen fold predictions and reports are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(folds_path, dtype={"id": str})
    development = registry["split"].astype(str) == "development"
    dev = registry.loc[development].reset_index(drop=True)
    sealed_ids = set(registry.loc[~development, "id"].astype(str))
    ids = np.asarray(dev["id"].astype(str).tolist(), dtype=str)
    labels = dev["label"].to_numpy(np.int8)
    categories = np.asarray(dev["category"].astype(str).tolist(), dtype=str)
    folds = dev["development_fold"].to_numpy(np.int8)
    robust = np.load(robust_base_path, allow_pickle=False)
    if (
        "protocol_version" not in robust.files
        or str(robust["protocol_version"]) != ROBUST_PROTOCOL_VERSION
    ):
        raise ValueError("robust base is not the strict nested v2 protocol")
    for key, expected in (
        ("ids", ids),
        ("labels", labels),
        ("categories", categories),
        ("folds", folds),
    ):
        if not np.array_equal(robust[key].astype(expected.dtype), expected):
            raise ValueError(f"robust base {key} mismatch")
    frames = []
    report_contracts = {}
    for fold, (prediction_path, report_path) in enumerate(zip(prediction_paths, report_paths)):
        frame = pd.read_csv(prediction_path, dtype={"id": str})
        expected_ids = ids[folds == fold]
        if not np.array_equal(frame["id"].astype(str).to_numpy(), expected_ids):
            raise ValueError(f"Qwen fold {fold} id/order mismatch")
        if set(frame["id"].astype(str)) & sealed_ids:
            raise ValueError(f"Qwen fold {fold} contains sealed ids")
        if not np.all(frame["fold"].to_numpy(np.int8) == fold):
            raise ValueError(f"Qwen fold {fold} fold column mismatch")
        report = json.loads(report_path.read_text())
        expected_report = {
            "holdout_fold": fold,
            "validation_version": "semantic_family_v3",
            "development_only": True,
            "sealed_rows_seen_by_train_selector_threshold_eval": 0,
            "first_image_max_edge": 448,
            "first_image_max_pixels": 262144,
            "inference_image_index": 0,
            "inference_images_per_row": 1,
            "inference_passes": 1,
            "parent_recipe": "experiment_110_exact",
        }
        for key, expected in expected_report.items():
            if report.get(key) != expected:
                raise ValueError(f"Qwen fold {fold} report {key} mismatch")
        report_contracts[str(fold)] = expected_report
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).set_index("id").loc[ids]
    if not np.array_equal(combined["label"].to_numpy(np.int8), labels):
        raise ValueError("combined Qwen labels mismatch")
    if not np.array_equal(combined["category"].astype(str).to_numpy(), categories):
        raise ValueError("combined Qwen categories mismatch")
    qwen_score = combined["lora_score"].to_numpy(np.float32)
    robust_score = robust["robust_base_score"].astype(np.float32)
    qwen_rank = fold_category_ranks(qwen_score, folds, categories)
    robust_rank = fold_category_ranks(robust_score, folds, categories)
    output_path = output_dir / "semantic_v3_visual_base_components.npz"
    np.savez_compressed(
        output_path,
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        semantic_components=np.asarray(dev["semantic_component"].astype(str).tolist(), dtype=str),
        robust_base_score=robust_score,
        robust_base_rank=robust_rank,
        qwen3vl_score=qwen_score,
        qwen3vl_rank=qwen_rank,
    )
    report = {
        "version": "semantic_v3_visual_base_components_v1",
        "status": "complete",
        "development_rows": len(ids),
        "sealed_rows_in_outputs": 0,
        "fold_rows": {str(fold): int((folds == fold).sum()) for fold in DEVELOPMENT_FOLDS},
        "qwen_report_contracts": report_contracts,
        "input_sha256": {
            "folds": sha256(folds_path),
            "robust_base": sha256(robust_base_path),
            **{
                f"qwen_predictions_fold_{i}": sha256(path)
                for i, path in enumerate(prediction_paths)
            },
            **{f"qwen_report_fold_{i}": sha256(path) for i, path in enumerate(report_paths)},
        },
        "output_sha256": {output_path.name: sha256(output_path)},
        "route400_component_order": ["robust_base_rank", "qwen3vl_rank", "qwen35_rank"],
        "qwen35_component_present": False,
    }
    (output_dir / "component_contract.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--robust-base", type=Path, required=True)
    parser.add_argument("--qwen-predictions", type=Path, nargs=5, required=True)
    parser.add_argument("--qwen-reports", type=Path, nargs=5, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = assemble(
        folds_path=args.folds,
        robust_base_path=args.robust_base,
        prediction_paths=args.qwen_predictions,
        report_paths=args.qwen_reports,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

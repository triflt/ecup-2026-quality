"""Fixed-threshold evaluator for one merged-adapter validation fold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_PREDICTION_COLUMNS = ("id", "category", "fold", "lora_score")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validation_labels(data_path: Path, *, outer_fold: int) -> pd.DataFrame:
    """Parse labels only after a row is proven to be the requested dev fold."""

    rows: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "category", "split", "development_fold", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("evaluation data lacks required columns")
        for raw in reader:
            if raw["split"] != "development":
                raise ValueError("sealed or non-development rows are forbidden")
            if int(raw["development_fold"]) != outer_fold:
                continue
            rows.append(
                {
                    "id": str(raw["id"]),
                    "category": str(raw["category"]),
                    "label": int(raw["label"]),
                    "fold": outer_fold,
                }
            )
    if not rows:
        raise ValueError("validation fold is empty")
    frame = pd.DataFrame(rows)
    if frame["id"].duplicated().any() or not frame["label"].isin([0, 1]).all():
        raise ValueError("invalid validation labels")
    return frame


def _f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if 2 * tp + fp + fn == 0:
        return 0.0
    return 2.0 * tp / (2 * tp + fp + fn)


def evaluate_fold(
    *,
    data_path: Path,
    predictions_path: Path,
    thresholds_path: Path,
    merged_manifest_path: Path,
    output_path: Path,
    outer_fold: int,
) -> dict[str, Any]:
    labels = _validation_labels(data_path, outer_fold=outer_fold)
    predictions = pd.read_csv(predictions_path, dtype={"id": str, "category": str})
    if "label" in predictions.columns:
        raise ValueError("predictions must not carry validation labels")
    if tuple(predictions.columns) != REQUIRED_PREDICTION_COLUMNS:
        raise ValueError("prediction schema/order mismatch")
    if not predictions["fold"].astype(int).eq(outer_fold).all():
        raise ValueError("prediction fold mismatch")
    if predictions["id"].duplicated().any():
        raise ValueError("predictions contain duplicate IDs")
    if predictions["id"].tolist() != labels["id"].tolist():
        raise ValueError("prediction IDs/order differ from validation fold")
    if not np.isfinite(predictions["lora_score"].astype(float).to_numpy()).all():
        raise ValueError("prediction scores contain non-finite values")
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    if not isinstance(thresholds, dict):
        raise TypeError("thresholds must be a JSON object")
    merged_manifest = json.loads(merged_manifest_path.read_text(encoding="utf-8"))
    if merged_manifest.get("outer_fold") != outer_fold:
        raise ValueError("merged adapter manifest fold mismatch")
    if (
        merged_manifest.get("sealed_rows") != 0
        or merged_manifest.get("validation_labels_loaded") is not False
    ):
        raise ValueError("merged adapter manifest is not development-only")
    joined = labels.merge(
        predictions, on=["id", "category", "fold"], how="left", validate="one_to_one"
    )
    category_metrics: dict[str, dict[str, float | int]] = {}
    for category, group in joined.groupby("category", sort=True):
        if category not in thresholds:
            raise ValueError(f"missing frozen threshold for category {category}")
        threshold = float(thresholds[category])
        if not pd.notna(threshold):
            raise ValueError(f"invalid threshold for category {category}")
        # Match the frozen parent/route threshold convention exactly.
        predicted = group["lora_score"].astype(float) >= threshold
        category_metrics[str(category)] = {
            "rows": len(group),
            "positive": int(group["label"].sum()),
            "f1": _f1(group["label"], predicted),
            "threshold": threshold,
        }
    macro = sum(float(item["f1"]) for item in category_metrics.values()) / len(category_metrics)
    result = {
        "protocol": "621_fixed_threshold_fold_evaluation_v1",
        "outer_fold": outer_fold,
        "macro_f1": macro,
        "categories": category_metrics,
        "predictions_sha256": sha256_file(predictions_path),
        "merged_manifest_sha256": sha256_file(merged_manifest_path),
        "validation_labels_loaded": True,
        "sealed_labels_loaded": False,
        "thresholds_tuned_on_validation": False,
        "decision": "EVALUATED_NO_PROMOTION",
    }
    if output_path.exists():
        raise FileExistsError("refusing to overwrite evaluation report")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one merged adapter fold with frozen thresholds."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--merged-manifest", required=True, type=Path)
    parser.add_argument("--outer-fold", required=True, type=int, choices=range(5))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_fold(
        data_path=args.data,
        predictions_path=args.predictions,
        thresholds_path=args.thresholds,
        merged_manifest_path=args.merged_manifest,
        output_path=args.output,
        outer_fold=args.outer_fold,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

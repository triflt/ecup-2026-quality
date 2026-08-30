from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CATEGORIES = ("БАД", "Легковоспламеняющиеся")
SCREEN_FOLDS = (0, 3)


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(((labels == 1) & predictions).sum())
    fp = int(((labels == 0) & predictions).sum())
    fn = int(((labels == 1) & ~predictions).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def load_scores(paths: list[Path], *, expected_rows: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream)
    frame = pd.DataFrame(records)
    required = {"global_index", "id", "fold", "category", "score", "prediction"}
    if set(frame.columns) != required:
        raise ValueError(f"score schema mismatch: {sorted(frame.columns)}")
    frame = frame.sort_values("global_index").reset_index(drop=True)
    if len(frame) != expected_rows or frame["global_index"].astype(int).tolist() != list(range(expected_rows)):
        raise ValueError("score shards do not exactly cover the expected scope")
    if frame["id"].astype(str).duplicated().any() or not np.isfinite(frame["score"]).all():
        raise ValueError("duplicate IDs or non-finite scores")
    if not np.array_equal(frame["prediction"].astype(int), frame["score"].to_numpy() >= 0.0):
        raise ValueError("prediction differs from the frozen zero threshold")
    return frame


def evaluate(
    *,
    registry_path: Path,
    small_paths: list[Path],
    large_paths: list[Path],
    output_path: Path,
    folds_to_evaluate: tuple[int, ...] = SCREEN_FOLDS,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite evaluation report")
    registry = pd.read_csv(registry_path, dtype={"id": str})
    registry = registry.loc[registry["split"].astype(str).eq("development")].copy()
    registry = registry.sort_values("id", key=lambda values: values.astype(str)).reset_index(drop=True)
    if len(registry) != 11118:
        raise ValueError("unexpected semantic-v3 development scope")
    small = load_scores(small_paths, expected_rows=len(registry))
    large = load_scores(large_paths, expected_rows=len(registry))
    expected_ids = registry["id"].astype(str).tolist()
    if small["id"].astype(str).tolist() != expected_ids or large["id"].astype(str).tolist() != expected_ids:
        raise ValueError("score IDs/order differ from the immutable registry")
    for frame in (small, large):
        if frame["fold"].astype(int).tolist() != registry["development_fold"].astype(int).tolist():
            raise ValueError("score folds differ from the immutable registry")
        if frame["category"].astype(str).tolist() != registry["category"].astype(str).tolist():
            raise ValueError("score categories differ from the immutable registry")
    labels = registry["label"].to_numpy(np.int8)
    folds = registry["development_fold"].to_numpy(np.int8)
    categories = registry["category"].astype(str).to_numpy()
    small_pred = small["prediction"].to_numpy(np.int8)
    large_pred = large["prediction"].to_numpy(np.int8)
    fold_metrics: dict[str, Any] = {}
    for fold in folds_to_evaluate:
        fold_metrics[str(fold)] = {"categories": {}}
        for category in CATEGORIES:
            local = (folds == fold) & (categories == category)
            small_f1 = f1(labels[local], small_pred[local])
            large_f1 = f1(labels[local], large_pred[local])
            fold_metrics[str(fold)]["categories"][category] = {
                "small_f1": small_f1,
                "large_f1": large_f1,
                "delta": large_f1 - small_f1,
            }
        fold_metrics[str(fold)]["small_macro_f1"] = float(
            np.mean([value["small_f1"] for value in fold_metrics[str(fold)]["categories"].values()])
        )
        fold_metrics[str(fold)]["large_macro_f1"] = float(
            np.mean([value["large_f1"] for value in fold_metrics[str(fold)]["categories"].values()])
        )
        fold_metrics[str(fold)]["delta"] = (
            fold_metrics[str(fold)]["large_macro_f1"] - fold_metrics[str(fold)]["small_macro_f1"]
        )
    scope = np.isin(folds, folds_to_evaluate)
    category_metrics = {}
    for category in CATEGORIES:
        local = scope & (categories == category)
        small_f1 = f1(labels[local], small_pred[local])
        large_f1 = f1(labels[local], large_pred[local])
        category_metrics[category] = {
            "small_f1": small_f1,
            "large_f1": large_f1,
            "delta": large_f1 - small_f1,
        }
    corrected = int((scope & (small_pred != labels) & (large_pred == labels)).sum())
    regressed = int((scope & (small_pred == labels) & (large_pred != labels)).sum())
    gates = {
        "each_screen_fold_positive": all(fold_metrics[str(fold)]["delta"] > 0 for fold in folds_to_evaluate),
        "no_category_drop_below_minus_0_002": all(value["delta"] >= -0.002 for value in category_metrics.values()),
        "corrected_to_regressed_at_least_1_5": corrected > 0 if regressed == 0 else corrected / regressed >= 1.5,
    }
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "experiment_id": "640",
        "evaluation_version": "semantic_family_v3",
        "folds_evaluated": list(folds_to_evaluate),
        "thinking": False,
        "threshold": 0.0,
        "threshold_tuned": False,
        "sealed_rows": 0,
        "folds": fold_metrics,
        "categories": category_metrics,
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": None if regressed == 0 else corrected / regressed,
        "gates": gates,
        "passed": passed,
        "decision": "GO_TRAINING_GRID" if passed else "NO_GO_LARGE_PROMPT_TEACHER",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--small-score", type=Path, action="append", required=True)
    parser.add_argument("--large-score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    folds = (0, 1, 2, 3, 4) if args.full else SCREEN_FOLDS
    result = evaluate(
        registry_path=args.registry,
        small_paths=args.small_score,
        large_paths=args.large_score,
        output_path=args.output,
        folds_to_evaluate=folds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

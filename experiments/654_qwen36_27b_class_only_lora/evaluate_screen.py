from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CATEGORIES = ("БАД", "Легковоспламеняющиеся")
SCREEN_FOLDS = (0, 3)
REQUIRED = {"global_index", "id", "fold", "category", "score", "prediction"}
FORBIDDEN = {"label", "target", "gold", "answer"}


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(((labels == 1) & predictions).sum())
    fp = int(((labels == 0) & predictions).sum())
    fn = int(((labels == 1) & ~predictions).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def load_scores(paths: list[Path], expected_indices: list[int]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream)
    if any(not REQUIRED <= set(row) or FORBIDDEN & set(row) for row in records):
        raise ValueError("prediction schema or supervision mismatch")
    frame = pd.DataFrame([{key: row[key] for key in REQUIRED} for row in records])
    frame = frame.sort_values("global_index").reset_index(drop=True)
    if frame["global_index"].astype(int).tolist() != expected_indices:
        raise ValueError("predictions do not exactly cover the screen scope")
    if frame["id"].astype(str).duplicated().any() or not np.isfinite(frame["score"]).all():
        raise ValueError("duplicate IDs or non-finite scores")
    if not np.array_equal(frame["prediction"].astype(int), frame["score"].to_numpy() >= 0.0):
        raise ValueError("prediction differs from the frozen zero threshold")
    return frame


def evaluate(
    registry_path: Path,
    baseline_paths: list[Path],
    candidate_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite an evaluation report")
    registry = pd.read_csv(registry_path, dtype={"id": str})
    registry = registry.loc[registry["split"].astype(str).eq("development")].copy()
    registry = registry.sort_values("id", key=lambda values: values.astype(str)).reset_index(drop=True)
    registry["global_index"] = np.arange(len(registry), dtype=np.int64)
    registry = registry.loc[
        registry["development_fold"].astype(int).isin(SCREEN_FOLDS)
    ].copy()
    expected_indices = registry["global_index"].astype(int).tolist()
    baseline = load_scores(baseline_paths, expected_indices)
    candidate = load_scores(candidate_paths, expected_indices)
    registry = registry.sort_values("global_index").reset_index(drop=True)
    expected_ids = registry["id"].astype(str).tolist()
    for frame in (baseline, candidate):
        if frame["id"].astype(str).tolist() != expected_ids:
            raise ValueError("prediction IDs differ from the immutable registry")
        if frame["fold"].astype(int).tolist() != registry["development_fold"].astype(int).tolist():
            raise ValueError("prediction folds differ from the immutable registry")
        if frame["category"].astype(str).tolist() != registry["category"].astype(str).tolist():
            raise ValueError("prediction categories differ from the immutable registry")
    labels = registry["label"].to_numpy(np.int8)
    folds = registry["development_fold"].to_numpy(np.int8)
    categories = registry["category"].astype(str).to_numpy()
    base_pred = baseline["prediction"].to_numpy(np.int8)
    candidate_pred = candidate["prediction"].to_numpy(np.int8)
    fold_metrics: dict[str, Any] = {}
    for fold in SCREEN_FOLDS:
        values = {}
        for category in CATEGORIES:
            local = (folds == fold) & (categories == category)
            baseline_f1 = f1(labels[local], base_pred[local])
            candidate_f1 = f1(labels[local], candidate_pred[local])
            values[category] = {
                "baseline_f1": baseline_f1,
                "candidate_f1": candidate_f1,
                "delta": candidate_f1 - baseline_f1,
            }
        baseline_macro = float(np.mean([value["baseline_f1"] for value in values.values()]))
        candidate_macro = float(np.mean([value["candidate_f1"] for value in values.values()]))
        fold_metrics[str(fold)] = {
            "categories": values,
            "baseline_macro_f1": baseline_macro,
            "candidate_macro_f1": candidate_macro,
            "delta": candidate_macro - baseline_macro,
        }
    category_metrics = {}
    for category in CATEGORIES:
        local = categories == category
        baseline_f1 = f1(labels[local], base_pred[local])
        candidate_f1 = f1(labels[local], candidate_pred[local])
        category_metrics[category] = {
            "baseline_f1": baseline_f1,
            "candidate_f1": candidate_f1,
            "delta": candidate_f1 - baseline_f1,
        }
    corrected = int(((base_pred != labels) & (candidate_pred == labels)).sum())
    regressed = int(((base_pred == labels) & (candidate_pred != labels)).sum())
    flammable = categories == "Легковоспламеняющиеся"
    baseline_flammable_fn = int((flammable & (labels == 1) & (base_pred == 0)).sum())
    candidate_flammable_fn = int((flammable & (labels == 1) & (candidate_pred == 0)).sum())
    ratio = None if regressed == 0 else corrected / regressed
    mean_delta = float(np.mean([value["delta"] for value in fold_metrics.values()]))
    gates = {
        "each_screen_fold_positive": all(value["delta"] > 0 for value in fold_metrics.values()),
        "mean_delta_at_least_0_0015": mean_delta >= 0.0015,
        "no_category_drop_below_minus_0_002": all(
            value["delta"] >= -0.002 for value in category_metrics.values()
        ),
        "corrected_to_regressed_at_least_1_5": corrected > 0
        if regressed == 0
        else ratio is not None and ratio >= 1.5,
        "flammable_false_negatives_do_not_increase": (
            candidate_flammable_fn <= baseline_flammable_fn
        ),
    }
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "experiment_id": "654",
        "control_experiment_id": "641",
        "evaluation_version": "semantic_family_v3",
        "screen_folds": list(SCREEN_FOLDS),
        "threshold": 0.0,
        "threshold_tuned": False,
        "folds": fold_metrics,
        "mean_fold_delta": mean_delta,
        "categories": category_metrics,
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": ratio,
        "flammable_false_negatives": {
            "baseline": baseline_flammable_fn,
            "candidate": candidate_flammable_fn,
            "delta": candidate_flammable_fn - baseline_flammable_fn,
        },
        "gates": gates,
        "passed": passed,
        "sealed_rows": 0,
        "public_used": False,
        "decision": "OPEN_REMAINING_FOLDS" if passed else "REJECT_AT_SCREEN",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--baseline-score", type=Path, action="append", required=True)
    result.add_argument("--candidate-score", type=Path, action="append", required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            evaluate(args.registry, args.baseline_score, args.candidate_score, args.output),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

SCREEN_FOLDS = (0, 3)
CATEGORIES = ("БАД", "Легковоспламеняющиеся")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(
    *,
    registry_path: Path,
    baseline_paths: list[Path],
    large_paths: list[Path],
    output_path: Path,
    parent_evaluator_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite an evaluation report")
    parent = load_module(parent_evaluator_path, "exp654_evaluator")
    registry = pd.read_csv(registry_path, dtype={"id": str})
    registry = registry.loc[registry["split"].astype(str).eq("development")].copy()
    registry = registry.sort_values("id", key=lambda values: values.astype(str)).reset_index(
        drop=True
    )
    registry["global_index"] = np.arange(len(registry), dtype=np.int64)
    registry = registry.loc[registry["development_fold"].astype(int).isin(SCREEN_FOLDS)].copy()
    expected_indices = registry["global_index"].astype(int).tolist()
    baseline = parent.load_scores(baseline_paths, expected_indices)
    large = parent.load_scores(large_paths, expected_indices)
    registry = registry.sort_values("global_index").reset_index(drop=True)
    for frame in (baseline, large):
        if frame["id"].astype(str).tolist() != registry["id"].astype(str).tolist():
            raise ValueError("prediction IDs differ from the immutable registry")
        if frame["fold"].astype(int).tolist() != registry["development_fold"].astype(int).tolist():
            raise ValueError("prediction folds differ from the immutable registry")
        if frame["category"].astype(str).tolist() != registry["category"].astype(str).tolist():
            raise ValueError("prediction categories differ from the immutable registry")

    route_mask = registry["category"].astype(str).eq("Легковоспламеняющиеся").to_numpy()
    routed_pred = baseline["prediction"].to_numpy(np.int8).copy()
    routed_pred[route_mask] = large["prediction"].to_numpy(np.int8)[route_mask]
    baseline_pred = baseline["prediction"].to_numpy(np.int8)
    labels = registry["label"].to_numpy(np.int8)
    folds = registry["development_fold"].to_numpy(np.int8)
    categories = registry["category"].astype(str).to_numpy()

    fold_metrics: dict[str, Any] = {}
    for fold in SCREEN_FOLDS:
        values: dict[str, Any] = {}
        for category in CATEGORIES:
            local = (folds == fold) & (categories == category)
            baseline_f1 = parent.f1(labels[local], baseline_pred[local])
            candidate_f1 = parent.f1(labels[local], routed_pred[local])
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

    category_metrics: dict[str, Any] = {}
    for category in CATEGORIES:
        local = categories == category
        baseline_f1 = parent.f1(labels[local], baseline_pred[local])
        candidate_f1 = parent.f1(labels[local], routed_pred[local])
        category_metrics[category] = {
            "baseline_f1": baseline_f1,
            "candidate_f1": candidate_f1,
            "delta": candidate_f1 - baseline_f1,
        }
    corrected = int(((baseline_pred != labels) & (routed_pred == labels)).sum())
    regressed = int(((baseline_pred == labels) & (routed_pred != labels)).sum())
    ratio = None if regressed == 0 else corrected / regressed
    flammable = categories == "Легковоспламеняющиеся"
    baseline_fn = int((flammable & (labels == 1) & (baseline_pred == 0)).sum())
    candidate_fn = int((flammable & (labels == 1) & (routed_pred == 0)).sum())
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
        "flammable_false_negatives_do_not_increase": candidate_fn <= baseline_fn,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "658",
        "control_experiment_id": "641",
        "large_component_experiment_id": "654",
        "evaluation_version": "semantic_family_v3",
        "route": {"БАД": "641", "Легковоспламеняющиеся": "654"},
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
            "baseline": baseline_fn,
            "candidate": candidate_fn,
            "delta": candidate_fn - baseline_fn,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "remaining_folds_launched": False,
        "sealed_rows": 0,
        "public_used": False,
    }
    result["decision"] = "OPEN_REMAINING_FOLDS" if result["passed"] else "REJECT_AT_SCREEN"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Evaluate the frozen 4B/27B category route.")
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--baseline-score", type=Path, action="append", required=True)
    result.add_argument("--large-score", type=Path, action="append", required=True)
    result.add_argument("--parent-evaluator", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            evaluate(
                registry_path=args.registry,
                baseline_paths=args.baseline_score,
                large_paths=args.large_score,
                output_path=args.output,
                parent_evaluator_path=args.parent_evaluator,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

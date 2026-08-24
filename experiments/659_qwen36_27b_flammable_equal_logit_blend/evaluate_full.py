from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

FOLDS = (0, 1, 2, 3, 4)
CATEGORIES = ("БАД", "Легковоспламеняющиеся")
BASELINE_WEIGHT = 0.5
LARGE_WEIGHT = 0.5
THRESHOLD = 0.0


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_screen_gate(path: Path) -> dict[str, Any]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(gate, dict):
        raise ValueError("screen gate is not an object")
    expected = (
        gate.get("experiment_id") == "659"
        and gate.get("large_component_experiment_id") == "654"
        and gate.get("passed") is True
        and gate.get("decision") == "OPEN_REMAINING_FOLDS"
        and gate.get("weights") == {"641": 0.5, "654": 0.5}
        and gate.get("threshold") == 0.0
        and gate.get("threshold_tuned") is False
        and gate.get("sealed_rows") == 0
        and gate.get("public_used") is False
    )
    if not expected:
        raise ValueError("screen gate contract mismatch")
    screen_folds = gate.get("folds")
    if not isinstance(screen_folds, dict) or any(
        str(fold) not in screen_folds or "delta" not in screen_folds[str(fold)]
        for fold in (0, 3)
    ):
        raise ValueError("screen gate fold metrics are incomplete")
    return gate


def screen_fold_deltas(gate: dict[str, Any]) -> dict[str, float]:
    return {str(fold): float(gate["folds"][str(fold)]["delta"]) for fold in (0, 3)}


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    if positives == 0:
        raise ValueError("PR-AUC requires at least one positive row")
    if not np.isfinite(scores).all():
        raise ValueError("PR-AUC scores must be finite")
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    true_positives = np.cumsum(ranked_labels == 1)
    threshold_ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(ranked_scores) - 1]
    precision = true_positives[threshold_ends] / (threshold_ends + 1)
    recall = true_positives[threshold_ends] / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def evaluate(
    *,
    registry_path: Path,
    baseline_paths: list[Path],
    large_paths: list[Path],
    screen_gate_path: Path,
    output_path: Path,
    parent_evaluator_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite a full evaluation report")
    gate = validate_screen_gate(screen_gate_path)
    parent = load_module(parent_evaluator_path, "exp654_evaluator_for_659_full")
    registry = pd.read_csv(registry_path, dtype={"id": str})
    registry = registry.loc[registry["split"].astype(str).eq("development")].copy()
    registry = registry.sort_values("id", key=lambda values: values.astype(str)).reset_index(
        drop=True
    )
    registry["global_index"] = np.arange(len(registry), dtype=np.int64)
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
    baseline_score = baseline["score"].to_numpy(np.float64)
    large_score = large["score"].to_numpy(np.float64)
    routed_score = baseline_score.copy()
    routed_score[route_mask] = (
        BASELINE_WEIGHT * baseline_score[route_mask] + LARGE_WEIGHT * large_score[route_mask]
    )
    routed_pred = (routed_score >= THRESHOLD).astype(np.int8)
    baseline_pred = baseline["prediction"].to_numpy(np.int8)
    labels = registry["label"].to_numpy(np.int8)
    folds = registry["development_fold"].to_numpy(np.int8)
    categories = registry["category"].astype(str).to_numpy()

    fold_metrics: dict[str, Any] = {}
    for fold in FOLDS:
        category_values: dict[str, Any] = {}
        for category in CATEGORIES:
            local = (folds == fold) & (categories == category)
            baseline_f1 = parent.f1(labels[local], baseline_pred[local])
            candidate_f1 = parent.f1(labels[local], routed_pred[local])
            category_values[category] = {
                "baseline_f1": baseline_f1,
                "candidate_f1": candidate_f1,
                "delta": candidate_f1 - baseline_f1,
            }
        baseline_macro = float(
            np.mean([value["baseline_f1"] for value in category_values.values()])
        )
        candidate_macro = float(
            np.mean([value["candidate_f1"] for value in category_values.values()])
        )
        fold_metrics[str(fold)] = {
            "categories": category_values,
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
    flammable_pr_auc_folds: dict[str, Any] = {}
    for fold in FOLDS:
        local = flammable & (folds == fold)
        flammable_pr_auc_folds[str(fold)] = {
            "baseline_4b": average_precision(labels[local], baseline_score[local]),
            "large_27b": average_precision(labels[local], large_score[local]),
            "equal_logit_blend": average_precision(labels[local], routed_score[local]),
            "positives": int((labels[local] == 1).sum()),
            "rows": int(local.sum()),
        }
    flammable_pr_auc = {
        "metric": "average_precision",
        "diagnostic_only": True,
        "used_for_weights_or_threshold": False,
        "folds": flammable_pr_auc_folds,
        "pooled": {
            "baseline_4b": average_precision(labels[flammable], baseline_score[flammable]),
            "large_27b": average_precision(labels[flammable], large_score[flammable]),
            "equal_logit_blend": average_precision(labels[flammable], routed_score[flammable]),
            "positives": int((labels[flammable] == 1).sum()),
            "rows": int(flammable.sum()),
        },
    }
    baseline_fn = int((flammable & (labels == 1) & (baseline_pred == 0)).sum())
    candidate_fn = int((flammable & (labels == 1) & (routed_pred == 0)).sum())
    fold_deltas = {key: float(value["delta"]) for key, value in fold_metrics.items()}
    fold_wins = sum(delta > 0.0 for delta in fold_deltas.values())
    mean_delta = float(np.mean(list(fold_deltas.values())))
    screen_deltas = screen_fold_deltas(gate)
    screen_reproduced = all(
        abs(fold_deltas[str(fold)] - float(screen_deltas[str(fold)])) <= 1e-12
        for fold in (0, 3)
    )
    gates = {
        "screen_folds_reproduced_exactly": screen_reproduced,
        "at_least_four_of_five_fold_wins": fold_wins >= 4,
        "mean_delta_at_least_0_0015": mean_delta >= 0.0015,
        "no_category_drop_below_minus_0_002": all(
            value["delta"] >= -0.002 for value in category_metrics.values()
        ),
        "corrected_to_regressed_at_least_1_5": corrected > 0
        if regressed == 0
        else ratio is not None and ratio >= 1.5,
        "flammable_false_negatives_do_not_increase": candidate_fn <= baseline_fn,
    }
    passed = all(gates.values())
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "659",
        "control_experiment_id": "641",
        "large_component_experiment_id": "654",
        "evaluation_version": "semantic_family_v3",
        "route": {"БАД": "641", "Легковоспламеняющиеся": "equal_logit_blend_641_654"},
        "weights": {"641": BASELINE_WEIGHT, "654": LARGE_WEIGHT},
        "folds": fold_metrics,
        "fold_wins": fold_wins,
        "mean_fold_delta": mean_delta,
        "categories": category_metrics,
        "threshold": THRESHOLD,
        "threshold_tuned": False,
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": ratio,
        "flammable_false_negatives": {
            "baseline": baseline_fn,
            "candidate": candidate_fn,
            "delta": candidate_fn - baseline_fn,
        },
        "flammable_pr_auc": flammable_pr_auc,
        "screen_gate_sha256": sha256_file(screen_gate_path),
        "gates": gates,
        "passed": passed,
        "sealed_rows": 0,
        "public_used": False,
        "decision": "ACCEPT_FULL_COMPONENT_ROUTE" if passed else "REJECT_FULL_COMPONENT_ROUTE",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Evaluate frozen full 5-fold route 659.")
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--baseline-score", type=Path, action="append", required=True)
    result.add_argument("--large-score", type=Path, action="append", required=True)
    result.add_argument("--screen-gate", type=Path, required=True)
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
                screen_gate_path=args.screen_gate,
                output_path=args.output,
                parent_evaluator_path=args.parent_evaluator,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

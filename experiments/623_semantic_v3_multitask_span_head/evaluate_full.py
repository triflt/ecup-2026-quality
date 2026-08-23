"""Fail-closed five-fold acceptance evaluator for experiment 623.

The threshold contract is frozen in a separate donor-only command.  It reads
only the checksum-locked experiment-600 original OOF predictions and the
development-only semantic-v3 registry.  The evaluator verifies that immutable
contract before it opens any experiment-623 runtime or artifact directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCREEN_PATH = HERE / "evaluate_screen.py"
FROZEN_SPEC_PATH = HERE / "frozen_spec.json"

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 623
BOOTSTRAP_BATCH_SIZE = 128


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


screen = _load_module("_exp623_full_screen", SCREEN_PATH)
parent_screen = screen.parent_screen

FOLDS = parent_screen.FOLDS
CATEGORIES = parent_screen.CATEGORIES
FLAMMABLE = parent_screen.FLAMMABLE


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path.name}")
    return value


def _frozen_acceptance() -> dict[str, float | int]:
    spec = _load_json(FROZEN_SPEC_PATH)
    expected_top = {
        "schema_version": "exp623_multitask_span_head_v1",
        "experiment_id": "623",
        "validation": "semantic_family_v3",
        "uses_sealed_holdout": False,
    }
    for key, value in expected_top.items():
        if spec.get(key) != value:
            raise ValueError(f"frozen spec mismatch: {key}")
    acceptance = spec.get("full_acceptance")
    if not isinstance(acceptance, dict):
        raise TypeError("frozen spec lacks full_acceptance")
    expected = {
        "minimum_macro_delta": 0.003,
        "minimum_winning_folds": 4,
        "minimum_category_delta": 0.0,
        "minimum_corrected_to_regressed_ratio": 1.5,
        "minimum_component_bootstrap_probability_positive": 0.9,
    }
    if acceptance != expected:
        raise ValueError("frozen full acceptance contract mismatch")
    return expected


def freeze_full_thresholds(
    *, registry_path: Path, baseline_dirs: Mapping[int, Path], output_path: Path
) -> dict[str, Any]:
    """Freeze all five leave-target-fold-out thresholds without candidate inputs."""

    if output_path.exists():
        raise FileExistsError("refusing to overwrite frozen full thresholds")
    if set(baseline_dirs) != set(FOLDS):
        raise ValueError("exactly baseline folds 0..4 are required")
    registry = parent_screen._registry(registry_path)
    frames: dict[int, pd.DataFrame] = {}
    provenance: dict[str, Any] = {}
    for fold in FOLDS:
        frames[fold], provenance[str(fold)] = parent_screen._load_baseline_fold(
            baseline_dirs[fold],
            fold=fold,
            expected=parent_screen._expected_fold(registry, fold),
        )
    donor = pd.concat([frames[fold] for fold in FOLDS], ignore_index=True)
    thresholds: dict[str, dict[str, Any]] = {}
    for target_fold in FOLDS:
        thresholds[str(target_fold)] = {}
        for category in CATEGORIES:
            selected = donor.loc[
                donor["category"].eq(category)
                & ~donor["fold"].astype(int).eq(target_fold)
            ]
            donor_f1, threshold = parent_screen._best_threshold(
                selected["label"].to_numpy(np.int8),
                selected["lora_score"].to_numpy(np.float64),
            )
            thresholds[str(target_fold)][category] = {
                "threshold": threshold,
                "donor_f1": donor_f1,
                "donor_rows": len(selected),
                "excluded_target_fold": target_fold,
            }
    result: dict[str, Any] = {
        "protocol": "621_exp600_seed42_donor_thresholds_v1",
        "registry_sha256": screen.sha256_file(registry_path),
        "baseline_component": "original",
        "baseline_seed": 42,
        "screen_folds": list(FOLDS),
        "calibration": "leave_target_fold_out_over_fixed_exp600_original_oof",
        "candidate_inputs_read": 0,
        "sealed_rows": 0,
        "baseline_provenance": provenance,
        "thresholds": thresholds,
    }
    result["threshold_contract_sha256"] = screen.canonical_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _verify_full_threshold_contract(
    *, contract_path: Path, registry_path: Path, baseline_dirs: Mapping[int, Path]
) -> tuple[dict[str, Any], dict[int, pd.DataFrame]]:
    contract = _load_json(contract_path)
    parent_screen._verify_self_contract(contract, "threshold_contract_sha256")
    expected_top = {
        "protocol": "621_exp600_seed42_donor_thresholds_v1",
        "registry_sha256": screen.sha256_file(registry_path),
        "baseline_component": "original",
        "baseline_seed": 42,
        "screen_folds": list(FOLDS),
        "calibration": "leave_target_fold_out_over_fixed_exp600_original_oof",
        "candidate_inputs_read": 0,
        "sealed_rows": 0,
    }
    for key, value in expected_top.items():
        if contract.get(key) != value:
            raise ValueError(f"frozen full threshold contract mismatch: {key}")
    registry = parent_screen._registry(registry_path)
    frames: dict[int, pd.DataFrame] = {}
    for fold in FOLDS:
        frames[fold], hashes = parent_screen._load_baseline_fold(
            baseline_dirs[fold],
            fold=fold,
            expected=parent_screen._expected_fold(registry, fold),
        )
        if contract.get("baseline_provenance", {}).get(str(fold)) != hashes:
            raise ValueError(f"frozen full threshold donor provenance mismatch: fold {fold}")
    donor = pd.concat([frames[fold] for fold in FOLDS], ignore_index=True)
    for target_fold in FOLDS:
        for category in CATEGORIES:
            selected = donor.loc[
                donor["category"].eq(category)
                & ~donor["fold"].astype(int).eq(target_fold)
            ]
            donor_f1, threshold = parent_screen._best_threshold(
                selected["label"].to_numpy(np.int8),
                selected["lora_score"].to_numpy(np.float64),
            )
            expected = {
                "threshold": threshold,
                "donor_f1": donor_f1,
                "donor_rows": len(selected),
                "excluded_target_fold": target_fold,
            }
            frozen = contract.get("thresholds", {}).get(str(target_fold), {}).get(category)
            if frozen != expected:
                raise ValueError(
                    f"frozen full threshold value mismatch: fold {target_fold}/{category}"
                )
    return contract, frames


def _confusion_by_component(
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    categories: np.ndarray,
    component_inverse: np.ndarray,
    component_count: int,
) -> np.ndarray:
    counts = np.zeros((component_count, len(CATEGORIES), 3), dtype=np.int64)
    for category_index, category in enumerate(CATEGORIES):
        local = categories == category
        for statistic_index, event in enumerate(
            (
                local & (labels == 1) & (predictions == 1),
                local & (labels == 0) & (predictions == 1),
                local & (labels == 1) & (predictions == 0),
            )
        ):
            counts[:, category_index, statistic_index] = np.bincount(
                component_inverse[event], minlength=component_count
            )
    return counts


def _macro_from_confusion(counts: np.ndarray) -> np.ndarray:
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denominator = 2 * tp + fp + fn
    category_f1 = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denominator != 0,
    )
    return category_f1.mean(axis=-1)


def grouped_component_bootstrap(
    *,
    labels: np.ndarray,
    categories: np.ndarray,
    components: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    batch_size: int = BOOTSTRAP_BATCH_SIZE,
) -> dict[str, Any]:
    if iterations != BOOTSTRAP_ITERATIONS or seed != BOOTSTRAP_SEED:
        raise ValueError("bootstrap iterations and seed are frozen")
    if batch_size < 1:
        raise ValueError("bootstrap batch size must be positive")
    unique_components, inverse = np.unique(components.astype(str), return_inverse=True)
    component_count = len(unique_components)
    if component_count < 2:
        raise ValueError("grouped bootstrap needs at least two semantic components")
    baseline_counts = _confusion_by_component(
        labels=labels,
        predictions=baseline,
        categories=categories,
        component_inverse=inverse,
        component_count=component_count,
    )
    candidate_counts = _confusion_by_component(
        labels=labels,
        predictions=candidate,
        categories=categories,
        component_inverse=inverse,
        component_count=component_count,
    )
    rng = np.random.default_rng(seed)
    probability = np.full(component_count, 1.0 / component_count, dtype=np.float64)
    deltas = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        weights = rng.multinomial(component_count, probability, size=stop - start)
        base = np.einsum("bg,gcs->bcs", weights, baseline_counts, optimize=True)
        cand = np.einsum("bg,gcs->bcs", weights, candidate_counts, optimize=True)
        deltas[start:stop] = _macro_from_confusion(cand) - _macro_from_confusion(base)
    return {
        "unit": "semantic_component",
        "iterations": iterations,
        "seed": seed,
        "probability_delta_positive": float(np.mean(deltas > 0)),
        "delta_mean": float(np.mean(deltas)),
        "delta_ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


def evaluate_full(
    *,
    registry_path: Path,
    baseline_dirs: Mapping[int, Path],
    runtime_dirs: Mapping[int, Path],
    artifact_dirs: Mapping[int, Path],
    threshold_contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite full acceptance report")
    if set(baseline_dirs) != set(FOLDS):
        raise ValueError("exactly baseline folds 0..4 are required")
    if set(runtime_dirs) != set(FOLDS) or set(artifact_dirs) != set(FOLDS):
        raise ValueError("full evaluation requires exactly runtime and artifact folds 0..4")
    acceptance = _frozen_acceptance()
    # Donors and thresholds are verified before any candidate path is opened.
    thresholds, baseline_frames = _verify_full_threshold_contract(
        contract_path=threshold_contract_path,
        registry_path=registry_path,
        baseline_dirs=baseline_dirs,
    )
    registry = parent_screen._registry(registry_path)
    candidate_frames: dict[int, pd.DataFrame] = {}
    candidate_provenance: dict[str, Any] = {}
    prediction_audits: dict[str, Any] = {}
    for fold in FOLDS:
        expected = parent_screen._expected_fold(registry, fold)
        candidate_frames[fold], candidate_provenance[str(fold)], prediction_audits[str(fold)] = (
            screen._load_candidate_fold(
                artifact_dirs[fold], runtime_dirs[fold], fold=fold, expected=expected
            )
        )

    rows: list[pd.DataFrame] = []
    fold_metrics: dict[str, Any] = {}
    coverage_folds: dict[str, Any] = {}
    for fold in FOLDS:
        expected = parent_screen._expected_fold(registry, fold)
        baseline = baseline_frames[fold]
        candidate = candidate_frames[fold]
        assembled = expected.copy()
        assembled["semantic_component"] = registry.loc[
            registry["development_fold"].astype(int).eq(fold), "semantic_component"
        ].astype(str).to_numpy()
        assembled["baseline_score"] = baseline["lora_score"].to_numpy(np.float64)
        assembled["candidate_score"] = candidate["lora_score"].to_numpy(np.float64)
        baseline_pred = np.zeros(len(assembled), dtype=np.int8)
        candidate_pred = np.zeros(len(assembled), dtype=np.int8)
        fold_categories: dict[str, Any] = {}
        for category in CATEGORIES:
            mask = assembled["category"].eq(category).to_numpy()
            threshold = float(thresholds["thresholds"][str(fold)][category]["threshold"])
            baseline_pred[mask] = (
                assembled.loc[mask, "baseline_score"].to_numpy(np.float64) >= threshold
            )
            candidate_pred[mask] = (
                assembled.loc[mask, "candidate_score"].to_numpy(np.float64) >= threshold
            )
            labels = assembled.loc[mask, "label"].to_numpy(np.int8)
            baseline_f1 = parent_screen._f1(labels, baseline_pred[mask])
            candidate_f1 = parent_screen._f1(labels, candidate_pred[mask])
            fold_categories[category] = {
                "baseline_f1": baseline_f1,
                "candidate_f1": candidate_f1,
                "delta": candidate_f1 - baseline_f1,
                "threshold": threshold,
            }
        assembled["baseline_prediction"] = baseline_pred
        assembled["candidate_prediction"] = candidate_pred
        baseline_macro = float(np.mean([fold_categories[c]["baseline_f1"] for c in CATEGORIES]))
        candidate_macro = float(
            np.mean([fold_categories[c]["candidate_f1"] for c in CATEGORIES])
        )
        fold_metrics[str(fold)] = {
            "rows": len(assembled),
            "baseline_macro_f1": baseline_macro,
            "candidate_macro_f1": candidate_macro,
            "delta": candidate_macro - baseline_macro,
            "categories": fold_categories,
        }
        grounded = candidate["evidence"].astype(str).ne("NO_EVIDENCE")
        coverage_folds[str(fold)] = {
            "rows": len(candidate),
            "grounded_rows": int(grounded.sum()),
            "grounded_coverage": float(grounded.mean()),
        }
        assembled["fold"] = fold
        rows.append(assembled)

    combined = pd.concat(rows, ignore_index=True)
    labels = combined["label"].to_numpy(np.int8)
    categories = combined["category"].astype(str).to_numpy()
    components = combined["semantic_component"].astype(str).to_numpy()
    baseline_pred = combined["baseline_prediction"].to_numpy(np.int8)
    candidate_pred = combined["candidate_prediction"].to_numpy(np.int8)
    category_metrics: dict[str, Any] = {}
    for category in CATEGORIES:
        mask = categories == category
        baseline_f1 = parent_screen._f1(labels[mask], baseline_pred[mask])
        candidate_f1 = parent_screen._f1(labels[mask], candidate_pred[mask])
        category_metrics[category] = {
            "baseline_f1": baseline_f1,
            "candidate_f1": candidate_f1,
            "delta": candidate_f1 - baseline_f1,
        }
    baseline_macro = float(np.mean([category_metrics[c]["baseline_f1"] for c in CATEGORIES]))
    candidate_macro = float(np.mean([category_metrics[c]["candidate_f1"] for c in CATEGORIES]))
    macro_delta = candidate_macro - baseline_macro
    corrected = int(((baseline_pred != labels) & (candidate_pred == labels)).sum())
    regressed = int(((baseline_pred == labels) & (candidate_pred != labels)).sum())
    ratio = None if regressed == 0 else corrected / regressed
    positive = labels == 1
    flammable = categories == FLAMMABLE
    false_negatives = {
        "flammable": {
            "baseline": int((flammable & positive & (baseline_pred == 0)).sum()),
            "candidate": int((flammable & positive & (candidate_pred == 0)).sum()),
        },
        "all_positive": {
            "baseline": int((positive & (baseline_pred == 0)).sum()),
            "candidate": int((positive & (candidate_pred == 0)).sum()),
        },
    }
    for values in false_negatives.values():
        values["delta"] = values["candidate"] - values["baseline"]
    bootstrap = grouped_component_bootstrap(
        labels=labels,
        categories=categories,
        components=components,
        baseline=baseline_pred,
        candidate=candidate_pred,
    )
    candidates = pd.concat([candidate_frames[fold] for fold in FOLDS], ignore_index=True)
    coverage_categories: dict[str, Any] = {}
    for category in CATEGORIES:
        selected = candidates.loc[candidates["category"].eq(category)]
        grounded = selected["evidence"].astype(str).ne("NO_EVIDENCE")
        coverage_categories[category] = {
            "rows": len(selected),
            "grounded_rows": int(grounded.sum()),
            "grounded_coverage": float(grounded.mean()),
        }
    grounded = candidates["evidence"].astype(str).ne("NO_EVIDENCE")
    wins = sum(fold_metrics[str(fold)]["delta"] > 0 for fold in FOLDS)
    gates = {
        "macro_delta_at_least_0_003": macro_delta >= acceptance["minimum_macro_delta"],
        "folds_won_at_least_4": wins >= acceptance["minimum_winning_folds"],
        "no_category_drop": all(
            category_metrics[c]["delta"] >= acceptance["minimum_category_delta"]
            for c in CATEGORIES
        ),
        "corrected_to_regressed_at_least_1_5": (
            corrected > 0
            if regressed == 0
            else ratio >= acceptance["minimum_corrected_to_regressed_ratio"]
        ),
        "component_bootstrap_probability_at_least_0_90": (
            bootstrap["probability_delta_positive"]
            >= acceptance["minimum_component_bootstrap_probability_positive"]
        ),
        "flammable_false_negatives_do_not_increase": false_negatives["flammable"]["delta"]
        <= 0,
        "all_positive_false_negatives_do_not_increase": false_negatives["all_positive"][
            "delta"
        ]
        <= 0,
    }
    passed = all(gates.values())
    result: dict[str, Any] = {
        "protocol": "623_frozen_donor_full_five_fold_v1",
        "validation": "semantic_family_v3",
        "folds_evaluated": list(FOLDS),
        "registry_sha256": screen.sha256_file(registry_path),
        "threshold_contract_sha256": screen.sha256_file(threshold_contract_path),
        "frozen_spec_sha256": screen.sha256_file(FROZEN_SPEC_PATH),
        "thresholds_tuned_after_candidate": False,
        "candidate_predictions_contained_labels": False,
        "sealed_rows_loaded": 0,
        "baseline_macro_f1": baseline_macro,
        "candidate_macro_f1": candidate_macro,
        "macro_delta": macro_delta,
        "winning_folds": wins,
        "folds": fold_metrics,
        "categories": category_metrics,
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": ratio,
        "corrected_to_regressed_infinite": regressed == 0 and corrected > 0,
        "false_negatives": false_negatives,
        "component_bootstrap": bootstrap,
        "grounding": {
            "metric": "structural_exact_substring_coverage",
            "overall": {
                "rows": len(candidates),
                "grounded_rows": int(grounded.sum()),
                "grounded_coverage": float(grounded.mean()),
            },
            "categories": coverage_categories,
            "folds": coverage_folds,
            "human_quality_evaluated": False,
            "human_quality": None,
        },
        "gates": gates,
        "passed": passed,
        "decision": "GO_INTEGRATE_624" if passed else "NO_GO_REJECT_623",
        "candidate_provenance": candidate_provenance,
        "prediction_audits": prediction_audits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _fold_paths(values: Sequence[str], *, required: tuple[int, ...]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        raw_fold, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError("fold path must use FOLD=PATH")
        fold = int(raw_fold)
        if fold in parsed:
            raise ValueError(f"duplicate fold path: {fold}")
        parsed[fold] = Path(raw_path)
    if set(parsed) != set(required):
        raise ValueError(f"required folds are {required}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze and evaluate experiment 623 full grid.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-full-thresholds")
    freeze.add_argument("--registry", required=True, type=Path)
    freeze.add_argument("--baseline-fold", action="append", required=True)
    freeze.add_argument("--output", required=True, type=Path)
    evaluate = subparsers.add_parser("evaluate-full")
    evaluate.add_argument("--registry", required=True, type=Path)
    evaluate.add_argument("--baseline-fold", action="append", required=True)
    evaluate.add_argument("--runtime-fold", action="append", required=True)
    evaluate.add_argument("--artifact-fold", action="append", required=True)
    evaluate.add_argument("--threshold-contract", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baselines = _fold_paths(args.baseline_fold, required=FOLDS)
    if args.command == "freeze-full-thresholds":
        result = freeze_full_thresholds(
            registry_path=args.registry,
            baseline_dirs=baselines,
            output_path=args.output,
        )
    else:
        result = evaluate_full(
            registry_path=args.registry,
            baseline_dirs=baselines,
            runtime_dirs=_fold_paths(args.runtime_fold, required=FOLDS),
            artifact_dirs=_fold_paths(args.artifact_fold, required=FOLDS),
            threshold_contract_path=args.threshold_contract,
            output_path=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

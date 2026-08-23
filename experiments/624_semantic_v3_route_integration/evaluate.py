"""Fail-closed full-route integration evaluator for experiment 624.

The sole candidate replaces the original Qwen3.5 fold logits inside the exact
experiment-603 route.  Visual components and route weights are unchanged.  A
threshold for a target fold is calibrated only on the other four folds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP603 = ROOT / "experiments/603_semantic_v3_route_baseline"
EXP623 = ROOT / "experiments/623_semantic_v3_multitask_span_head"
SPEC_PATH = HERE / "frozen_spec.json"
ROUTE_PROTOCOL_PATH = EXP603 / "frozen_protocol.json"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


route603 = _module("_exp624_route603", EXP603 / "evaluate.py")
full623 = _module("_exp624_full623", EXP623 / "evaluate_full.py")
screen623 = full623.screen

FOLDS = (0, 1, 2, 3, 4)
CATEGORIES = ("БАД", "Легковоспламеняющиеся")
FLAMMABLE = "Легковоспламеняющиеся"
EXPECTED_BASELINE_MACRO_F1 = 0.9136312475000699
MAXIMUM_REFERENCE_ABSOLUTE_ERROR = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path.name}")
    return value


def _spec() -> dict[str, Any]:
    spec = _json(SPEC_PATH)
    expected = {
        "schema_version": "exp624_route_integration_v1",
        "experiment_id": "624",
        "winner_component_experiment": "623",
        "source_seed": 42,
        "validation": "semantic_family_v3",
        "outer_folds": list(FOLDS),
        "uses_sealed_holdout": False,
        "maximum_new_components": 1,
        "reference_route_weights_unchanged": True,
        "trained_second_level_router_allowed": False,
        "threshold_calibration": "leave_target_fold_out_donor_only",
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"experiment-624 frozen spec mismatch: {key}")
    return spec


def _verify_recipe(spec: dict[str, Any]) -> None:
    if sha256_file(ROUTE_PROTOCOL_PATH) != spec["experiment_603_protocol_sha256"]:
        raise ValueError("experiment-603 frozen route protocol checksum mismatch")
    if sha256_file(EXP623 / "frozen_spec.json") != spec["experiment_623_frozen_spec_sha256"]:
        raise ValueError("experiment-623 frozen spec checksum mismatch")
    for name, expected in spec["source_recipe_sha256"].items():
        if sha256_file(EXP623 / name) != expected:
            raise ValueError(f"experiment-623 source recipe checksum mismatch: {name}")


def _verify_accepted_623(
    *,
    report_path: Path,
    registry_path: Path,
    baseline_dirs: Mapping[int, Path],
    runtime_dirs: Mapping[int, Path],
    artifact_dirs: Mapping[int, Path],
    threshold_contract_path: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    if sha256_file(threshold_contract_path) != spec[
        "experiment_623_full_threshold_contract_sha256"
    ]:
        raise ValueError("experiment-623 threshold contract checksum mismatch")
    declared = _json(report_path)
    with tempfile.TemporaryDirectory(prefix="exp624-upstream-audit-") as temporary:
        recomputed = full623.evaluate_full(
            registry_path=registry_path,
            baseline_dirs=baseline_dirs,
            runtime_dirs=runtime_dirs,
            artifact_dirs=artifact_dirs,
            threshold_contract_path=threshold_contract_path,
            output_path=Path(temporary) / "full.json",
        )
    if declared != recomputed:
        raise ValueError("experiment-623 accepted report does not match strict recomputation")
    if declared.get("passed") is not True or declared.get("decision") != "GO_INTEGRATE_624":
        raise ValueError("experiment-623 did not pass its complete frozen gate")
    if declared.get("sealed_rows_loaded") != 0 or not all(declared.get("gates", {}).values()):
        raise ValueError("experiment-623 acceptance or sealed-data contract mismatch")
    return declared


def _assemble_original_logits(
    *, registry: pd.DataFrame, baseline_dirs: Mapping[int, Path]
) -> tuple[np.ndarray, dict[str, Any]]:
    logits = np.empty(len(registry), dtype=np.float64)
    provenance: dict[str, Any] = {}
    folds = registry["development_fold"].to_numpy(np.int8)
    for fold in FOLDS:
        expected = full623.parent_screen._expected_fold(registry, fold)
        frame, hashes = full623.parent_screen._load_baseline_fold(
            baseline_dirs[fold], fold=fold, expected=expected
        )
        logits[folds == fold] = frame["lora_score"].to_numpy(np.float64)
        provenance[str(fold)] = hashes
    return logits, provenance


def _assemble_candidate_logits(
    *,
    registry: pd.DataFrame,
    runtime_dirs: Mapping[int, Path],
    artifact_dirs: Mapping[int, Path],
) -> tuple[np.ndarray, dict[str, Any]]:
    logits = np.empty(len(registry), dtype=np.float64)
    provenance: dict[str, Any] = {}
    folds = registry["development_fold"].to_numpy(np.int8)
    for fold in FOLDS:
        expected = full623.parent_screen._expected_fold(registry, fold)
        frame, hashes, prediction_audit = screen623._load_candidate_fold(
            artifact_dirs[fold], runtime_dirs[fold], fold=fold, expected=expected
        )
        logits[folds == fold] = frame["lora_score"].to_numpy(np.float64)
        provenance[str(fold)] = {"artifact": hashes, "prediction_audit": prediction_audit}
    return logits, provenance


def _route_scores(
    *, qwen_logits: np.ndarray, visual: dict[str, np.ndarray], route: dict[str, Any]
) -> np.ndarray:
    categories = visual["categories"].astype(str)
    folds = visual["folds"].astype(np.int8)
    qwen_rank = route603.fold_category_ranks(route603.sigmoid(qwen_logits), folds, categories)
    scores = np.empty(len(categories), dtype=np.float64)
    for category in CATEGORIES:
        mask = categories == category
        weights = route[category]["weights"]
        scores[mask] = (
            weights["robust_base_rank"] * visual["robust_base_rank"][mask]
            + weights["qwen3vl_rank"] * visual["qwen3vl_rank"][mask]
            + weights["qwen35_rank"] * qwen_rank[mask]
        )
    return scores


def _donor_only_predictions(
    *, labels: np.ndarray, categories: np.ndarray, folds: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    predictions = np.zeros(len(labels), dtype=np.int8)
    calibration: dict[str, Any] = {}
    for fold in FOLDS:
        calibration[str(fold)] = {}
        for category in CATEGORIES:
            donor = (folds != fold) & (categories == category)
            target = (folds == fold) & (categories == category)
            donor_f1, threshold = route603.best_threshold(labels[donor], scores[donor])
            predictions[target] = (scores[target] >= threshold).astype(np.int8)
            calibration[str(fold)][category] = {
                "threshold": threshold,
                "donor_f1": donor_f1,
                "donor_rows": int(donor.sum()),
                "target_rows": int(target.sum()),
                "excluded_target_fold": fold,
            }
    return predictions, calibration


def _f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return route603.f1(labels, predictions)


def evaluate(
    *,
    visual_bundle_path: Path,
    visual_contract_path: Path,
    registry_path: Path,
    baseline_dirs: Mapping[int, Path],
    runtime_dirs: Mapping[int, Path],
    artifact_dirs: Mapping[int, Path],
    threshold_contract_path: Path,
    accepted_623_report_path: Path,
    output_path: Path,
    manifest_path: Path,
    enforce_frozen_registry: bool = True,
) -> dict[str, Any]:
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite route evaluation or accepted manifest")
    if set(baseline_dirs) != set(FOLDS):
        raise ValueError("exactly baseline folds 0..4 are required")
    if set(runtime_dirs) != set(FOLDS) or set(artifact_dirs) != set(FOLDS):
        raise ValueError("exactly candidate runtime/artifact folds 0..4 are required")
    spec = _spec()
    _verify_recipe(spec)
    accepted_623 = _verify_accepted_623(
        report_path=accepted_623_report_path,
        registry_path=registry_path,
        baseline_dirs=baseline_dirs,
        runtime_dirs=runtime_dirs,
        artifact_dirs=artifact_dirs,
        threshold_contract_path=threshold_contract_path,
        spec=spec,
    )
    protocol = _json(ROUTE_PROTOCOL_PATH)
    visual = route603.load_visual_bundle(
        bundle_path=visual_bundle_path,
        contract_path=visual_contract_path,
        protocol=protocol,
    )
    registry = route603.load_registry(
        registry_path, visual, enforce_frozen=enforce_frozen_registry
    )
    if int((registry["split"] != "development").sum()) != 0:
        raise ValueError("route evaluation loaded non-development rows")
    original_logits, original_provenance = _assemble_original_logits(
        registry=registry, baseline_dirs=baseline_dirs
    )
    candidate_logits, candidate_provenance = _assemble_candidate_logits(
        registry=registry, runtime_dirs=runtime_dirs, artifact_dirs=artifact_dirs
    )
    route = protocol["route"]
    original_scores = _route_scores(qwen_logits=original_logits, visual=visual, route=route)
    candidate_scores = _route_scores(qwen_logits=candidate_logits, visual=visual, route=route)
    labels = visual["labels"].astype(np.int8)
    categories = visual["categories"].astype(str)
    folds = visual["folds"].astype(np.int8)
    components = visual["semantic_components"].astype(str)
    baseline, baseline_calibration = _donor_only_predictions(
        labels=labels, categories=categories, folds=folds, scores=original_scores
    )
    candidate, candidate_calibration = _donor_only_predictions(
        labels=labels, categories=categories, folds=folds, scores=candidate_scores
    )

    category_metrics: dict[str, Any] = {}
    for category in CATEGORIES:
        mask = categories == category
        before = _f1(labels[mask], baseline[mask])
        after = _f1(labels[mask], candidate[mask])
        category_metrics[category] = {"baseline_f1": before, "candidate_f1": after, "delta": after - before}
    baseline_macro = float(np.mean([category_metrics[c]["baseline_f1"] for c in CATEGORIES]))
    candidate_macro = float(np.mean([category_metrics[c]["candidate_f1"] for c in CATEGORIES]))
    if abs(baseline_macro - EXPECTED_BASELINE_MACRO_F1) > MAXIMUM_REFERENCE_ABSOLUTE_ERROR:
        raise AssertionError("experiment-603 full-route reference mismatch")
    macro_delta = candidate_macro - baseline_macro
    fold_metrics: dict[str, Any] = {}
    for fold in FOLDS:
        by_category = {}
        for category in CATEGORIES:
            mask = (folds == fold) & (categories == category)
            before = _f1(labels[mask], baseline[mask])
            after = _f1(labels[mask], candidate[mask])
            by_category[category] = {"baseline_f1": before, "candidate_f1": after, "delta": after - before}
        before_macro = float(np.mean([by_category[c]["baseline_f1"] for c in CATEGORIES]))
        after_macro = float(np.mean([by_category[c]["candidate_f1"] for c in CATEGORIES]))
        fold_metrics[str(fold)] = {
            "baseline_macro_f1": before_macro,
            "candidate_macro_f1": after_macro,
            "delta": after_macro - before_macro,
            "categories": by_category,
        }
    corrected = int(((baseline != labels) & (candidate == labels)).sum())
    regressed = int(((baseline == labels) & (candidate != labels)).sum())
    ratio = None if regressed == 0 else corrected / regressed
    positive = labels == 1
    flammable = categories == FLAMMABLE
    false_negatives = {
        "flammable": {
            "baseline": int((flammable & positive & (baseline == 0)).sum()),
            "candidate": int((flammable & positive & (candidate == 0)).sum()),
        },
        "all_positive": {
            "baseline": int((positive & (baseline == 0)).sum()),
            "candidate": int((positive & (candidate == 0)).sum()),
        },
    }
    for values in false_negatives.values():
        values["delta"] = values["candidate"] - values["baseline"]
    bootstrap = full623.grouped_component_bootstrap(
        labels=labels,
        categories=categories,
        components=components,
        baseline=baseline,
        candidate=candidate,
    )
    wins = sum(fold_metrics[str(fold)]["delta"] > 0 for fold in FOLDS)
    gate = spec["acceptance"]
    gates = {
        "macro_delta_at_least_0_003": macro_delta >= gate["minimum_macro_delta"],
        "folds_won_at_least_4": wins >= gate["minimum_winning_folds"],
        "no_category_drop": all(category_metrics[c]["delta"] >= gate["minimum_category_delta"] for c in CATEGORIES),
        "corrected_to_regressed_at_least_1_5": corrected > 0 if regressed == 0 else ratio >= gate["minimum_corrected_to_regressed_ratio"],
        "component_bootstrap_probability_at_least_0_90": bootstrap["probability_delta_positive"] >= gate["minimum_component_bootstrap_probability_positive"],
        "flammable_false_negatives_do_not_increase": false_negatives["flammable"]["delta"] <= gate["maximum_flammable_false_negative_increase"],
        "all_positive_false_negatives_do_not_increase": false_negatives["all_positive"]["delta"] <= gate["maximum_all_positive_false_negative_increase"],
    }
    passed = all(gates.values())
    route_threshold_contract = {
        "calibration": "leave_target_fold_out_donor_only",
        "candidate_variants_evaluated": 1,
        "target_fold_labels_used_for_own_threshold": 0,
        "sealed_rows": 0,
        "baseline": baseline_calibration,
        "candidate": candidate_calibration,
    }
    route_threshold_contract["contract_sha256"] = canonical_sha256(route_threshold_contract)
    result: dict[str, Any] = {
        "schema_version": "exp624_full_route_evaluation_v1",
        "experiment_id": "624",
        "validation": "semantic_family_v3",
        "winner_component_experiment": "623",
        "folds_evaluated": list(FOLDS),
        "uses_sealed_holdout": False,
        "sealed_rows_loaded": 0,
        "reference_route_weights_unchanged": True,
        "trained_second_level_router_used": False,
        "candidate_variants_evaluated": 1,
        "route_protocol_sha256": sha256_file(ROUTE_PROTOCOL_PATH),
        "experiment_623_full_report_sha256": sha256_file(accepted_623_report_path),
        "experiment_623_threshold_contract_sha256": sha256_file(threshold_contract_path),
        "baseline_macro_f1": baseline_macro,
        "candidate_macro_f1": candidate_macro,
        "macro_delta": macro_delta,
        "winning_folds": wins,
        "categories": category_metrics,
        "folds": fold_metrics,
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": ratio,
        "corrected_to_regressed_infinite": regressed == 0 and corrected > 0,
        "false_negatives": false_negatives,
        "component_bootstrap": bootstrap,
        "route_threshold_contract": route_threshold_contract,
        "baseline_provenance": original_provenance,
        "candidate_provenance": candidate_provenance,
        "upstream_623_gates": accepted_623["gates"],
        "gates": gates,
        "passed": passed,
        "decision": "ACCEPT" if passed else "REJECT_KEEP_ORIGINAL_603",
    }
    result["report_sha256"] = canonical_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if passed:
        manifest: dict[str, Any] = {
            "schema_version": "exp624_route_recipe_v1",
            "experiment_id": "624",
            "decision": "ACCEPT",
            "winner_component_experiment": "623",
            "validation": "semantic_family_v3",
            "source_seed": spec["source_seed"],
            "threshold_contract_sha256": spec["experiment_623_full_threshold_contract_sha256"],
            "uses_sealed_holdout": False,
            "reference_route_weights_unchanged": True,
            "trained_second_level_router_used": False,
            "source_recipe_sha256": spec["source_recipe_sha256"],
            "route_protocol_sha256": sha256_file(ROUTE_PROTOCOL_PATH),
            "route_threshold_contract_sha256": route_threshold_contract["contract_sha256"],
            "route_evaluation_report_sha256": sha256_file(output_path),
            "experiment_623_full_report_sha256": sha256_file(accepted_623_report_path),
            "experiment_623_candidate_provenance": accepted_623["candidate_provenance"],
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _fold_paths(values: Sequence[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        raw_fold, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError("fold path must use FOLD=PATH")
        fold = int(raw_fold)
        if fold in parsed:
            raise ValueError(f"duplicate fold path: {fold}")
        parsed[fold] = Path(raw_path)
    if set(parsed) != set(FOLDS):
        raise ValueError("required folds are 0..4")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen experiment-624 route.")
    parser.add_argument("--visual-bundle", required=True, type=Path)
    parser.add_argument("--visual-contract", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--baseline-fold", required=True, action="append")
    parser.add_argument("--runtime-fold", required=True, action="append")
    parser.add_argument("--artifact-fold", required=True, action="append")
    parser.add_argument("--exp623-threshold-contract", required=True, type=Path)
    parser.add_argument("--accepted-exp623-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--accepted-manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        visual_bundle_path=args.visual_bundle,
        visual_contract_path=args.visual_contract,
        registry_path=args.registry,
        baseline_dirs=_fold_paths(args.baseline_fold),
        runtime_dirs=_fold_paths(args.runtime_fold),
        artifact_dirs=_fold_paths(args.artifact_fold),
        threshold_contract_path=args.exp623_threshold_contract,
        accepted_623_report_path=args.accepted_exp623_report,
        output_path=args.output,
        manifest_path=args.accepted_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

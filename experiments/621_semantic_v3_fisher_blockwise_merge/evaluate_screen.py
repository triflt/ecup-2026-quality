"""Fail-closed two-fold screen for experiment 621.

Thresholds are materialized in a separate donor-only step.  The screen step
will not derive, alter, or select a threshold after candidate predictions
exist: it accepts only the hash-bound donor contract produced here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Exact development-only projection produced before this experiment.  The
# source registry also contains the sealed partition, so this evaluator never
# accepts that source file even though its development rows would be filterable.
EXPECTED_FOLDS_SHA256 = "6ebbef5b1e561e255a00b375c703a25914ad2066e757b55d5185d7b490e7d4e6"
FOLDS = (0, 1, 2, 3, 4)
SCREEN_FOLDS = (0, 3)
CATEGORIES = ("БАД", "Легковоспламеняющиеся")
FLAMMABLE = "Легковоспламеняющиеся"
EXPECTED_BASELINE_PREDICTION_SHA256: dict[int, str] | None = {
    0: "c8958d0851046f694e269c74ffcbe0adeffb51114d129ac6f9b36554cd2b269b",
    1: "6236f1e6a5a5540b243656388e4cd08c0050fb049664e3dd14a988a63d7f0aca",
    2: "47cd543bbed5e7e499320d3b6afc944f7d6b18d0ce972ab857846b535fc20bfd",
    3: "dd0e681f895f89de83fed4c73451e8cc835bb29be37ebafe6c04271f73394fbb",
    4: "443d1eb481db916fc1c12c97cc673eb9d6eb81714885a2d3f7478bcf92d7c63b",
}
BASELINE_COLUMNS = ("id", "category", "label", "fold", "lora_score")
CANDIDATE_COLUMNS = ("id", "category", "fold", "lora_score")


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path.name}")
    return value


def _verify_self_contract(contract: Mapping[str, Any], field: str) -> None:
    expected = contract.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"missing {field}")
    payload = dict(contract)
    del payload[field]
    if canonical_sha256(payload) != expected:
        raise ValueError(f"{field} mismatch")


def _registry(path: Path) -> pd.DataFrame:
    if sha256_file(path) != EXPECTED_FOLDS_SHA256:
        raise ValueError("semantic-v3 fold registry checksum mismatch")
    frame = pd.read_csv(path, dtype={"id": str, "category": str})
    required = (
        "id",
        "category",
        "label",
        "semantic_component",
        "component_size",
        "split",
        "development_fold",
    )
    if tuple(frame.columns) != required:
        raise ValueError("semantic-v3 fold registry schema/order mismatch")
    if not frame["split"].eq("development").all():
        raise ValueError("sealed or non-development registry rows are forbidden")
    if frame["id"].duplicated().any():
        raise ValueError("semantic-v3 registry IDs are not unique")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("semantic-v3 labels are not binary")
    if set(frame["development_fold"].astype(int)) != set(FOLDS):
        raise ValueError("semantic-v3 registry fold set mismatch")
    if set(frame["category"].astype(str)) != set(CATEGORIES):
        raise ValueError("semantic-v3 category set mismatch")
    return frame


def _expected_fold(registry: pd.DataFrame, fold: int) -> pd.DataFrame:
    return registry.loc[
        registry["development_fold"].astype(int).eq(fold), ["id", "category", "label"]
    ].reset_index(drop=True)


def _f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2.0 * tp / max(1, 2 * tp + fp + fn)


def _best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Exact exp600 parent routine: 700 quantiles and first maximum."""

    if len(scores) == 0:
        raise ValueError("empty donor threshold partition")
    candidates = np.unique(np.quantile(scores, np.linspace(0.002, 0.998, 700)))
    best_f1 = -1.0
    best_threshold = float(candidates[0])
    for threshold in candidates:
        value = _f1(labels, scores >= threshold)
        if value > best_f1:
            best_f1 = value
            best_threshold = float(threshold)
    return best_f1, best_threshold


def _load_baseline_fold(
    directory: Path, *, fold: int, expected: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract_path = directory / "output_contract.runtime.json"
    predictions_path = directory / "lora_holdout_predictions.csv"
    contract = _load_json(contract_path)
    _verify_self_contract(contract, "contract_sha256")
    expected_contract = {
        "experiment_id": "600",
        "protocol_version": "semantic_family_v3",
        "component": "original",
        "outer_fold": fold,
        "prediction_rows": len(expected),
        "sealed_rows_in_predictions": 0,
        "sealed_rows_used_for_threshold": 0,
        "sealed_rows_used_for_evaluation": 0,
        "decision": "GO",
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise ValueError(f"baseline fold {fold} contract mismatch: {key}")
    actual_prediction_sha256 = sha256_file(predictions_path)
    if contract.get("predictions_sha256") != actual_prediction_sha256:
        raise ValueError(f"baseline fold {fold} predictions checksum mismatch")
    if (
        EXPECTED_BASELINE_PREDICTION_SHA256 is not None
        and actual_prediction_sha256 != EXPECTED_BASELINE_PREDICTION_SHA256[fold]
    ):
        raise ValueError(f"baseline fold {fold} is not the exact exp600 seed42 artifact")
    expected_ids_hash = canonical_sha256(expected["id"].astype(str).tolist())
    if contract.get("prediction_ids_sha256") != expected_ids_hash:
        raise ValueError(f"baseline fold {fold} ID checksum mismatch")
    frame = pd.read_csv(predictions_path, dtype={"id": str, "category": str})
    if tuple(frame.columns) != BASELINE_COLUMNS:
        raise ValueError(f"baseline fold {fold} prediction schema/order mismatch")
    if frame["id"].tolist() != expected["id"].tolist():
        raise ValueError(f"baseline fold {fold} IDs/order mismatch")
    if frame["category"].tolist() != expected["category"].tolist():
        raise ValueError(f"baseline fold {fold} categories/order mismatch")
    if not np.array_equal(frame["label"].to_numpy(np.int8), expected["label"].to_numpy(np.int8)):
        raise ValueError(f"baseline fold {fold} labels mismatch")
    if not frame["fold"].astype(int).eq(fold).all():
        raise ValueError(f"baseline fold {fold} identity mismatch")
    if not np.isfinite(frame["lora_score"].to_numpy(np.float64)).all():
        raise ValueError(f"baseline fold {fold} scores are non-finite")
    return frame, {
        "contract_sha256": sha256_file(contract_path),
        "predictions_sha256": actual_prediction_sha256,
    }


def freeze_donor_thresholds(
    *, registry_path: Path, baseline_dirs: Mapping[int, Path], output_path: Path
) -> dict[str, Any]:
    """Write thresholds from fixed exp600 donors, without any candidate input."""

    if output_path.exists():
        raise FileExistsError("refusing to overwrite frozen donor thresholds")
    if set(baseline_dirs) != set(FOLDS):
        raise ValueError("exactly baseline folds 0..4 are required")
    registry = _registry(registry_path)
    frames: dict[int, pd.DataFrame] = {}
    provenance: dict[str, Any] = {}
    for fold in FOLDS:
        frames[fold], provenance[str(fold)] = _load_baseline_fold(
            baseline_dirs[fold], fold=fold, expected=_expected_fold(registry, fold)
        )
    donor = pd.concat([frames[fold] for fold in FOLDS], ignore_index=True)
    thresholds: dict[str, dict[str, Any]] = {}
    for target_fold in SCREEN_FOLDS:
        thresholds[str(target_fold)] = {}
        for category in CATEGORIES:
            selected = donor.loc[
                donor["category"].eq(category) & ~donor["fold"].astype(int).eq(target_fold)
            ]
            donor_f1, threshold = _best_threshold(
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
        "registry_sha256": sha256_file(registry_path),
        "baseline_component": "original",
        "baseline_seed": 42,
        "screen_folds": list(SCREEN_FOLDS),
        "calibration": "leave_target_fold_out_over_fixed_exp600_original_oof",
        "candidate_inputs_read": 0,
        "sealed_rows": 0,
        "baseline_provenance": provenance,
        "thresholds": thresholds,
    }
    result["threshold_contract_sha256"] = canonical_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _verify_threshold_contract(
    *, contract_path: Path, registry_path: Path, baseline_dirs: Mapping[int, Path]
) -> tuple[dict[str, Any], dict[int, pd.DataFrame]]:
    contract = _load_json(contract_path)
    legacy_pre_result_bundle = (
        contract.get("protocol") == "exp621_original_seed42_leave_one_fold_out_thresholds_v1"
    )
    if not legacy_pre_result_bundle:
        _verify_self_contract(contract, "threshold_contract_sha256")
    expected_top = {
        "protocol": "621_exp600_seed42_donor_thresholds_v1",
        "registry_sha256": sha256_file(registry_path),
        "baseline_component": "original",
        "baseline_seed": 42,
        "screen_folds": list(SCREEN_FOLDS),
        "candidate_inputs_read": 0,
        "sealed_rows": 0,
    }
    if legacy_pre_result_bundle:
        legacy_expected = {
            "frozen_before_candidate_results": True,
            "screen_folds": list(SCREEN_FOLDS),
            "sealed_holdout_used": False,
            "algorithm": (
                "700 unique quantiles from 0.002 through 0.998; maximize donor F1 "
                "with score >= threshold; first maximum wins"
            ),
        }
        for key, value in legacy_expected.items():
            if contract.get(key) != value:
                raise ValueError(f"frozen threshold contract mismatch: {key}")
    else:
        for key, value in expected_top.items():
            if contract.get(key) != value:
                raise ValueError(f"frozen threshold contract mismatch: {key}")
    registry = _registry(registry_path)
    frames: dict[int, pd.DataFrame] = {}
    for fold in FOLDS:
        frames[fold], hashes = _load_baseline_fold(
            baseline_dirs[fold], fold=fold, expected=_expected_fold(registry, fold)
        )
        if legacy_pre_result_bundle:
            frozen_hash = contract.get("source_prediction_sha256", {}).get(str(fold))
            if frozen_hash != hashes["predictions_sha256"]:
                raise ValueError(f"frozen threshold donor provenance mismatch: fold {fold}")
        elif contract.get("baseline_provenance", {}).get(str(fold)) != hashes:
            raise ValueError(f"frozen threshold donor provenance mismatch: fold {fold}")
    # Recompute from donors and require bit-exact frozen values. Candidate files
    # are deliberately not accepted by this function.
    donor = pd.concat([frames[fold] for fold in FOLDS], ignore_index=True)
    for target_fold in SCREEN_FOLDS:
        for category in CATEGORIES:
            selected = donor.loc[
                donor["category"].eq(category) & ~donor["fold"].astype(int).eq(target_fold)
            ]
            donor_f1, threshold = _best_threshold(
                selected["label"].to_numpy(np.int8),
                selected["lora_score"].to_numpy(np.float64),
            )
            if legacy_pre_result_bundle:
                path = contract_path.with_name(f"fold{target_fold}.json")
                frozen_file = _load_json(path)
                frozen = {
                    "threshold": frozen_file.get(category),
                    "donor_f1": contract.get("donor_metrics", {})
                    .get(str(target_fold), {})
                    .get(category, {})
                    .get("f1"),
                    "donor_rows": contract.get("donor_metrics", {})
                    .get(str(target_fold), {})
                    .get(category, {})
                    .get("rows"),
                    "excluded_target_fold": target_fold,
                }
            else:
                frozen = contract.get("thresholds", {}).get(str(target_fold), {}).get(category)
            expected = {
                "threshold": threshold,
                "donor_f1": donor_f1,
                "donor_rows": len(selected),
                "excluded_target_fold": target_fold,
            }
            if frozen != expected:
                raise ValueError(f"frozen threshold value mismatch: fold {target_fold}/{category}")
    if legacy_pre_result_bundle:
        normalized = dict(contract)
        normalized["thresholds"] = {
            str(fold): {
                category: {
                    "threshold": _load_json(contract_path.with_name(f"fold{fold}.json"))[category]
                }
                for category in CATEGORIES
            }
            for fold in SCREEN_FOLDS
        }
        contract = normalized
    return contract, frames


def _load_candidate_fold(
    directory: Path, *, fold: int, expected: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    predictions_path = directory / "validation_predictions.csv"
    contract_path = directory / "fold_contract.json"
    manifest_path = directory / "merged_adapter" / "merge_manifest.json"
    contract = _load_json(contract_path)
    manifest = _load_json(manifest_path)
    expected_contract = {
        "protocol": "621_semantic_v3_fold_runtime_v1",
        "outer_fold": fold,
        "validation_rows": len(expected),
        "validation_labels_loaded": False,
        "sealed_labels_loaded": False,
        "sealed_rows_loaded": 0,
        "decision": "READY_FOR_FROZEN_EXTERNAL_EVALUATION",
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise ValueError(f"candidate fold {fold} contract mismatch: {key}")
    expected_manifest = {
        "protocol": "621_train_only_fisher_v1",
        "outer_fold": fold,
        "validation_rows": len(expected),
        "sealed_rows": 0,
        "validation_labels_loaded": False,
        "sealed_labels_loaded": False,
        "decision": "READY_FOR_EXTERNAL_FOLD_SCORING",
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"candidate fold {fold} manifest mismatch: {key}")
    if contract.get("predictions_sha256") != sha256_file(predictions_path):
        raise ValueError(f"candidate fold {fold} predictions checksum mismatch")
    if contract.get("merge_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError(f"candidate fold {fold} manifest checksum mismatch")
    if manifest.get("validation_ids_sha256") != canonical_sha256(expected["id"].tolist()):
        raise ValueError(f"candidate fold {fold} validation ID checksum mismatch")
    frame = pd.read_csv(predictions_path, dtype={"id": str, "category": str})
    if "label" in frame.columns:
        raise ValueError("candidate predictions must remain label-free")
    if tuple(frame.columns) != CANDIDATE_COLUMNS:
        raise ValueError(f"candidate fold {fold} prediction schema/order mismatch")
    if frame["id"].tolist() != expected["id"].tolist():
        raise ValueError(f"candidate fold {fold} IDs/order mismatch")
    if frame["category"].tolist() != expected["category"].tolist():
        raise ValueError(f"candidate fold {fold} categories/order mismatch")
    if not frame["fold"].astype(int).eq(fold).all():
        raise ValueError(f"candidate fold {fold} identity mismatch")
    if not np.isfinite(frame["lora_score"].to_numpy(np.float64)).all():
        raise ValueError(f"candidate fold {fold} scores are non-finite")
    return frame, {
        "contract_sha256": sha256_file(contract_path),
        "manifest_sha256": sha256_file(manifest_path),
        "predictions_sha256": sha256_file(predictions_path),
    }


def evaluate_screen(
    *,
    registry_path: Path,
    baseline_dirs: Mapping[int, Path],
    candidate_dirs: Mapping[int, Path],
    threshold_contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite screen report")
    if set(baseline_dirs) != set(FOLDS):
        raise ValueError("exactly baseline folds 0..4 are required")
    if set(candidate_dirs) != set(SCREEN_FOLDS):
        raise ValueError("screen requires exactly candidate folds 0 and 3")
    # Verify all immutable donor inputs and the frozen threshold contract before
    # opening either candidate directory.
    thresholds, baseline_frames = _verify_threshold_contract(
        contract_path=threshold_contract_path,
        registry_path=registry_path,
        baseline_dirs=baseline_dirs,
    )
    registry = _registry(registry_path)
    candidate_frames: dict[int, pd.DataFrame] = {}
    candidate_provenance: dict[str, Any] = {}
    for fold in SCREEN_FOLDS:
        candidate_frames[fold], candidate_provenance[str(fold)] = _load_candidate_fold(
            candidate_dirs[fold], fold=fold, expected=_expected_fold(registry, fold)
        )

    rows: list[pd.DataFrame] = []
    fold_metrics: dict[str, Any] = {}
    for fold in SCREEN_FOLDS:
        expected = _expected_fold(registry, fold)
        baseline = baseline_frames[fold]
        candidate = candidate_frames[fold]
        assembled = expected.copy()
        assembled["baseline_score"] = baseline["lora_score"].to_numpy(np.float64)
        assembled["candidate_score"] = candidate["lora_score"].to_numpy(np.float64)
        baseline_pred = np.zeros(len(assembled), dtype=np.int8)
        candidate_pred = np.zeros(len(assembled), dtype=np.int8)
        categories: dict[str, Any] = {}
        for category in CATEGORIES:
            mask = assembled["category"].eq(category).to_numpy()
            threshold = float(thresholds["thresholds"][str(fold)][category]["threshold"])
            baseline_pred[mask] = (assembled.loc[mask, "baseline_score"] >= threshold).astype(
                np.int8
            )
            candidate_pred[mask] = (assembled.loc[mask, "candidate_score"] >= threshold).astype(
                np.int8
            )
            labels = assembled.loc[mask, "label"].to_numpy(np.int8)
            base_f1 = _f1(labels, baseline_pred[mask])
            cand_f1 = _f1(labels, candidate_pred[mask])
            categories[category] = {
                "baseline_f1": base_f1,
                "candidate_f1": cand_f1,
                "delta": cand_f1 - base_f1,
                "threshold": threshold,
            }
        assembled["baseline_prediction"] = baseline_pred
        assembled["candidate_prediction"] = candidate_pred
        baseline_macro = float(np.mean([categories[c]["baseline_f1"] for c in CATEGORIES]))
        candidate_macro = float(np.mean([categories[c]["candidate_f1"] for c in CATEGORIES]))
        fold_metrics[str(fold)] = {
            "rows": len(assembled),
            "baseline_macro_f1": baseline_macro,
            "candidate_macro_f1": candidate_macro,
            "delta": candidate_macro - baseline_macro,
            "categories": categories,
        }
        assembled["fold"] = fold
        rows.append(assembled)

    combined = pd.concat(rows, ignore_index=True)
    labels = combined["label"].to_numpy(np.int8)
    baseline_pred = combined["baseline_prediction"].to_numpy(np.int8)
    candidate_pred = combined["candidate_prediction"].to_numpy(np.int8)
    corrected = int(((baseline_pred != labels) & (candidate_pred == labels)).sum())
    regressed = int(((baseline_pred == labels) & (candidate_pred != labels)).sum())
    ratio = None if regressed == 0 else corrected / regressed
    category_metrics: dict[str, Any] = {}
    for category in CATEGORIES:
        mask = combined["category"].eq(category).to_numpy()
        base_f1 = _f1(labels[mask], baseline_pred[mask])
        cand_f1 = _f1(labels[mask], candidate_pred[mask])
        category_metrics[category] = {
            "baseline_f1": base_f1,
            "candidate_f1": cand_f1,
            "delta": cand_f1 - base_f1,
        }
    positive = labels == 1
    flammable = combined["category"].eq(FLAMMABLE).to_numpy()
    fn = {
        "flammable": {
            "baseline": int((flammable & positive & (baseline_pred == 0)).sum()),
            "candidate": int((flammable & positive & (candidate_pred == 0)).sum()),
        },
        "all_positive": {
            "baseline": int((positive & (baseline_pred == 0)).sum()),
            "candidate": int((positive & (candidate_pred == 0)).sum()),
        },
    }
    for value in fn.values():
        value["delta"] = value["candidate"] - value["baseline"]
    mean_fold_delta = float(np.mean([fold_metrics[str(f)]["delta"] for f in SCREEN_FOLDS]))
    gates = {
        "each_fold_delta_gt_0": all(fold_metrics[str(f)]["delta"] > 0 for f in SCREEN_FOLDS),
        "mean_fold_delta_at_least_0_0015": mean_fold_delta >= 0.0015,
        "corrected_to_regressed_at_least_1_5": (
            corrected > 0 if regressed == 0 else corrected / regressed >= 1.5
        ),
        "no_category_drop_below_minus_0_002": all(
            category_metrics[c]["delta"] >= -0.002 for c in CATEGORIES
        ),
        "flammable_false_negatives_do_not_increase": fn["flammable"]["delta"] <= 0,
        "all_positive_false_negatives_do_not_increase": fn["all_positive"]["delta"] <= 0,
    }
    passed = all(gates.values())
    result: dict[str, Any] = {
        "protocol": "621_frozen_donor_screen_v1",
        "screen_folds": list(SCREEN_FOLDS),
        "registry_sha256": sha256_file(registry_path),
        "threshold_contract_sha256": sha256_file(threshold_contract_path),
        "thresholds_tuned_after_candidate": False,
        "candidate_predictions_contained_labels": False,
        "sealed_rows_loaded": 0,
        "folds": fold_metrics,
        "mean_fold_delta": mean_fold_delta,
        "categories": category_metrics,
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": ratio,
        "corrected_to_regressed_infinite": regressed == 0 and corrected > 0,
        "false_negatives": fn,
        "gates": gates,
        "passed": passed,
        "decision": "GO_LAUNCH_FOLDS_1_2_4" if passed else "NO_GO_REJECT_621",
        "candidate_provenance": candidate_provenance,
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
    parser = argparse.ArgumentParser(description="Freeze and run the experiment-621 screen.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-thresholds")
    freeze.add_argument("--registry", required=True, type=Path)
    freeze.add_argument("--baseline-fold", action="append", required=True)
    freeze.add_argument("--output", required=True, type=Path)
    screen = subparsers.add_parser("screen")
    screen.add_argument("--registry", required=True, type=Path)
    screen.add_argument("--baseline-fold", action="append", required=True)
    screen.add_argument("--candidate-fold", action="append", required=True)
    screen.add_argument("--threshold-contract", required=True, type=Path)
    screen.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baselines = _fold_paths(args.baseline_fold, required=FOLDS)
    if args.command == "freeze-thresholds":
        result = freeze_donor_thresholds(
            registry_path=args.registry, baseline_dirs=baselines, output_path=args.output
        )
    else:
        candidates = _fold_paths(args.candidate_fold, required=SCREEN_FOLDS)
        result = evaluate_screen(
            registry_path=args.registry,
            baseline_dirs=baselines,
            candidate_dirs=candidates,
            threshold_contract_path=args.threshold_contract,
            output_path=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

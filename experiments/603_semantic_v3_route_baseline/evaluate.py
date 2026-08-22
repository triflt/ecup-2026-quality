from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_FOLDS = ROOT / "validation" / "semantic_family_v3" / "folds.csv"
DEFAULT_PROTOCOL = HERE / "frozen_protocol.json"
EXPECTED_FOLDS_SHA256 = "16b9c47999c6c1e97b1317182adc356931db60a1156ec237fa496fa48c5387ae"
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (42, 31415, 271828, 161803)
NEW_SEEDS = (31415, 271828, 161803)
COMPONENTS = ("original", "specialist")
SAFE_FOLD_COLUMNS = (
    "id",
    "category",
    "label",
    "semantic_component",
    "split",
    "development_fold",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Reproduce the exact locked-190 threshold routine, including tie behavior."""

    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered = labels[order]
    true_positive = np.cumsum(ordered == 1)
    false_positive = np.cumsum(ordered == 0)
    false_negative = int((labels == 1).sum()) - true_positive
    values = 2 * true_positive / np.maximum(1, 2 * true_positive + false_positive + false_negative)
    best = int(np.argmax(values))
    if best + 1 < len(scores):
        threshold = float((scores[order[best]] + scores[order[best + 1]]) / 2)
    else:
        threshold = float(scores[order[best]] - 1e-7)
    return float(values[best]), threshold


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def fold_category_ranks(
    values: np.ndarray, folds: np.ndarray, categories: np.ndarray
) -> np.ndarray:
    output = np.empty(len(values), dtype=np.float32)
    for fold in FOLDS:
        for category in sorted(np.unique(categories)):
            mask = (folds == fold) & (categories == category)
            output[mask] = rank01(np.asarray(values)[mask])
    return output


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _verify_self_contract(contract: dict[str, Any]) -> None:
    payload = dict(contract)
    observed = payload.pop("contract_sha256", None)
    if observed != canonical_sha256(payload):
        raise ValueError("runtime contract self-checksum mismatch")


def _require_sha256_fields(contract: dict[str, Any], fields: tuple[str, ...]) -> None:
    hexadecimal = set("0123456789abcdef")
    for field in fields:
        value = contract.get(field)
        if not isinstance(value, str) or len(value) != 64 or not set(value) <= hexadecimal:
            raise ValueError(f"runtime contract has invalid SHA-256 field: {field}")


def _verify_file(directory: Path, name: str, expected_sha256: str) -> None:
    path = directory / name
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"artifact checksum mismatch for {name}")


def load_visual_bundle(
    *, bundle_path: Path, contract_path: Path, protocol: dict[str, Any]
) -> dict[str, np.ndarray]:
    contract = _load_json(contract_path)
    expected = {
        "version": "semantic_v3_visual_base_components_v1",
        "status": "complete",
        "development_rows": protocol["development_rows"],
        "sealed_rows_in_outputs": 0,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"visual component contract mismatch for {key}")
    if contract.get("output_sha256", {}).get(bundle_path.name) != sha256_file(bundle_path):
        raise ValueError("visual component bundle checksum mismatch")
    required = {
        "ids",
        "labels",
        "categories",
        "folds",
        "semantic_components",
        "robust_base_rank",
        "qwen3vl_rank",
    }
    with np.load(bundle_path, allow_pickle=False) as bundle:
        missing = required - set(bundle.files)
        if missing:
            raise ValueError(f"visual bundle lacks arrays: {sorted(missing)}")
        output = {key: np.asarray(bundle[key]) for key in required}
    ids = output["ids"].astype(str)
    if len(ids) != protocol["development_rows"] or len(set(ids)) != len(ids):
        raise ValueError("visual bundle has invalid development IDs")
    if set(output["folds"].astype(int)) != set(FOLDS):
        raise ValueError("visual bundle folds are not exactly 0..4")
    for key in ("robust_base_rank", "qwen3vl_rank"):
        if not np.isfinite(output[key]).all():
            raise ValueError(f"visual bundle {key} contains non-finite values")
    return output


def load_registry(
    path: Path, visual: dict[str, np.ndarray], *, enforce_frozen: bool = True
) -> pd.DataFrame:
    if enforce_frozen and sha256_file(path) != EXPECTED_FOLDS_SHA256:
        raise ValueError("semantic-v3 fold registry checksum mismatch")
    registry = pd.read_csv(
        path,
        usecols=list(SAFE_FOLD_COLUMNS),
        dtype={"id": str, "semantic_component": str},
    )
    development = registry.loc[registry["split"].eq("development")].reset_index(drop=True)
    checks = {
        "id": visual["ids"].astype(str),
        "category": visual["categories"].astype(str),
        "label": visual["labels"].astype(np.int8),
        "semantic_component": visual["semantic_components"].astype(str),
        "development_fold": visual["folds"].astype(np.int8),
    }
    for key, expected in checks.items():
        actual = development[key].to_numpy(dtype=expected.dtype)
        if not np.array_equal(actual, expected):
            raise ValueError(f"visual bundle differs from semantic-v3 registry: {key}")
    return development


def _load_prediction_directory(
    *,
    directory: Path,
    experiment: str,
    fold: int,
    ids: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
) -> tuple[np.ndarray, dict[str, str]]:
    if experiment == "600":
        contract_name = "output_contract.runtime.json"
        expected_protocol = "semantic_family_v3"
    else:
        contract_name = "seed_output_contract.runtime.json"
        expected_protocol = "semantic_family_v3_seed_variance_v1"
    contract_path = directory / contract_name
    prediction_path = directory / "lora_holdout_predictions.csv"
    contract = _load_json(contract_path)
    _verify_self_contract(contract)
    expected_contract = {
        "experiment_id": experiment,
        "outer_fold": fold,
        "protocol_version": expected_protocol,
        "decision": "GO",
        "sealed_rows_in_predictions": 0,
        "prediction_rows": len(ids),
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise ValueError(f"experiment-{experiment} contract mismatch for {key}")
    if experiment == "600":
        _require_sha256_fields(
            contract,
            (
                "predictions_sha256",
                "adapter_sha256",
                "parent_report_sha256",
                "selection_audit_sha256",
                "protocol_audit_sha256",
                "prediction_ids_sha256",
            ),
        )
        if (
            contract.get("sealed_rows_used_for_threshold") != 0
            or contract.get("sealed_rows_used_for_evaluation") != 0
        ):
            raise ValueError("experiment-600 contract used sealed rows")
        if contract["prediction_ids_sha256"] != canonical_sha256(ids.astype(str).tolist()):
            raise ValueError("experiment-600 prediction ID checksum mismatch")
        _verify_file(directory, "lora_holdout_report.json", contract["parent_report_sha256"])
        _verify_file(directory, "selection_audit.runtime.json", contract["selection_audit_sha256"])
    else:
        _require_sha256_fields(
            contract,
            (
                "predictions_sha256",
                "adapter_sha256",
                "selection_audit_sha256",
                "protocol_input_audit_sha256",
            ),
        )
        _verify_file(
            directory,
            "seed_selection_audit.runtime.json",
            contract["selection_audit_sha256"],
        )
    _verify_file(directory, "adapter.zip", contract["adapter_sha256"])
    _verify_file(directory, prediction_path.name, contract["predictions_sha256"])

    frame = pd.read_csv(prediction_path, dtype={"id": str})
    if set(frame.columns) != {"id", "category", "label", "fold", "lora_score"}:
        raise ValueError("prediction CSV schema mismatch")
    if not np.array_equal(frame["id"].astype(str).to_numpy(), ids.astype(str)):
        raise ValueError("prediction IDs/order differ from the expected outer fold")
    if not np.array_equal(frame["label"].to_numpy(np.int8), labels.astype(np.int8)):
        raise ValueError("prediction labels differ from the visual/registry contract")
    if not np.array_equal(frame["category"].astype(str).to_numpy(), categories.astype(str)):
        raise ValueError("prediction categories differ from the visual/registry contract")
    if not frame["fold"].astype(int).eq(fold).all():
        raise ValueError("prediction fold mismatch")
    logits = frame["lora_score"].to_numpy(np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("prediction logits contain non-finite values")
    return logits, {
        "contract_sha256": sha256_file(contract_path),
        "predictions_sha256": sha256_file(prediction_path),
        "adapter_sha256": sha256_file(directory / "adapter.zip"),
    }


def _parse_grid(
    specifications: list[list[str]],
    *,
    required: set[tuple[str, int]],
    name: str,
) -> dict[tuple[str, int], Path]:
    observed: dict[tuple[str, int], Path] = {}
    for raw_key, raw_fold, raw_directory in specifications:
        key = (str(raw_key), int(raw_fold))
        if key not in required or key in observed:
            raise ValueError(f"{name} entries must be unique members of the frozen grid")
        observed[key] = Path(raw_directory).resolve()
    if set(observed) != required:
        raise ValueError(f"incomplete {name} grid: missing {sorted(required - set(observed))}")
    return observed


def load_oof_grids(
    *,
    exp600_specifications: list[list[str]],
    exp602_specifications: list[list[str]],
    visual: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    grid600 = _parse_grid(
        exp600_specifications,
        required={(component, fold) for component in COMPONENTS for fold in FOLDS},
        name="experiment-600",
    )
    grid602 = _parse_grid(
        exp602_specifications,
        required={(str(seed), fold) for seed in NEW_SEEDS for fold in FOLDS},
        name="experiment-602",
    )
    ids = visual["ids"].astype(str)
    labels = visual["labels"].astype(np.int8)
    categories = visual["categories"].astype(str)
    folds = visual["folds"].astype(np.int8)
    outputs600 = {component: np.empty(len(ids), np.float64) for component in COMPONENTS}
    outputs602 = {seed: np.empty(len(ids), np.float64) for seed in NEW_SEEDS}
    provenance: dict[str, Any] = {"experiment_600": {}, "experiment_602": {}}
    for component in COMPONENTS:
        for fold in FOLDS:
            mask = folds == fold
            logits, hashes = _load_prediction_directory(
                directory=grid600[(component, fold)],
                experiment="600",
                fold=fold,
                ids=ids[mask],
                labels=labels[mask],
                categories=categories[mask],
            )
            contract = _load_json(grid600[(component, fold)] / "output_contract.runtime.json")
            if contract.get("component") != component:
                raise ValueError("experiment-600 component identity mismatch")
            outputs600[component][mask] = logits
            provenance["experiment_600"][f"{component}_fold_{fold}"] = hashes
    for seed in NEW_SEEDS:
        for fold in FOLDS:
            mask = folds == fold
            logits, hashes = _load_prediction_directory(
                directory=grid602[(str(seed), fold)],
                experiment="602",
                fold=fold,
                ids=ids[mask],
                labels=labels[mask],
                categories=categories[mask],
            )
            contract = _load_json(grid602[(str(seed), fold)] / "seed_output_contract.runtime.json")
            if contract.get("seed") != seed:
                raise ValueError("experiment-602 seed identity mismatch")
            outputs602[seed][mask] = logits
            provenance["experiment_602"][f"seed_{seed}_fold_{fold}"] = hashes
    for values in (*outputs600.values(), *outputs602.values()):
        if not np.isfinite(values).all():
            raise ValueError("assembled OOF grid is incomplete or non-finite")
    return outputs600, outputs602, provenance


def evaluate_route(
    *,
    name: str,
    bad_rank: np.ndarray,
    flammable_rank: np.ndarray,
    visual: dict[str, np.ndarray],
    route: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = visual["labels"].astype(np.int8)
    categories = visual["categories"].astype(str)
    folds = visual["folds"].astype(np.int8)
    scores = np.empty(len(labels), dtype=np.float32)
    rank_by_category = {"БАД": bad_rank, "Легковоспламеняющиеся": flammable_rank}
    predictions = np.zeros(len(labels), dtype=np.int8)
    calibration: dict[str, list[dict[str, Any]]] = {}
    for category in sorted(np.unique(categories)):
        weights = route[category]["weights"]
        mask = categories == category
        scores[mask] = (
            weights["robust_base_rank"] * visual["robust_base_rank"][mask]
            + weights["qwen3vl_rank"] * visual["qwen3vl_rank"][mask]
            + weights["qwen35_rank"] * rank_by_category[category][mask]
        )
        calibration[category] = []
        for fold in FOLDS:
            train = mask & (folds != fold)
            valid = mask & (folds == fold)
            train_f1, threshold = best_threshold(labels[train], scores[train])
            predictions[valid] = (scores[valid] >= threshold).astype(np.int8)
            calibration[category].append(
                {
                    "fold": fold,
                    "threshold": threshold,
                    "calibration_rows": int(train.sum()),
                    "validation_rows": int(valid.sum()),
                    "calibration_f1": train_f1,
                }
            )
    category_f1 = {
        category: f1(labels[categories == category], predictions[categories == category])
        for category in sorted(np.unique(categories))
    }
    fold_macro = {}
    for fold in FOLDS:
        values = [
            f1(
                labels[(folds == fold) & (categories == category)],
                predictions[(folds == fold) & (categories == category)],
            )
            for category in sorted(np.unique(categories))
        ]
        fold_macro[str(fold)] = float(np.mean(values))
    return predictions, {
        "name": name,
        "macro_f1": float(np.mean(list(category_f1.values()))),
        "category_f1": category_f1,
        "fold_macro_f1": fold_macro,
        "leave_one_fold_out_calibration": calibration,
    }


def _paired_change(
    *, labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    changed = baseline != candidate
    corrected = int((changed & (candidate == labels) & (baseline != labels)).sum())
    regressed = int((changed & (candidate != labels) & (baseline == labels)).sum())
    return {
        "changed": int(changed.sum()),
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": None if regressed == 0 else corrected / regressed,
    }


def evaluate(
    *,
    visual_bundle_path: Path,
    visual_contract_path: Path,
    folds_path: Path,
    exp600_specifications: list[list[str]],
    exp602_specifications: list[list[str]],
    protocol_path: Path,
    output_path: Path,
    local_predictions_path: Path | None = None,
    enforce_frozen: bool = True,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite an existing public summary")
    if local_predictions_path is not None and local_predictions_path.exists():
        raise FileExistsError("refusing to overwrite existing local predictions")
    protocol = _load_json(protocol_path)
    if protocol.get("version") != "semantic_v3_route_baseline_v1":
        raise ValueError("frozen protocol version mismatch")
    visual = load_visual_bundle(
        bundle_path=visual_bundle_path,
        contract_path=visual_contract_path,
        protocol=protocol,
    )
    load_registry(folds_path, visual, enforce_frozen=enforce_frozen)
    outputs600, outputs602, provenance = load_oof_grids(
        exp600_specifications=exp600_specifications,
        exp602_specifications=exp602_specifications,
        visual=visual,
    )
    categories = visual["categories"].astype(str)
    folds = visual["folds"].astype(np.int8)
    original_probability = sigmoid(outputs600["original"])
    specialist_probability = sigmoid(outputs600["specialist"])
    seed_probabilities = {
        42: original_probability,
        **{seed: sigmoid(outputs602[seed]) for seed in NEW_SEEDS},
    }
    original_rank = fold_category_ranks(original_probability, folds, categories)
    specialist_rank = fold_category_ranks(specialist_probability, folds, categories)
    fixed_mean_probability = np.mean(
        np.stack([seed_probabilities[seed] for seed in SEEDS], axis=0), axis=0
    )
    fixed_mean_rank = fold_category_ranks(fixed_mean_probability, folds, categories)

    baseline_predictions, baseline = evaluate_route(
        name="original_route_baseline",
        bad_rank=original_rank,
        flammable_rank=original_rank,
        visual=visual,
        route=protocol["route"],
    )
    category_predictions, category_route = evaluate_route(
        name="category_routed_specialist_400",
        bad_rank=original_rank,
        flammable_rank=specialist_rank,
        visual=visual,
        route=protocol["route"],
    )
    seed_predictions, four_seed = evaluate_route(
        name="fixed_four_seed_probability_mean_route",
        bad_rank=fixed_mean_rank,
        flammable_rank=fixed_mean_rank,
        visual=visual,
        route=protocol["route"],
    )
    labels = visual["labels"].astype(np.int8)
    category_route["delta_vs_original"] = category_route["macro_f1"] - baseline["macro_f1"]
    category_route["paired_change_vs_original"] = _paired_change(
        labels=labels, baseline=baseline_predictions, candidate=category_predictions
    )
    four_seed["delta_vs_original"] = four_seed["macro_f1"] - baseline["macro_f1"]
    four_seed["paired_change_vs_original"] = _paired_change(
        labels=labels, baseline=baseline_predictions, candidate=seed_predictions
    )

    expected = protocol["expected_macro_f1"]
    tolerance = float(protocol["maximum_reference_absolute_error"])
    observed = {
        "original_route_baseline": baseline["macro_f1"],
        "category_routed_specialist_400": category_route["macro_f1"],
        "fixed_four_seed_probability_mean_route": four_seed["macro_f1"],
    }
    reference_checks = {
        name: {
            "expected": float(expected[name]),
            "observed": float(value),
            "absolute_error": abs(float(value) - float(expected[name])),
            "passed": abs(float(value) - float(expected[name])) <= tolerance,
        }
        for name, value in observed.items()
    }
    if not all(item["passed"] for item in reference_checks.values()):
        raise AssertionError(f"frozen route reference mismatch: {reference_checks}")

    local_sha256 = None
    if local_predictions_path is not None:
        local_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            local_predictions_path,
            ids=visual["ids"].astype(str),
            labels=labels,
            categories=categories,
            folds=folds,
            semantic_components=visual["semantic_components"].astype(str),
            original_route_predictions=baseline_predictions,
            category_routed_specialist_predictions=category_predictions,
            fixed_four_seed_predictions=seed_predictions,
            fixed_four_seed_probability=fixed_mean_probability,
        )
        local_sha256 = sha256_file(local_predictions_path)

    summary = {
        "experiment_id": "603",
        "status": "complete",
        "protocol_version": protocol["version"],
        "validation_version": "semantic_family_v3",
        "development_rows": len(labels),
        "sealed_holdout_used": False,
        "gpu_jobs_launched": 0,
        "weights_tuned": False,
        "threshold_calibration": "leave_one_fold_out_over_fixed_oof_scores",
        "fully_nested_meta_validation": False,
        "validation_limitation": protocol["validation_limitation"],
        "comparisons_frozen": protocol["comparisons"],
        "results": {
            "original_route_baseline": baseline,
            "category_routed_specialist_400": category_route,
            "fixed_four_seed_probability_mean_route": four_seed,
        },
        "reference_checks": reference_checks,
        "input_sha256": {
            "frozen_protocol": sha256_file(protocol_path),
            "semantic_v3_folds": sha256_file(folds_path),
            "visual_bundle": sha256_file(visual_bundle_path),
            "visual_contract": sha256_file(visual_contract_path),
            **provenance,
        },
        "local_predictions_sha256": local_sha256,
        "public_summary_contains_local_paths_or_job_names": False,
        "decision": "REPRODUCED_FROZEN_COMPARISONS",
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the three frozen semantic-v3 route baselines."
    )
    parser.add_argument("--visual-bundle", type=Path, required=True)
    parser.add_argument("--visual-contract", type=Path, required=True)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--exp600-fold-output",
        action="append",
        nargs=3,
        metavar=("COMPONENT", "FOLD", "DIRECTORY"),
        default=[],
    )
    parser.add_argument(
        "--exp602-seed-output",
        action="append",
        nargs=3,
        metavar=("SEED", "FOLD", "DIRECTORY"),
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-predictions", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(
        visual_bundle_path=args.visual_bundle.resolve(),
        visual_contract_path=args.visual_contract.resolve(),
        folds_path=args.folds.resolve(),
        exp600_specifications=args.exp600_fold_output,
        exp602_specifications=args.exp602_seed_output,
        protocol_path=args.protocol.resolve(),
        output_path=args.output.resolve(),
        local_predictions_path=(
            None if args.local_predictions is None else args.local_predictions.resolve()
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

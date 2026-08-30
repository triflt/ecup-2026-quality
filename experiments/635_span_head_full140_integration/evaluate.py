"""Fail-closed full-system-140 integration evaluator for experiment 635.

This module never evaluates the span component in isolation.  It reconstructs
the frozen decision route of experiment 140, applies the same donor-only prior
override to every variant, and compares complete-system predictions only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC_PATH = HERE / "frozen_spec.json"
DEFAULT_REGISTRY = ROOT / "validation/semantic_family_v3/folds.csv"

FOLDS = (0, 1, 2, 3, 4)
SCREEN_FOLDS = (0, 3)
CATEGORIES = ("БАД", "Легковоспламеняющиеся")
FLAMMABLE = "Легковоспламеняющиеся"
BASELINE = "original_qwen35"
CANDIDATES = (
    "replace_with_632",
    "fixed_mean_original_632",
    "evidence_gated_632",
)
CONCEPTS = (
    "OBJECT_OF_SALE",
    "COMPOSITION",
    "COMPLETENESS",
    "FUEL_OR_IGNITION",
    "NEGATION",
)
REQUIRED_ARRAYS = (
    "ids",
    "categories",
    "folds",
    "semantic_components",
    "robust_base_score",
    "qwen3vl_score",
    "qwen35_original_logit",
    "qwen35_seed632_logit",
    "seed632_evidence",
    "seed632_concept",
    "seed632_char_start",
    "seed632_char_end",
    "canonical_text",
    "prior_override",
    "prior_source",
)
FORBIDDEN_ARRAY_TOKENS = ("label", "target", "ground_truth", "sealed")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _verify_self_hash(value: Mapping[str, Any], key: str) -> None:
    observed = value.get(key)
    if not isinstance(observed, str) or not SHA256_RE.fullmatch(observed):
        raise ValueError(f"missing or malformed {key}")
    payload = {name: item for name, item in value.items() if name != key}
    if canonical_sha256(payload) != observed:
        raise ValueError(f"{key} mismatch")


def load_spec() -> dict[str, Any]:
    spec = _json(SPEC_PATH)
    expected = {
        "schema_version": "exp635_full140_integration_v1",
        "experiment_id": "635",
        "parent_full_system": "140",
        "upstream_component_experiment": "632",
        "upstream_seed": 31415,
        "validation": "semantic_family_v3",
        "screen_folds": list(SCREEN_FOLDS),
        "full_folds": list(FOLDS),
        "uses_public_for_selection": False,
        "sealed_rows_allowed": 0,
        "component_f1_compared_to_full_ensemble": False,
        "experiment_603_allowed_as_parent": False,
        "variants": [BASELINE, *CANDIDATES],
        "rank_transform": "stable fold-category rank01",
        "postprocess": "same donor-only exact/name override for every variant",
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"frozen spec mismatch: {key}")
    return spec


def verify_source_recipe(spec: Mapping[str, Any]) -> None:
    for relative, expected in spec["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen source checksum mismatch: {relative}")


def verify_accepted_632(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    if sha256_file(path) != contract.get("accepted_632_report_sha256"):
        raise ValueError("accepted 632 report checksum mismatch")
    report = _json(path)
    experiment_id = report.get("candidate_experiment_id", report.get("experiment_id"))
    seed = report.get("candidate_seed", report.get("seed", -1))
    if str(experiment_id) != "632" or int(seed) != 31415:
        raise ValueError("accepted report is not experiment 632 seed 31415")
    validation = report.get("full_validation", report)
    if not isinstance(validation, dict):
        raise TypeError("accepted 632 report lacks full validation object")
    folds = validation.get("folds_evaluated")
    if folds is None and isinstance(validation.get("folds"), dict):
        folds = sorted(int(value) for value in validation["folds"])
    passed = validation.get("passed", report.get("passed"))
    wins = validation.get("winning_folds", report.get("winning_folds"))
    sealed = validation.get(
        "sealed_rows_loaded",
        validation.get("sealed_rows_used", report.get("sealed_rows_used", 0)),
    )
    public = validation.get("public_used", report.get("public_used", False))
    if folds != list(FOLDS) or passed is not True or int(wins or -1) < 4:
        raise ValueError("experiment 632 did not pass its frozen five-fold gate")
    if int(sealed) != 0 or public is not False:
        raise ValueError("experiment 632 acceptance used sealed or Public feedback")
    return report


def _verify_component_provenance(contract: Mapping[str, Any]) -> None:
    provenance = contract.get("component_provenance")
    required = (
        "robust_base",
        "qwen3vl",
        "qwen35_original",
        "qwen35_seed632",
        "prior_override",
    )
    if not isinstance(provenance, dict) or set(provenance) != set(required):
        raise ValueError("replay contract component provenance is incomplete")
    for component in required:
        fold_records = provenance[component].get("folds")
        if not isinstance(fold_records, dict) or set(fold_records) != {str(f) for f in FOLDS}:
            raise ValueError(f"component provenance folds mismatch: {component}")
        for fold in FOLDS:
            record = fold_records[str(fold)]
            if record.get("target_fold_excluded") is not True:
                raise ValueError(f"target fold was not excluded: {component}/{fold}")
            artifact_sha = record.get("artifact_sha256")
            if not isinstance(artifact_sha, str) or not SHA256_RE.fullmatch(artifact_sha):
                raise ValueError(f"invalid artifact hash: {component}/{fold}")


def verify_replay_contract(
    *, path: Path, bundle_path: Path, registry_path: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    contract = _json(path)
    _verify_self_hash(contract, "contract_sha256")
    requirements = spec["required_replay_contract"]
    for key, expected in requirements.items():
        if contract.get(key) != expected:
            raise ValueError(f"replay contract mismatch: {key}")
    if contract.get("experiment_id") != "635":
        raise ValueError("replay contract experiment mismatch")
    if contract.get("bundle_sha256") != sha256_file(bundle_path):
        raise ValueError("replay bundle checksum mismatch")
    if contract.get("registry_sha256") != sha256_file(registry_path):
        raise ValueError("replay registry checksum mismatch")
    if contract.get("source_sha256") != spec["source_sha256"]:
        raise ValueError("replay source provenance mismatch")
    if contract.get("route") != spec["route"]:
        raise ValueError("replay route differs from frozen system 140")
    if contract.get("source_experiment_603") is not False:
        raise ValueError("experiment 603 cannot substitute for full system 140")
    _verify_component_provenance(contract)
    return contract


def _load_registry(path: Path, *, spec: Mapping[str, Any]) -> pd.DataFrame:
    if sha256_file(path) != spec["source_sha256"]["validation/semantic_family_v3/folds.csv"]:
        raise ValueError("semantic-v3 registry checksum mismatch")
    frame = pd.read_csv(path, dtype={"id": str, "category": str, "split": str})
    required = {
        "id",
        "category",
        "label",
        "semantic_component",
        "split",
        "development_fold",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"semantic-v3 registry missing columns: {sorted(required - set(frame))}")
    development = frame.loc[frame["split"].eq("development")].copy()
    if len(development) != 11118 or development["id"].duplicated().any():
        raise ValueError("semantic-v3 development registry row contract mismatch")
    if set(development["development_fold"].astype(int)) != set(FOLDS):
        raise ValueError("semantic-v3 development folds mismatch")
    if set(development["category"].astype(str)) != set(CATEGORIES):
        raise ValueError("semantic-v3 categories mismatch")
    return development.reset_index(drop=True)


def _array_schema(bundle: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {"dtype": str(bundle[name].dtype), "shape": list(bundle[name].shape)}
        for name in REQUIRED_ARRAYS
    }


def load_replay_bundle(
    *, path: Path, contract: Mapping[str, Any], registry: pd.DataFrame
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        forbidden = sorted(
            key for key in keys if any(token in key.lower() for token in FORBIDDEN_ARRAY_TOKENS)
        )
        if forbidden:
            raise ValueError(f"label/sealed arrays are forbidden in replay bundle: {forbidden}")
        if keys != set(REQUIRED_ARRAYS):
            raise ValueError(
                f"replay arrays mismatch: missing={sorted(set(REQUIRED_ARRAYS) - keys)} "
                f"extra={sorted(keys - set(REQUIRED_ARRAYS))}"
            )
        bundle = {name: np.asarray(payload[name]) for name in REQUIRED_ARRAYS}
    rows = len(registry)
    if any(values.ndim != 1 or len(values) != rows for values in bundle.values()):
        raise ValueError("every replay array must be one-dimensional and development-sized")
    if any(values.dtype.hasobject for values in bundle.values()):
        raise ValueError("object arrays are forbidden")
    if contract.get("array_schema") != _array_schema(bundle):
        raise ValueError("replay array schema does not match contract")

    expected_ids = registry["id"].astype(str).to_numpy()
    expected_categories = registry["category"].astype(str).to_numpy()
    expected_folds = registry["development_fold"].astype(np.int8).to_numpy()
    expected_components = registry["semantic_component"].astype(str).to_numpy()
    if not np.array_equal(bundle["ids"].astype(str), expected_ids):
        raise ValueError("replay IDs/order differ from semantic-v3 development registry")
    if not np.array_equal(bundle["categories"].astype(str), expected_categories):
        raise ValueError("replay categories differ from semantic-v3 registry")
    if not np.array_equal(bundle["folds"].astype(np.int8), expected_folds):
        raise ValueError("replay folds differ from semantic-v3 registry")
    if not np.array_equal(bundle["semantic_components"].astype(str), expected_components):
        raise ValueError("replay semantic components differ from registry")

    numeric = (
        "robust_base_score",
        "qwen3vl_score",
        "qwen35_original_logit",
        "qwen35_seed632_logit",
    )
    if any(not np.isfinite(bundle[name].astype(np.float64)).all() for name in numeric):
        raise ValueError("non-finite component score")
    prior = bundle["prior_override"].astype(np.int8)
    prior_source = bundle["prior_source"].astype(str)
    if not set(np.unique(prior)).issubset({-1, 0, 1}):
        raise ValueError("prior_override must contain only -1/0/1")
    if not set(np.unique(prior_source)).issubset({"NONE", "EXACT", "NAME"}):
        raise ValueError("unknown prior source")
    if np.any((prior_source == "NONE") != (prior == -1)):
        raise ValueError("prior source and override disagree")
    return bundle


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("rank01 requires a finite vector")
    if len(values) == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    return result


def fold_category_ranks(
    values: np.ndarray, folds: np.ndarray, categories: np.ndarray
) -> np.ndarray:
    output = np.empty(len(values), dtype=np.float64)
    for fold in FOLDS:
        for category in CATEGORIES:
            mask = (folds == fold) & (categories == category)
            if not mask.any():
                raise ValueError(f"empty fold/category rank cell: {fold}/{category}")
            output[mask] = rank01(values[mask])
    return output


def structural_evidence_mask(bundle: Mapping[str, np.ndarray]) -> np.ndarray:
    evidence = bundle["seed632_evidence"].astype(str)
    concept = bundle["seed632_concept"].astype(str)
    starts = bundle["seed632_char_start"].astype(np.int64)
    ends = bundle["seed632_char_end"].astype(np.int64)
    texts = bundle["canonical_text"].astype(str)
    valid = np.zeros(len(evidence), dtype=bool)
    for index, (quote, kind, start, end, text) in enumerate(
        zip(evidence, concept, starts, ends, texts, strict=True)
    ):
        if quote == "NO_EVIDENCE":
            if kind != "NO_EVIDENCE" or start != -1 or end != -1:
                raise ValueError(f"malformed NO_EVIDENCE row: {index}")
            continue
        exact = (
            kind in CONCEPTS
            and 0 <= start < end <= len(text)
            and bool(quote.strip())
            and text[start:end] == quote
        )
        if not exact:
            raise ValueError(f"non-exact rendered evidence row: {index}")
        valid[index] = True
    return valid


def variant_probabilities(
    bundle: Mapping[str, np.ndarray], evidence_valid: np.ndarray
) -> dict[str, np.ndarray]:
    original = sigmoid(bundle["qwen35_original_logit"])
    candidate = sigmoid(bundle["qwen35_seed632_logit"])
    return {
        BASELINE: original,
        "replace_with_632": candidate,
        "fixed_mean_original_632": 0.5 * original + 0.5 * candidate,
        "evidence_gated_632": np.where(evidence_valid, candidate, original),
    }


def route_predictions(
    *,
    bundle: Mapping[str, np.ndarray],
    qwen_probability: np.ndarray,
    route: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    categories = bundle["categories"].astype(str)
    folds = bundle["folds"].astype(np.int8)
    robust_rank = fold_category_ranks(
        bundle["robust_base_score"].astype(np.float64), folds, categories
    )
    qwen3vl_rank = fold_category_ranks(
        bundle["qwen3vl_score"].astype(np.float64), folds, categories
    )
    qwen35_rank = fold_category_ranks(qwen_probability, folds, categories)
    scores = np.empty(len(categories), dtype=np.float64)
    predictions = np.zeros(len(categories), dtype=np.int8)
    for category in CATEGORIES:
        mask = categories == category
        config = route[category]
        weights = config["weights"]
        scores[mask] = (
            weights["robust_base_rank"] * robust_rank[mask]
            + weights["qwen3vl_rank"] * qwen3vl_rank[mask]
            + weights["qwen35_rank"] * qwen35_rank[mask]
        )
        predictions[mask] = (scores[mask] >= float(config["threshold"])).astype(np.int8)
    prior = bundle["prior_override"].astype(np.int8)
    override = prior >= 0
    predictions[override] = prior[override]
    return predictions, scores


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def _confusion_by_component(
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    categories: np.ndarray,
    inverse: np.ndarray,
    component_count: int,
) -> np.ndarray:
    counts = np.zeros((component_count, len(CATEGORIES), 3), dtype=np.int64)
    for category_index, category in enumerate(CATEGORIES):
        local = categories == category
        events = (
            local & (labels == 1) & (predictions == 1),
            local & (labels == 0) & (predictions == 1),
            local & (labels == 1) & (predictions == 0),
        )
        for statistic_index, event in enumerate(events):
            counts[:, category_index, statistic_index] = np.bincount(
                inverse[event], minlength=component_count
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


def grouped_bootstrap(
    *,
    labels: np.ndarray,
    categories: np.ndarray,
    components: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    iterations: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    unique, inverse = np.unique(components.astype(str), return_inverse=True)
    if len(unique) < 2:
        raise ValueError("grouped bootstrap needs at least two semantic components")
    baseline_counts = _confusion_by_component(
        labels=labels,
        predictions=baseline,
        categories=categories,
        inverse=inverse,
        component_count=len(unique),
    )
    candidate_counts = _confusion_by_component(
        labels=labels,
        predictions=candidate,
        categories=categories,
        inverse=inverse,
        component_count=len(unique),
    )
    rng = np.random.default_rng(seed)
    probability = np.full(len(unique), 1.0 / len(unique), dtype=np.float64)
    deltas = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        weights = rng.multinomial(len(unique), probability, size=stop - start)
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


def _metrics(
    *,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    components: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    selected_folds: Sequence[int],
    bootstrap: Mapping[str, Any] | None,
) -> dict[str, Any]:
    selected = np.isin(folds, selected_folds)
    labels = labels[selected]
    categories = categories[selected]
    folds = folds[selected]
    components = components[selected]
    baseline = baseline[selected]
    candidate = candidate[selected]
    category_metrics: dict[str, Any] = {}
    for category in CATEGORIES:
        mask = categories == category
        before = f1(labels[mask], baseline[mask])
        after = f1(labels[mask], candidate[mask])
        category_metrics[category] = {
            "baseline_f1": before,
            "candidate_f1": after,
            "delta": after - before,
        }
    baseline_macro = float(np.mean([category_metrics[c]["baseline_f1"] for c in CATEGORIES]))
    candidate_macro = float(np.mean([category_metrics[c]["candidate_f1"] for c in CATEGORIES]))
    fold_metrics: dict[str, Any] = {}
    for fold in selected_folds:
        by_category: dict[str, Any] = {}
        for category in CATEGORIES:
            mask = (folds == fold) & (categories == category)
            before = f1(labels[mask], baseline[mask])
            after = f1(labels[mask], candidate[mask])
            by_category[category] = {
                "baseline_f1": before,
                "candidate_f1": after,
                "delta": after - before,
            }
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
    result: dict[str, Any] = {
        "baseline_macro_f1": baseline_macro,
        "candidate_macro_f1": candidate_macro,
        "macro_delta": candidate_macro - baseline_macro,
        "mean_fold_delta": float(np.mean([item["delta"] for item in fold_metrics.values()])),
        "winning_folds": sum(item["delta"] > 0 for item in fold_metrics.values()),
        "categories": category_metrics,
        "folds": fold_metrics,
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": ratio,
        "corrected_to_regressed_infinite": regressed == 0 and corrected > 0,
        "false_negatives": false_negatives,
    }
    if bootstrap is not None:
        result["component_bootstrap"] = grouped_bootstrap(
            labels=labels,
            categories=categories,
            components=components,
            baseline=baseline,
            candidate=candidate,
            iterations=int(bootstrap["iterations"]),
            seed=int(bootstrap["seed"]),
            batch_size=int(bootstrap["batch_size"]),
        )
    return result


def _screen_gates(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, bool]:
    ratio = metrics["corrected_to_regressed"]
    return {
        "both_screen_folds_win": metrics["winning_folds"] >= gate["minimum_winning_folds"],
        "mean_macro_delta_at_least_0_0015": metrics["mean_fold_delta"]
        >= gate["minimum_mean_macro_delta"],
        "category_drop_within_guardrail": all(
            values["delta"] >= gate["minimum_category_delta"]
            for values in metrics["categories"].values()
        ),
        "corrected_to_regressed_at_least_1_5": (
            metrics["corrected"] > 0
            if metrics["regressed"] == 0
            else ratio >= gate["minimum_corrected_to_regressed_ratio"]
        ),
        "flammable_false_negatives_do_not_increase": metrics["false_negatives"]["flammable"][
            "delta"
        ]
        <= gate["maximum_flammable_false_negative_increase"],
        "all_positive_false_negatives_do_not_increase": metrics["false_negatives"]["all_positive"][
            "delta"
        ]
        <= gate["maximum_all_positive_false_negative_increase"],
    }


def _verify_runtime_report(
    *, path: Path, bundle_path: Path, replay_contract_path: Path
) -> dict[str, Any]:
    report = _json(path)
    _verify_self_hash(report, "report_sha256")
    expected = {
        "schema_version": "exp635_runtime_smoke_v1",
        "experiment_id": "635",
        "rows": 600,
        "input_schema_valid": True,
        "output_schema_valid": True,
        "route_weights_unchanged": True,
        "route_thresholds_unchanged": True,
        "public_feedback_used": False,
        "sealed_rows_used": 0,
        "bundle_sha256": sha256_file(bundle_path),
        "replay_contract_sha256": sha256_file(replay_contract_path),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"runtime report mismatch: {key}")
    variants = report.get("variants")
    if not isinstance(variants, dict) or set(variants) != {BASELINE, *CANDIDATES}:
        raise ValueError("runtime report must cover all four frozen variants")
    for name, values in variants.items():
        if values.get("adapter_manifest_verified") is not True:
            raise ValueError(f"runtime adapter manifest not verified: {name}")
        if values.get("optimized_predictions_identical") is not True:
            raise ValueError(f"runtime optimization changes predictions: {name}")
        for field in ("projected_public_minutes", "projected_private_minutes"):
            value = values.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid runtime field: {name}/{field}")
    return report


def _full_gates(
    metrics: Mapping[str, Any], gate: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, bool]:
    ratio = metrics["corrected_to_regressed"]
    return {
        "macro_delta_at_least_0_003": metrics["macro_delta"] >= gate["minimum_macro_delta"],
        "folds_won_at_least_4": metrics["winning_folds"] >= gate["minimum_winning_folds"],
        "no_category_drop": all(
            values["delta"] >= gate["minimum_category_delta"]
            for values in metrics["categories"].values()
        ),
        "corrected_to_regressed_at_least_1_5": (
            metrics["corrected"] > 0
            if metrics["regressed"] == 0
            else ratio >= gate["minimum_corrected_to_regressed_ratio"]
        ),
        "component_bootstrap_probability_at_least_0_90": metrics["component_bootstrap"][
            "probability_delta_positive"
        ]
        >= gate["minimum_component_bootstrap_probability_positive"],
        "flammable_false_negatives_do_not_increase": metrics["false_negatives"]["flammable"][
            "delta"
        ]
        <= gate["maximum_flammable_false_negative_increase"],
        "all_positive_false_negatives_do_not_increase": metrics["false_negatives"]["all_positive"][
            "delta"
        ]
        <= gate["maximum_all_positive_false_negative_increase"],
        "public_runtime_within_20_minutes": runtime["projected_public_minutes"]
        <= gate["maximum_public_runtime_minutes"],
        "private_runtime_within_40_minutes": runtime["projected_private_minutes"]
        <= gate["maximum_private_runtime_minutes"],
        "optimized_predictions_identical": runtime["optimized_predictions_identical"] is True,
        "adapter_manifest_verified": runtime["adapter_manifest_verified"] is True,
    }


def _verify_screen_report(
    *,
    path: Path,
    bundle_path: Path,
    replay_contract_path: Path,
    accepted_632_report_path: Path,
) -> list[str]:
    report = _json(path)
    _verify_self_hash(report, "report_sha256")
    expected = {
        "schema_version": "exp635_full140_evaluation_v1",
        "experiment_id": "635",
        "stage": "screen",
        "folds_evaluated": list(SCREEN_FOLDS),
        "bundle_sha256": sha256_file(bundle_path),
        "replay_contract_sha256": sha256_file(replay_contract_path),
        "accepted_632_report_sha256": sha256_file(accepted_632_report_path),
        "sealed_rows_loaded": 0,
        "public_used_for_selection": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"screen report mismatch: {key}")
    accepted = report.get("accepted_variants")
    if not isinstance(accepted, list) or not accepted:
        raise ValueError("no candidate passed the frozen screen")
    if not set(accepted).issubset(set(CANDIDATES)):
        raise ValueError("screen report contains an unknown candidate")
    for name in accepted:
        if not all(report["variants"][name]["gates"].values()):
            raise ValueError(f"screen report accepted a failed candidate: {name}")
    return accepted


def evaluate(
    *,
    stage: str,
    bundle_path: Path,
    replay_contract_path: Path,
    accepted_632_report_path: Path,
    registry_path: Path,
    output_path: Path,
    screen_report_path: Path | None = None,
    runtime_report_path: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite evaluation report")
    if stage not in {"screen", "full"}:
        raise ValueError("stage must be screen or full")
    spec = load_spec()
    verify_source_recipe(spec)
    contract = verify_replay_contract(
        path=replay_contract_path,
        bundle_path=bundle_path,
        registry_path=registry_path,
        spec=spec,
    )
    verify_accepted_632(accepted_632_report_path, contract)
    runtime_report: dict[str, Any] | None = None
    if stage == "screen":
        if screen_report_path is not None or runtime_report_path is not None:
            raise ValueError("screen does not accept prior screen/runtime reports")
        selected_folds: Sequence[int] = SCREEN_FOLDS
        candidate_names: Sequence[str] = CANDIDATES
    else:
        if screen_report_path is None or runtime_report_path is None:
            raise ValueError("full evaluation requires frozen screen and runtime reports")
        candidate_names = _verify_screen_report(
            path=screen_report_path,
            bundle_path=bundle_path,
            replay_contract_path=replay_contract_path,
            accepted_632_report_path=accepted_632_report_path,
        )
        runtime_report = _verify_runtime_report(
            path=runtime_report_path,
            bundle_path=bundle_path,
            replay_contract_path=replay_contract_path,
        )
        selected_folds = FOLDS
    registry = _load_registry(registry_path, spec=spec)
    bundle = load_replay_bundle(path=bundle_path, contract=contract, registry=registry)
    evidence_valid = structural_evidence_mask(bundle)
    probabilities = variant_probabilities(bundle, evidence_valid)
    evaluated_names = (BASELINE, *candidate_names)
    predictions = {
        name: route_predictions(bundle=bundle, qwen_probability=values, route=spec["route"])[0]
        for name, values in probabilities.items()
        if name in evaluated_names
    }
    labels = registry["label"].astype(np.int8).to_numpy()
    categories = bundle["categories"].astype(str)
    folds = bundle["folds"].astype(np.int8)
    components = bundle["semantic_components"].astype(str)

    baseline_metrics = _metrics(
        labels=labels,
        categories=categories,
        folds=folds,
        components=components,
        baseline=predictions[BASELINE],
        candidate=predictions[BASELINE],
        selected_folds=selected_folds,
        bootstrap=None,
    )
    variants: dict[str, Any] = {
        BASELINE: {
            "role": "full_system_control",
            "macro_f1": baseline_metrics["candidate_macro_f1"],
            "folds": baseline_metrics["folds"],
            "categories": baseline_metrics["categories"],
        }
    }
    accepted: list[str] = []
    for name in candidate_names:
        metrics = _metrics(
            labels=labels,
            categories=categories,
            folds=folds,
            components=components,
            baseline=predictions[BASELINE],
            candidate=predictions[name],
            selected_folds=selected_folds,
            bootstrap=spec["bootstrap"] if stage == "full" else None,
        )
        if stage == "screen":
            gates = _screen_gates(metrics, spec["acceptance"]["screen"])
            runtime = None
        else:
            assert runtime_report is not None
            runtime = runtime_report["variants"][name]
            gates = _full_gates(metrics, spec["acceptance"]["full"], runtime)
        passed = all(gates.values())
        if passed:
            accepted.append(name)
        variants[name] = {
            **metrics,
            "runtime": runtime,
            "gates": gates,
            "passed": passed,
        }
    winner = (
        max(accepted, key=lambda name: (variants[name]["candidate_macro_f1"], name))
        if accepted
        else None
    )
    result: dict[str, Any] = {
        "schema_version": "exp635_full140_evaluation_v1",
        "experiment_id": "635",
        "stage": stage,
        "validation": "semantic_family_v3",
        "folds_evaluated": list(selected_folds),
        "parent_full_system": "140",
        "source_experiment_603_used": False,
        "component_f1_compared_to_full_ensemble": False,
        "weights_changed": False,
        "thresholds_changed": False,
        "candidate_parameters_tuned": 0,
        "sealed_rows_loaded": 0,
        "public_used_for_selection": False,
        "bundle_sha256": sha256_file(bundle_path),
        "replay_contract_sha256": sha256_file(replay_contract_path),
        "accepted_632_report_sha256": sha256_file(accepted_632_report_path),
        "screen_report_sha256": None
        if screen_report_path is None
        else sha256_file(screen_report_path),
        "runtime_report_sha256": None
        if runtime_report_path is None
        else sha256_file(runtime_report_path),
        "structural_exact_evidence": {
            "valid_rows": int(evidence_valid.sum()),
            "coverage": float(evidence_valid.mean()),
            "human_quality_evaluated": False,
        },
        "variants": variants,
        "accepted_variants": accepted,
        "winner": winner,
        "passed": bool(accepted),
        "decision": (
            "GO_FULL_ONLY_FOR_ACCEPTED_SCREEN_VARIANTS"
            if stage == "screen" and accepted
            else "REJECT_ALL_AT_SCREEN"
            if stage == "screen"
            else "ACCEPT_FULL140_INTEGRATION"
            if accepted
            else "REJECT_KEEP_FULL_SYSTEM_140"
        ),
    }
    result["report_sha256"] = canonical_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen experiment-635 full-system-140 variants."
    )
    parser.add_argument("--stage", choices=("screen", "full"), required=True)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--replay-contract", required=True, type=Path)
    parser.add_argument("--accepted-632-report", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--screen-report", type=Path)
    parser.add_argument("--runtime-report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        stage=args.stage,
        bundle_path=args.bundle,
        replay_contract_path=args.replay_contract,
        accepted_632_report_path=args.accepted_632_report,
        registry_path=args.registry,
        output_path=args.output,
        screen_report_path=args.screen_report,
        runtime_report_path=args.runtime_report,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP600 = HERE.parent / "600_semantic_v3_qwen35_baselines"

EXPERIMENT_ID = "602"
PROTOCOL_VERSION = "semantic_family_v3_seed_variance_v1"
DEPENDENCY_EXPERIMENT_ID = "600"
SEEDS = (42, 31415, 271828, 161803)
REFERENCE_SEED = 42
NEW_SEEDS = (31415, 271828, 161803)
FOLDS = (0, 1, 2, 3, 4)
DEVELOPMENT_ROWS = 11_118
SEALED_ROWS = 1_853
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 602042
ACCEPTANCE = {
    "macro_delta": 0.003,
    "fold_wins": 4,
    "maximum_category_drop": 0.005,
    "minimum_corrected_to_regressed": 1.5,
    "bootstrap_probability_positive": 0.90,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load required dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_exp600_dependencies():
    """Load the shared strict split wrapper and reject an incompatible interface."""
    protocol_path = EXP600 / "protocol.py"
    trainer_path = EXP600 / "train_component.py"
    if not protocol_path.is_file() or not trainer_path.is_file():
        raise FileNotFoundError(
            "experiment 602 requires the prepared experiment-600 wrapper; it is absent"
        )
    shared_protocol = _load_module("_exp602_shared_protocol", protocol_path)
    # The dependency uses a legacy absolute ``from protocol`` import. Inject its
    # sibling module only while importing, otherwise a caller's local protocol
    # module could silently be captured instead.
    previous_protocol = sys.modules.get("protocol")
    sys.modules["protocol"] = shared_protocol
    try:
        shared_trainer = _load_module("_exp602_shared_trainer", trainer_path)
    finally:
        if previous_protocol is None:
            del sys.modules["protocol"]
        else:
            sys.modules["protocol"] = previous_protocol
    required_protocol = {
        "EXPERIMENT_ID": DEPENDENCY_EXPERIMENT_ID,
        "PROTOCOL_VERSION": "semantic_family_v3",
        "DEVELOPMENT_ROWS": DEVELOPMENT_ROWS,
        "SEALED_ROWS": SEALED_ROWS,
        "DEVELOPMENT_FOLDS": FOLDS,
    }
    required_functions = (
        "materialize_runtime_fold",
        "read_folds",
        "sha256_file",
        "canonical_sha256",
        "expected_step_policy",
    )
    mismatches = {
        key: {"expected": expected, "actual": getattr(shared_protocol, key, None)}
        for key, expected in required_protocol.items()
        if getattr(shared_protocol, key, None) != expected
    }
    missing = [name for name in required_functions if not callable(getattr(shared_protocol, name, None))]
    if mismatches or missing:
        raise RuntimeError(
            "experiment-600 strict-wrapper interface changed; do not launch 602 until "
            f"the dependency is reviewed (mismatches={mismatches}, missing={missing})"
        )
    if not isinstance(getattr(shared_trainer, "PARENT_SHA256", None), dict):
        raise TypeError("experiment-600 parent checksum contract is unavailable")
    return shared_protocol, shared_trainer


def runtime_input_paths(runtime_input_dir: Path) -> dict[str, Path]:
    """Resolve only the local, shared selector inputs prepared for experiment 600.

    Raw competition data, images, and model files intentionally are not copied into
    this experiment. The caller supplies their existing paths; this directory must
    contain the two frozen development-only selector artifacts shared with 600.
    """
    root = runtime_input_dir.resolve()
    paths = {
        "data": root / "development_data.csv",
        "folds": root / "development_folds.csv",
        "selector": root / "development_selector_oof.npz",
        "selector_provenance": root / "development_selector_provenance.json",
        "manifest": root / "development_first_image_manifest.tsv.gz",
        "mapping": root / "development_id_map.csv",
        "audit": root / "zero_sealed_runtime_audit.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "shared experiment-600 runtime inputs are incomplete: " + ", ".join(missing)
        )
    return paths


def validate_seed(seed: int) -> int:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    return seed


def seed_job_manifest() -> list[dict[str, int]]:
    return [{"seed": seed, "fold": fold, "sealed_rows": 0} for seed in NEW_SEEDS for fold in FOLDS]


def expected_step_policy(training_records: int) -> dict[str, int]:
    batches = math.ceil(training_records / 4)
    return {"batches": batches, "optimizer_steps": math.ceil(batches / 4)}


def _f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def strictly_nested_predictions(
    *, probabilities: np.ndarray, labels: np.ndarray, categories: np.ndarray, folds: np.ndarray
) -> tuple[np.ndarray, dict[str, list[dict[str, float | int]]]]:
    """Fit one predetermined threshold per category/outer fold using the other four folds.

    There is no threshold search across experiment variants. Every supplied score
    vector receives the same deterministic calibration procedure; the only ensemble
    score accepted by this protocol is the arithmetic mean of all four probabilities.
    """
    output = np.zeros(len(labels), dtype=np.int8)
    detail: dict[str, list[dict[str, float | int]]] = {}
    for category in sorted(np.unique(categories)):
        category_mask = categories == category
        rows: list[dict[str, float | int]] = []
        for fold in FOLDS:
            train = category_mask & (folds != fold)
            valid = category_mask & (folds == fold)
            values = probabilities[train]
            # Fixed quantile grid inherited from the exact parent recipe.
            candidates = np.unique(np.quantile(values, np.linspace(0.002, 0.998, 700)))
            candidate_f1 = np.asarray(
                [_f1(labels[train], values >= threshold) for threshold in candidates],
                dtype=np.float64,
            )
            best_index = int(np.argmax(candidate_f1))
            threshold = float(candidates[best_index])
            output[valid] = (probabilities[valid] >= threshold).astype(np.int8)
            rows.append(
                {
                    "fold": fold,
                    "threshold": threshold,
                    "calibration_rows": int(train.sum()),
                    "validation_rows": int(valid.sum()),
                    "calibration_f1": float(candidate_f1[best_index]),
                }
            )
        detail[str(category)] = rows
    return output, detail


def paired_component_bootstrap(
    *,
    labels: np.ndarray,
    categories: np.ndarray,
    components: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    by_category: dict[str, list[np.ndarray]] = {}
    for category in sorted(np.unique(categories)):
        positions = np.flatnonzero(categories == category)
        grouped: dict[str, list[int]] = {}
        for position in positions:
            grouped.setdefault(str(components[position]), []).append(int(position))
        by_category[category] = [np.asarray(rows, dtype=np.int64) for rows in grouped.values()]
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        baseline_values, candidate_values = [], []
        for category in sorted(by_category):
            groups = by_category[category]
            sampled = rng.integers(0, len(groups), size=len(groups))
            positions = np.concatenate([groups[index] for index in sampled])
            baseline_values.append(_f1(labels[positions], baseline[positions]))
            candidate_values.append(_f1(labels[positions], candidate[positions]))
        deltas[iteration] = float(np.mean(candidate_values) - np.mean(baseline_values))
    return {
        "unit": "semantic_component_within_category",
        "iterations": iterations,
        "seed": seed,
        "delta_mean": float(deltas.mean()),
        "delta_ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "probability_delta_positive": float((deltas > 0).mean()),
    }


def audit_candidate(
    *,
    name: str,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    components: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    calibration: dict[str, list[dict[str, float | int]]],
) -> dict[str, Any]:
    baseline_category = {
        category: _f1(labels[categories == category], baseline[categories == category])
        for category in sorted(np.unique(categories))
    }
    candidate_category = {
        category: _f1(labels[categories == category], candidate[categories == category])
        for category in sorted(np.unique(categories))
    }
    fold_rows = []
    for fold in FOLDS:
        mask = folds == fold
        old = [_f1(labels[mask & (categories == category)], baseline[mask & (categories == category)]) for category in sorted(baseline_category)]
        new = [_f1(labels[mask & (categories == category)], candidate[mask & (categories == category)]) for category in sorted(candidate_category)]
        fold_rows.append({"fold": fold, "baseline_macro_f1": float(np.mean(old)), "candidate_macro_f1": float(np.mean(new)), "delta": float(np.mean(new) - np.mean(old))})
    corrected = int(((baseline != labels) & (candidate == labels)).sum())
    regressed = int(((baseline == labels) & (candidate != labels)).sum())
    category_delta = {key: candidate_category[key] - baseline_category[key] for key in baseline_category}
    bootstrap = paired_component_bootstrap(
        labels=labels,
        categories=categories,
        components=components,
        baseline=baseline,
        candidate=candidate,
    )
    delta = float(np.mean(list(candidate_category.values())) - np.mean(list(baseline_category.values())))
    ratio = None if regressed == 0 else corrected / regressed
    acceptance = {
        "macro_delta_at_least_0_003": delta >= ACCEPTANCE["macro_delta"],
        "wins_at_least_4_of_5": sum(row["delta"] > 0 for row in fold_rows) >= ACCEPTANCE["fold_wins"],
        "no_category_drop_over_0_005": min(category_delta.values()) >= -ACCEPTANCE["maximum_category_drop"],
        "bootstrap_positive": bootstrap["probability_delta_positive"] >= ACCEPTANCE["bootstrap_probability_positive"],
        "corrected_to_regressed_at_least_1_5": regressed == 0 or ratio >= ACCEPTANCE["minimum_corrected_to_regressed"],
    }
    return {
        "candidate": name,
        "baseline_macro_f1": float(np.mean(list(baseline_category.values()))),
        "candidate_macro_f1": float(np.mean(list(candidate_category.values()))),
        "delta_macro_f1": delta,
        "baseline_category_f1": baseline_category,
        "candidate_category_f1": candidate_category,
        "category_delta": category_delta,
        "folds": fold_rows,
        "folds_won": sum(row["delta"] > 0 for row in fold_rows),
        "changed_predictions": int((baseline != candidate).sum()),
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": ratio,
        "calibration": calibration,
        "component_bootstrap": bootstrap,
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
    }

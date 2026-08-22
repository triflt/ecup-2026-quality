from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from shingle_neighbor_prior_cv import canonical


ROOT = Path("research")
DATA = Path(os.environ.get("ECUP_DATA", ROOT / "data.csv"))
QWEN35 = ROOT / "qwen35-hard-5fold-robust-fusion-report.npz"
QWEN3VL = ROOT / "lora-hard-5fold-robust-fusion-report.npz"
REPORT = ROOT / "qwen3vl-qwen35-soft-annotator-prior-report.json"

MODEL_CONFIG = {
    "БАД": ((0.50, 0.25, 0.25), 0.27193570137023926),
    "Легковоспламеняющиеся": ((0.15, 0.10, 0.75), 0.953912615776062),
}


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def compose_text(name: object, description: object) -> str:
    name = normalize(name)
    return f"{name}\n{name}\n{normalize(description)}"


def fingerprint(value: object) -> str:
    return hashlib.sha1(normalize(value).encode("utf-8")).hexdigest()


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def calibrate_at_threshold(score: np.ndarray, threshold: float) -> np.ndarray:
    """Monotonic piecewise-linear calibration that maps the model boundary to 0.5."""
    score = np.asarray(score, dtype=np.float32)
    result = np.empty_like(score)
    below = score < threshold
    result[below] = 0.5 * score[below] / max(threshold, 1e-8)
    result[~below] = 0.5 + 0.5 * (score[~below] - threshold) / max(1 - threshold, 1e-8)
    return np.clip(result, 0, 1)


def posterior_for_targets(
    frame: pd.DataFrame,
    donors: np.ndarray,
    targets: np.ndarray,
    key: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    donor = frame.iloc[donors]
    global_rate = float(donor.label.mean())
    stats = donor.groupby(key, sort=False).label.agg(["count", "sum"])
    count_map = stats["count"].to_dict()
    sum_map = stats["sum"].to_dict()
    values = frame.iloc[targets][key].astype(str).to_numpy()
    counts = np.fromiter((count_map.get(value, 0) for value in values), dtype=np.int32)
    sums = np.fromiter((sum_map.get(value, 0) for value in values), dtype=np.float32)
    posterior = np.divide(
        sums + alpha * global_rate,
        counts + alpha,
        out=np.full(len(targets), global_rate, dtype=np.float32),
        where=(counts + alpha) > 0,
    )
    confidence = np.maximum(posterior, 1 - posterior)
    return counts, posterior.astype(np.float32), confidence.astype(np.float32)


def family_posterior(
    family_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    policy: tuple[int, float, int, float, int, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    exact_min, exact_conf, name_min, name_conf, canonical_min, canonical_conf = policy
    size = len(family_arrays["text_hash"][0])
    posterior = np.full(size, np.nan, dtype=np.float32)
    unresolved = np.ones(size, dtype=bool)
    hits: dict[str, int] = {}
    for label, key, minimum, confidence_minimum in [
        ("exact", "text_hash", exact_min, exact_conf),
        ("name", "normalized_name", name_min, name_conf),
        ("canonical", "canonical_text_mask_digits", canonical_min, canonical_conf),
    ]:
        counts, values, confidence = family_arrays[key]
        selected = (
            unresolved
            & (counts >= minimum)
            & (confidence >= confidence_minimum)
            & (minimum < 999)
        )
        posterior[selected] = values[selected]
        unresolved[selected] = False
        hits[label] = int(selected.sum())
    return posterior, ~unresolved, hits


def policies(category: str) -> list[tuple[int, float, int, float, int, float]]:
    exact = list(product([1, 2, 4], [0.55, 2 / 3, 0.75, 0.90, 0.999]))
    if category == "БАД":
        name = [(2, 0.999), (3, 0.999), (8, 0.999), (2, 0.90), (3, 0.90), (999, 0.999)]
        canonical_options = [(1, 0.999), (2, 0.999), (1, 0.90), (2, 0.90), (999, 0.999)]
    else:
        name = [(999, 0.999)]
        canonical_options = [(999, 0.999)]
    return [(*a, *n, *c) for a, n, c in product(exact, name, canonical_options)]


def apply_soft(
    model_probability: np.ndarray,
    family_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: tuple[float, float, float, tuple[int, float, int, float, int, float]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    weight, _alpha, threshold, policy = config
    posterior, covered, hits = family_posterior(family_arrays, policy)
    score = model_probability.copy()
    score[covered] = (1 - weight) * score[covered] + weight * posterior[covered]
    return (score >= threshold).astype(np.int8), covered, hits


def build_split_cache(
    frame: pd.DataFrame,
    donors: np.ndarray,
    targets: np.ndarray,
    alphas: list[float],
) -> dict[float, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    return {
        alpha: {
            key: posterior_for_targets(frame, donors, targets, key, alpha)
            for key in ["text_hash", "normalized_name", "canonical_text_mask_digits"]
        }
        for alpha in alphas
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()

    frame = pd.read_csv(DATA)
    frame["name"] = frame.name.fillna("").astype(str)
    frame["description"] = frame.description.fillna("").astype(str)
    frame["normalized_name"] = frame.name.map(normalize)
    texts = [compose_text(a, b) for a, b in zip(frame.name, frame.description)]
    frame["text_hash"] = [fingerprint(text) for text in texts]
    frame["canonical_text_mask_digits"] = [canonical(text, mask_digits=True) for text in texts]

    qwen35 = np.load(QWEN35, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    ids = frame.id.astype(str).to_numpy()
    if not np.array_equal(ids, qwen35["ids"].astype(str)) or not np.array_equal(ids, qwen3vl["ids"].astype(str)):
        raise ValueError("id mismatch")
    frame["fold"] = qwen35["folds"]
    labels = frame.label.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy(dtype=np.int8)
    categories = frame.category.astype(str).to_numpy()

    alphas = [0.0, 1.0, 3.0, 10.0]
    weights = [0.25, 0.50, 0.75, 1.0]
    thresholds = [0.45, 0.475, 0.50, 0.525, 0.55]
    report: dict[str, object] = {
        "protocol": "donor-only nested grouped CV; beta-smoothed exact/name/canonical family posterior blended with calibrated continuous three-model rank score",
        "grid": {"alphas": alphas, "weights": weights, "thresholds": thresholds},
        "categories": {},
    }
    macro = []

    for category in sorted(frame.category.unique()):
        category_mask = categories == category
        category_indices = np.flatnonzero(category_mask)
        (base_weight, vl_weight, q35_weight), model_threshold = MODEL_CONFIG[category]
        model_score = (
            base_weight * qwen35["base_rank"].astype(np.float32)
            + vl_weight * qwen3vl["lora_rank"].astype(np.float32)
            + q35_weight * qwen35["lora_rank"].astype(np.float32)
        )
        model_probability = calibrate_at_threshold(model_score, model_threshold)
        base_predictions = (model_score >= model_threshold).astype(np.int8)
        category_policies = policies(category)
        nested = base_predictions.copy()
        chosen_configs = []
        fold_rows = []

        for outer_fold in sorted(frame.fold.unique()):
            outer_train_mask = category_mask & (folds != outer_fold)
            inner_splits = []
            for inner_fold in sorted(frame.fold.unique()):
                if inner_fold == outer_fold:
                    continue
                donors = np.flatnonzero(outer_train_mask & (folds != inner_fold))
                targets = np.flatnonzero(outer_train_mask & (folds == inner_fold))
                cache = build_split_cache(frame, donors, targets, alphas)
                inner_splits.append((targets, cache))

            best = None
            for alpha, weight, threshold, policy in product(alphas, weights, thresholds, category_policies):
                split_labels, split_predictions = [], []
                total_covered = 0
                for targets, cache in inner_splits:
                    config = (weight, alpha, threshold, policy)
                    predictions, covered, _ = apply_soft(model_probability[targets], cache[alpha], config)
                    split_labels.append(labels[targets])
                    split_predictions.append(predictions)
                    total_covered += int(covered.sum())
                value = f1(np.concatenate(split_labels), np.concatenate(split_predictions))
                # Prefer smaller family influence and fewer covered rows when F1 ties.
                preference = (value, -weight, -total_covered, -abs(threshold - 0.5), -alpha)
                if best is None or preference > best[0]:
                    best = (preference, (weight, alpha, threshold, policy), value)
            assert best is not None
            config = best[1]
            chosen_configs.append(config)
            donors = np.flatnonzero(outer_train_mask)
            targets = np.flatnonzero(category_mask & (folds == outer_fold))
            cache = build_split_cache(frame, donors, targets, [config[1]])
            predictions, covered, hits = apply_soft(model_probability[targets], cache[config[1]], config)
            nested[targets] = predictions
            fold_rows.append({
                "fold": int(outer_fold),
                "config": [config[0], config[1], config[2], list(config[3])],
                "inner_f1": best[2],
                "validation_f1": f1(labels[targets], predictions),
                "covered": int(covered.sum()),
                "hits": hits,
            })
            print(f"category={category} outer_fold={outer_fold} config={config} validation_f1={fold_rows[-1]['validation_f1']:.8f}", flush=True)

        nested_f1 = f1(labels[category_indices], nested[category_indices])
        baseline_f1 = f1(labels[category_indices], base_predictions[category_indices])
        encoded = [json.dumps([c[0], c[1], c[2], list(c[3])]) for c in chosen_configs]
        winner_encoded = Counter(encoded).most_common(1)[0][0]
        winner_raw = json.loads(winner_encoded)
        production_config = (
            float(winner_raw[0]), float(winner_raw[1]), float(winner_raw[2]), tuple(winner_raw[3])
        )
        full_cache = build_split_cache(frame, category_indices, category_indices, [production_config[1]])
        production_predictions, production_covered, production_hits = apply_soft(
            model_probability[category_indices], full_cache[production_config[1]], production_config
        )
        report["categories"][category] = {
            "rows": int(category_mask.sum()),
            "baseline_f1": baseline_f1,
            "nested_soft_f1": nested_f1,
            "nested_delta": nested_f1 - baseline_f1,
            "production_config": [production_config[0], production_config[1], production_config[2], list(production_config[3])],
            "production_full_fit_apparent_f1_not_cv": f1(labels[category_indices], production_predictions),
            "production_covered": int(production_covered.sum()),
            "production_hits": production_hits,
            "choice_counts": dict(Counter(encoded)),
            "folds": fold_rows,
        }
        macro.append(nested_f1)

    report["nested_macro_f1"] = float(np.mean(macro))
    report["baseline_macro_f1"] = float(np.mean([
        value["baseline_f1"] for value in report["categories"].values()
    ]))
    report["nested_delta"] = report["nested_macro_f1"] - report["baseline_macro_f1"]
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

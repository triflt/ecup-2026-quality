from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ("БАД", "Легковоспламеняющиеся")
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 620_042


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def macro_f1(labels: np.ndarray, predictions: np.ndarray, categories: np.ndarray) -> float:
    return float(
        np.mean(
            [
                f1_score(
                    labels[categories == category],
                    predictions[categories == category],
                    zero_division=0,
                )
                for category in CATEGORIES
            ]
        )
    )


def metric_summary(
    labels: np.ndarray, predictions: np.ndarray, categories: np.ndarray
) -> dict[str, Any]:
    category_f1 = {
        category: float(
            f1_score(
                labels[categories == category],
                predictions[categories == category],
                zero_division=0,
            )
        )
        for category in CATEGORIES
    }
    return {"category_f1": category_f1, "macro_f1": float(np.mean(list(category_f1.values())))}


def correction_counts(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> tuple[int, int]:
    corrected = int(((baseline != labels) & (candidate == labels)).sum())
    regressed = int(((baseline == labels) & (candidate != labels)).sum())
    return corrected, regressed


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
    tp = counts[..., 0]
    fp = counts[..., 1]
    fn = counts[..., 2]
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
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
    batch_size: int = 128,
) -> dict[str, Any]:
    if repeats < 1 or batch_size < 1:
        raise ValueError("bootstrap repeats and batch size must be positive")
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
    probability = np.full(component_count, 1.0 / component_count, dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, batch_size):
        stop = min(start + batch_size, repeats)
        weights = rng.multinomial(component_count, probability, size=stop - start)
        baseline_sample = np.einsum("bg,gcs->bcs", weights, baseline_counts, optimize=True)
        candidate_sample = np.einsum("bg,gcs->bcs", weights, candidate_counts, optimize=True)
        deltas[start:stop] = _macro_from_confusion(candidate_sample) - _macro_from_confusion(
            baseline_sample
        )
    return {
        "unit": "semantic_component",
        "components": component_count,
        "repeats": repeats,
        "seed": seed,
        "probability_delta_positive": float(np.mean(deltas > 0)),
        "delta_quantiles": {
            "0.025": float(np.quantile(deltas, 0.025)),
            "0.5": float(np.quantile(deltas, 0.5)),
            "0.975": float(np.quantile(deltas, 0.975)),
        },
    }


def _slice_delta(
    *,
    mask: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    category_rows = {category: int((mask & (categories == category)).sum()) for category in CATEGORIES}
    if any(rows == 0 for rows in category_rows.values()):
        raise ValueError("slice lacks one frozen category")
    old = macro_f1(labels[mask], baseline[mask], categories[mask])
    new = macro_f1(labels[mask], candidate[mask], categories[mask])
    corrected, regressed = correction_counts(labels[mask], baseline[mask], candidate[mask])
    return {
        "rows": int(mask.sum()),
        "category_rows": category_rows,
        "baseline_macro_f1": old,
        "candidate_macro_f1": new,
        "delta_macro_f1": new - old,
        "corrected": corrected,
        "regressed": regressed,
    }


def _load_prior_module() -> Any:
    path = ROOT / "research" / "audit_component_decision_survival.py"
    research = str(ROOT / "research")
    if research not in sys.path:
        sys.path.insert(0, research)
    spec = importlib.util.spec_from_file_location("exp620_prior_replay", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load prior replay module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def post_prior_replay(
    *,
    data_path: Path,
    ids: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    prior = _load_prior_module()
    frame = pd.read_csv(data_path, dtype={"id": str})
    required = {"id", "category", "name", "description"}
    if set(frame.columns) != required:
        raise ValueError("label-free development data schema mismatch for prior replay")
    if not np.array_equal(frame["id"].astype(str).to_numpy(), ids):
        raise ValueError("prior replay data IDs/order mismatch")
    if not np.array_equal(frame["category"].astype(str).to_numpy(), categories):
        raise ValueError("prior replay category mismatch")
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["label"] = labels
    frame["fold"] = folds
    frame["normalized_name"] = frame["name"].map(prior.normalize)
    texts = [
        prior.compose_text(name, description)
        for name, description in zip(frame["name"], frame["description"], strict=True)
    ]
    frame["text_hash"] = [prior.fingerprint(text) for text in texts]
    frame["canonical_text_mask_digits"] = [
        prior.canonical(text, mask_digits=True) for text in texts
    ]
    neighbor_indices, neighbor_scores = prior.build_neighbor_graph(frame)
    baseline_after, baseline_audit = prior.apply_downstream_priors(
        frame, baseline, neighbor_indices, neighbor_scores
    )
    candidate_after, candidate_audit = prior.apply_downstream_priors(
        frame, candidate, neighbor_indices, neighbor_scores
    )
    baseline_metrics = metric_summary(labels, baseline_after, categories)
    candidate_metrics = metric_summary(labels, candidate_after, categories)
    return {
        "protocol": "donor_only_outer_fold_replay",
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta_macro_f1": candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "category_delta": {
            category: candidate_metrics["category_f1"][category]
            - baseline_metrics["category_f1"][category]
            for category in CATEGORIES
        },
        "change_survival": prior.changed_audit(
            labels, categories, baseline, candidate, baseline_after, candidate_after
        ),
        "baseline_prior_hits": baseline_audit,
        "candidate_prior_hits": candidate_audit,
    }


def evaluate(
    *,
    route_predictions_path: Path,
    screen_dir: Path,
    development_data_path: Path,
    output_dir: Path,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    screen_report = json.loads((screen_dir / "screen_metrics.json").read_text(encoding="utf-8"))
    if screen_report.get("status") != "screen_passed_full_audit_pending":
        raise RuntimeError("full audit is forbidden unless the frozen screen passed")
    with np.load(route_predictions_path, allow_pickle=False) as route:
        ids = route["ids"].astype(str)
        labels = route["labels"].astype(np.int8)
        categories = route["categories"].astype(str)
        folds = route["folds"].astype(np.int8)
        components = route["semantic_components"].astype(str)
        baseline = route["original_route_predictions"].astype(np.int8)
    with np.load(screen_dir / "screen_predictions.npz", allow_pickle=False) as screen:
        for key, expected in (("ids", ids), ("folds", folds), ("baseline_predictions", baseline)):
            if not np.array_equal(screen[key].astype(expected.dtype), expected):
                raise ValueError(f"screen prediction mismatch for {key}")
        candidate = screen["candidate_predictions"].astype(np.int8)
    baseline_metrics = metric_summary(labels, baseline, categories)
    candidate_metrics = metric_summary(labels, candidate, categories)
    category_delta = {
        category: candidate_metrics["category_f1"][category]
        - baseline_metrics["category_f1"][category]
        for category in CATEGORIES
    }
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        old = macro_f1(labels[mask], baseline[mask], categories[mask])
        new = macro_f1(labels[mask], candidate[mask], categories[mask])
        fold_rows.append(
            {
                "fold": int(fold),
                "baseline_macro_f1": old,
                "candidate_macro_f1": new,
                "delta_macro_f1": new - old,
            }
        )
    corrected, regressed = correction_counts(labels, baseline, candidate)
    ratio = None if regressed == 0 else corrected / regressed
    _, inverse, component_sizes = np.unique(components, return_inverse=True, return_counts=True)
    singleton = component_sizes[inverse] == 1
    slices = {
        "singleton": _slice_delta(
            mask=singleton,
            labels=labels,
            categories=categories,
            baseline=baseline,
            candidate=candidate,
        ),
        "repeated": _slice_delta(
            mask=~singleton,
            labels=labels,
            categories=categories,
            baseline=baseline,
            candidate=candidate,
        ),
    }
    bootstrap = grouped_component_bootstrap(
        labels=labels,
        categories=categories,
        components=components,
        baseline=baseline,
        candidate=candidate,
        repeats=bootstrap_repeats,
    )
    prior_replay = post_prior_replay(
        data_path=development_data_path,
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        baseline=baseline,
        candidate=candidate,
    )
    macro_delta = candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"]
    gates = {
        "macro_delta_at_least_0_003": macro_delta >= 0.003,
        "folds_won_at_least_4": sum(row["delta_macro_f1"] > 0 for row in fold_rows) >= 4,
        "no_category_drop": min(category_delta.values()) >= 0,
        "corrected_to_regressed_at_least_1_5": (
            corrected > 0 if ratio is None else ratio >= 1.5
        ),
        "component_bootstrap_probability_at_least_0_90": bootstrap[
            "probability_delta_positive"
        ]
        >= 0.90,
        "singleton_delta_positive": slices["singleton"]["delta_macro_f1"] > 0,
        "repeated_delta_nonnegative": slices["repeated"]["delta_macro_f1"] >= 0,
        "post_prior_delta_positive": prior_replay["delta_macro_f1"] > 0,
    }
    result = {
        "experiment_id": "620",
        "status": "complete",
        "validation_version": "semantic_family_v3",
        "fully_nested_meta_validation": False,
        "validation_limitation": (
            "The fixed route baseline still uses donor OOF component scores whose generating model "
            "may have trained on labels from the nominal evaluated outer fold."
        ),
        "sealed_holdout_used": False,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta_macro_f1": macro_delta,
        "category_delta": category_delta,
        "folds": fold_rows,
        "folds_won": int(sum(row["delta_macro_f1"] > 0 for row in fold_rows)),
        "changed": int((candidate != baseline).sum()),
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": ratio,
        "component_bootstrap": bootstrap,
        "slices": slices,
        "post_prior_replay": prior_replay,
        "gates": gates,
        "decision": "GO" if all(gates.values()) else "NO_GO",
        "input_sha256": {
            "route_predictions": sha256_file(route_predictions_path),
            "screen_metrics": sha256_file(screen_dir / "screen_metrics.json"),
            "screen_predictions": sha256_file(screen_dir / "screen_predictions.npz"),
            "development_data": sha256_file(development_data_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "full_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-predictions", type=Path, required=True)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    args = parser.parse_args()
    result = evaluate(
        route_predictions_path=args.route_predictions,
        screen_dir=args.screen_dir,
        development_data_path=args.development_data,
        output_dir=args.output_dir,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

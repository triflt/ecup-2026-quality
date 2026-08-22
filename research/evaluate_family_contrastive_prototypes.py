from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT / "research/first-image-artifacts/extracted/train_embeddings_fp16.npz"
QWEN3VL = ROOT / "research/lora-hard-5fold-robust-fusion-report.npz"
QWEN35 = ROOT / "research/qwen35-hard-5fold-robust-fusion-report.npz"
LOCKED = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
FOLDS = ROOT / "validation/grouped_text_v1/folds.csv"
CONFIG = {
    "БАД": {
        "weights": np.asarray([0.50, 0.25, 0.25], dtype=np.float32),
        "production_threshold": 0.27193570137023926,
    },
    "Легковоспламеняющиеся": {
        "weights": np.asarray([0.15, 0.10, 0.75], dtype=np.float32),
        "production_threshold": 0.953912615776062,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float32)
    result[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return result


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered = labels[order]
    tp = np.cumsum(ordered == 1)
    fp = np.cumsum(ordered == 0)
    fn = int((labels == 1).sum()) - tp
    values = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    best = int(np.argmax(values))
    if best + 1 == len(scores):
        threshold = float(scores[order[best]] - 1e-7)
    else:
        threshold = float((scores[order[best]] + scores[order[best + 1]]) / 2)
    return float(values[best]), threshold


def family_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    group_hashes: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    groups, inverse = np.unique(group_hashes[positions], return_inverse=True)
    sums = np.zeros((len(groups), features.shape[1]), dtype=np.float32)
    counts = np.zeros(len(groups), dtype=np.int32)
    minimum = np.ones(len(groups), dtype=np.int8)
    maximum = np.zeros(len(groups), dtype=np.int8)
    np.add.at(sums, inverse, features[positions])
    np.add.at(counts, inverse, 1)
    np.minimum.at(minimum, inverse, labels[positions])
    np.maximum.at(maximum, inverse, labels[positions])
    consistent = minimum == maximum
    prototypes = normalize_rows(sums[consistent] / counts[consistent, None])
    prototype_labels = minimum[consistent]
    return (
        prototypes[prototype_labels == 1],
        prototypes[prototype_labels == 0],
        {
            "families_total": int(len(groups)),
            "families_consistent": int(consistent.sum()),
            "families_conflicting_excluded": int((~consistent).sum()),
            "positive_prototypes": int((prototype_labels == 1).sum()),
            "negative_prototypes": int((prototype_labels == 0).sum()),
        },
    )


def neighbor_difference(
    query: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    neighbors: int,
    batch_size: int,
) -> np.ndarray:
    if len(positive) < neighbors or len(negative) < neighbors:
        raise ValueError("not enough class prototypes")
    result = np.empty(len(query), dtype=np.float32)
    for start in range(0, len(query), batch_size):
        block = query[start : start + batch_size]
        # Apple Accelerate may emit spurious floating-point warnings for finite
        # float32 GEMM outputs, so validate the result explicitly after matmul.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            positive_similarity = block @ positive.T
            negative_similarity = block @ negative.T
        if not np.isfinite(positive_similarity).all() or not np.isfinite(
            negative_similarity
        ).all():
            raise ValueError("non-finite prototype similarity")
        positive_top = np.partition(
            positive_similarity, -neighbors, axis=1
        )[:, -neighbors:]
        negative_top = np.partition(
            negative_similarity, -neighbors, axis=1
        )[:, -neighbors:]
        result[start : start + len(block)] = (
            positive_top.mean(axis=1) - negative_top.mean(axis=1)
        )
    return result


def category_scores(
    labels: np.ndarray,
    predictions: np.ndarray,
    categories: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    return {
        category: f1(
            labels[mask & (categories == category)],
            predictions[mask & (categories == category)],
        )
        for category in sorted(np.unique(categories))
    }


def paired_audit(
    *,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    group_hashes: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    baseline_category = category_scores(labels, baseline, categories)
    candidate_category = category_scores(labels, candidate, categories)
    baseline_macro = float(np.mean(list(baseline_category.values())))
    candidate_macro = float(np.mean(list(candidate_category.values())))
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        old = category_scores(labels, baseline, categories, mask)
        new = category_scores(labels, candidate, categories, mask)
        old_macro = float(np.mean(list(old.values())))
        new_macro = float(np.mean(list(new.values())))
        fold_rows.append(
            {
                "fold": int(fold),
                "baseline_macro_f1": old_macro,
                "candidate_macro_f1": new_macro,
                "delta": new_macro - old_macro,
                "baseline_category_f1": old,
                "candidate_category_f1": new,
            }
        )
    group_rows: dict[str, list[np.ndarray]] = {}
    for category in sorted(np.unique(categories)):
        grouped: dict[str, list[int]] = {}
        for position in np.flatnonzero(categories == category):
            grouped.setdefault(group_hashes[position], []).append(position)
        group_rows[category] = [np.asarray(rows) for rows in grouped.values()]
    rng = np.random.default_rng(seed)
    deltas = np.empty(bootstrap, dtype=np.float64)
    for iteration in range(bootstrap):
        old_values, new_values = [], []
        for category in sorted(group_rows):
            groups = group_rows[category]
            chosen = rng.integers(0, len(groups), size=len(groups))
            sample = np.concatenate([groups[index] for index in chosen])
            old_values.append(f1(labels[sample], baseline[sample]))
            new_values.append(f1(labels[sample], candidate[sample]))
        deltas[iteration] = np.mean(new_values) - np.mean(old_values)
    category_delta = {
        category: candidate_category[category] - baseline_category[category]
        for category in baseline_category
    }
    group_sizes = pd.Series(group_hashes).map(pd.Series(group_hashes).value_counts()).to_numpy()
    singleton = group_sizes == 1
    return {
        "baseline_macro_f1": baseline_macro,
        "candidate_macro_f1": candidate_macro,
        "delta_macro_f1": candidate_macro - baseline_macro,
        "baseline_category_f1": baseline_category,
        "candidate_category_f1": candidate_category,
        "category_delta": category_delta,
        "folds_won": int(sum(row["delta"] > 0 for row in fold_rows)),
        "folds": fold_rows,
        "changed_predictions": int((baseline != candidate).sum()),
        "corrected": int(((baseline != labels) & (candidate == labels)).sum()),
        "regressed": int(((baseline == labels) & (candidate != labels)).sum()),
        "singleton_changed": int((singleton & (baseline != candidate)).sum()),
        "singleton_corrected": int((singleton & (baseline != labels) & (candidate == labels)).sum()),
        "singleton_regressed": int((singleton & (baseline == labels) & (candidate != labels)).sum()),
        "group_bootstrap": {
            "iterations": bootstrap,
            "seed": seed,
            "probability_delta_positive": float((deltas > 0).mean()),
            "delta_ci95": [
                float(np.quantile(deltas, 0.025)),
                float(np.quantile(deltas, 0.975)),
            ],
        },
        "acceptance": {
            "delta_at_least_0_003": candidate_macro - baseline_macro >= 0.003,
            "wins_at_least_4_of_5_folds": sum(row["delta"] > 0 for row in fold_rows) >= 4,
            "no_category_drop_over_0_005": min(category_delta.values()) >= -0.005,
            "bootstrap_probability_at_least_0_90": float((deltas > 0).mean()) >= 0.90,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--weight", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=34042)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.weight < 0.5:
        raise ValueError("weight must be in (0, 0.5)")
    started = time.monotonic()
    embedding_archive = np.load(EMBEDDINGS, allow_pickle=False)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    qwen35 = np.load(QWEN35, allow_pickle=True)
    locked = np.load(LOCKED, allow_pickle=False)
    ids = locked["ids"].astype(str)
    labels = locked["labels"].astype(np.int8)
    categories = locked["categories"].astype(str)
    folds = locked["folds"].astype(np.int8)
    for name, source, fold_key in (
        ("embeddings", embedding_archive, None),
        ("qwen3vl", qwen3vl, "folds"),
        ("qwen35", qwen35, "folds"),
    ):
        if not np.array_equal(source["ids"].astype(str), ids):
            raise ValueError(f"{name} id mismatch")
        if not np.array_equal(source["labels"].astype(np.int8), labels):
            raise ValueError(f"{name} label mismatch")
        if not np.array_equal(source["categories"].astype(str), categories):
            raise ValueError(f"{name} category mismatch")
        if fold_key and not np.array_equal(source[fold_key].astype(np.int8), folds):
            raise ValueError(f"{name} fold mismatch")
    fold_frame = pd.read_csv(FOLDS, dtype={"id": str}).set_index("id").loc[ids]
    if not np.array_equal(fold_frame.fold.to_numpy(np.int8), folds):
        raise ValueError("fold registry mismatch")
    group_hashes = fold_frame.group_hash.astype(str).to_numpy()
    embeddings = embedding_archive["embeddings"].astype(np.float32)
    rng = np.random.default_rng(args.seed)
    projection = rng.normal(
        0.0,
        1.0 / np.sqrt(args.projection_dim),
        size=(embeddings.shape[1], args.projection_dim),
    ).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected = normalize_rows(embeddings @ projection)
    if not np.isfinite(projected).all():
        raise ValueError("non-finite projected embedding")
    del embeddings
    print(
        json.dumps(
            {
                "event": "projection_complete",
                "shape": list(projected.shape),
                "elapsed_min": (time.monotonic() - started) / 60,
            }
        ),
        flush=True,
    )

    raw_prototype_score = np.full(len(ids), np.nan, dtype=np.float32)
    prototype_detail = []
    for category in sorted(np.unique(categories)):
        category_mask = categories == category
        for outer in sorted(np.unique(folds)):
            donor = np.flatnonzero(category_mask & (folds != outer))
            valid = np.flatnonzero(category_mask & (folds == outer))
            positive, negative, detail = family_prototypes(
                projected, labels, group_hashes, donor
            )
            raw_prototype_score[valid] = neighbor_difference(
                projected[valid],
                positive,
                negative,
                args.neighbors,
                args.batch_size,
            )
            detail.update(
                {
                    "category": category,
                    "outer_fold": int(outer),
                    "validation_rows": int(len(valid)),
                }
            )
            prototype_detail.append(detail)
            print(
                json.dumps(
                    {
                        "event": "fold_complete",
                        **detail,
                        "elapsed_min": (time.monotonic() - started) / 60,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if not np.isfinite(raw_prototype_score).all():
        raise ValueError("incomplete prototype scores")
    prototype_rank = np.empty(len(ids), dtype=np.float32)
    for category in sorted(np.unique(categories)):
        for fold in sorted(np.unique(folds)):
            positions = np.flatnonzero((categories == category) & (folds == fold))
            prototype_rank[positions] = rank01(raw_prototype_score[positions])

    base_rank = qwen35["base_rank"].astype(np.float32)
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)
    qwen35_rank = qwen35["lora_rank"].astype(np.float32)
    baseline_score = np.empty(len(ids), dtype=np.float32)
    candidate_score = np.empty(len(ids), dtype=np.float32)
    nested_candidate = np.zeros(len(ids), dtype=np.int8)
    production_candidate = np.zeros(len(ids), dtype=np.int8)
    threshold_detail = []
    for category in sorted(np.unique(categories)):
        mask = categories == category
        weights = CONFIG[category]["weights"]
        baseline_score[mask] = np.sum(
            np.column_stack(
                [base_rank[mask], qwen3vl_rank[mask], qwen35_rank[mask]]
            )
            * weights[None, :],
            axis=1,
            dtype=np.float32,
        )
        candidate_score[mask] = (
            (1.0 - args.weight) * baseline_score[mask]
            + args.weight * prototype_rank[mask]
        )
        for outer in sorted(np.unique(folds)):
            donor = mask & (folds != outer)
            valid = mask & (folds == outer)
            donor_f1, threshold = best_threshold(labels[donor], candidate_score[donor])
            nested_candidate[valid] = (candidate_score[valid] >= threshold).astype(np.int8)
            threshold_detail.append(
                {
                    "category": category,
                    "outer_fold": int(outer),
                    "threshold": threshold,
                    "donor_f1": donor_f1,
                }
            )
        production_candidate[mask] = (
            candidate_score[mask] >= CONFIG[category]["production_threshold"]
        ).astype(np.int8)

    nested_baseline = locked["baseline_nested_predictions"].astype(np.int8)
    production_baseline = locked["baseline_production_predictions"].astype(np.int8)
    nested_audit = paired_audit(
        labels=labels,
        categories=categories,
        folds=folds,
        group_hashes=group_hashes,
        baseline=nested_baseline,
        candidate=nested_candidate,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    production_audit = paired_audit(
        labels=labels,
        categories=categories,
        folds=folds,
        group_hashes=group_hashes,
        baseline=production_baseline,
        candidate=production_candidate,
        bootstrap=args.bootstrap,
        seed=args.seed + 1,
    )
    report = {
        "protocol": "family_contrastive_prototype_grouped_screen_v1",
        "data_version": "competition_train_v1",
        "evaluation_version": "component_transfer_gate_v1",
        "fixed_before_evaluation": {
            "projection": f"Gaussian 2048 to {args.projection_dim}",
            "projection_seed": args.seed,
            "neighbors_per_label": args.neighbors,
            "prototype_weight": args.weight,
            "conflicting_families": "excluded",
            "family_weighting": "one mean prototype per group_hash",
        },
        "prototype_detail": prototype_detail,
        "threshold_detail": threshold_detail,
        "locked_nested_audit": nested_audit,
        "fixed_production_threshold_audit": production_audit,
        "runtime_minutes": (time.monotonic() - started) / 60,
        "input_sha256": {
            "embeddings": sha256(EMBEDDINGS),
            "qwen3vl": sha256(QWEN3VL),
            "qwen35": sha256(QWEN35),
            "locked": sha256(LOCKED),
            "folds": sha256(FOLDS),
        },
    }
    report["accepted_screen"] = all(nested_audit["acceptance"].values()) and all(
        production_audit["acceptance"].values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_npz,
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        group_hashes=group_hashes,
        raw_prototype_score=raw_prototype_score,
        prototype_rank=prototype_rank,
        nested_baseline_predictions=nested_baseline,
        nested_candidate_predictions=nested_candidate,
        production_baseline_predictions=production_baseline,
        production_candidate_predictions=production_candidate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

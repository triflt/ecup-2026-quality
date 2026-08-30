from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shingle_neighbor_prior_cv import (
    apply_neighbor_prior,
    base_prior_predictions,
    build_neighbor_graph,
    canonical,
    compose_text,
    f1,
    fingerprint,
    normalize,
)


BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"
PRODUCTION_SHINGLE_CONFIG = (0.95, 2, 3, 0.999)


def prepare_frame(data_path: Path, arrays: np.lib.npyio.NpzFile) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    frame["id"] = frame["id"].astype(str)
    if not np.array_equal(frame["id"].to_numpy(), arrays["ids"].astype(str)):
        raise ValueError("id mismatch between data and prediction arrays")
    if not np.array_equal(frame["label"].to_numpy(np.int8), arrays["labels"].astype(np.int8)):
        raise ValueError("label mismatch between data and prediction arrays")
    if not np.array_equal(frame["category"].astype(str).to_numpy(), arrays["categories"].astype(str)):
        raise ValueError("category mismatch between data and prediction arrays")
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["normalized_name"] = frame["name"].map(normalize)
    texts = [
        compose_text(name, description)
        for name, description in zip(frame["name"], frame["description"])
    ]
    frame["text_hash"] = [fingerprint(text) for text in texts]
    frame["canonical_text_mask_digits"] = [
        canonical(text, mask_digits=True) for text in texts
    ]
    frame["fold"] = arrays["folds"].astype(np.int8)
    return frame


def apply_downstream_priors(
    frame: pd.DataFrame,
    base_predictions: np.ndarray,
    neighbor_indices: list[np.ndarray],
    neighbor_scores: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    folds = frame["fold"].to_numpy(np.int8)
    categories = frame["category"].astype(str).to_numpy()
    result = base_predictions.copy()
    audit: dict[str, dict[str, int]] = {}
    for category in (BAD, FLAMMABLE):
        category_mask = categories == category
        counters = {
            "rows": int(category_mask.sum()),
            "exact_hits": 0,
            "name_hits": 0,
            "canonical_hits": 0,
            "shingle_hits": 0,
            "shingle_changes": 0,
        }
        for fold in sorted(np.unique(folds)):
            donors = np.flatnonzero(category_mask & (folds != fold))
            targets = np.flatnonzero(category_mask & (folds == fold))
            predictions, unresolved, hits = base_prior_predictions(
                frame, base_predictions, donors, targets, category
            )
            for key in ("exact", "name", "canonical"):
                counters[f"{key}_hits"] += int(hits[key])
            if category == BAD:
                predictions, shingle_hits, shingle_changes = apply_neighbor_prior(
                    frame,
                    donors,
                    targets,
                    predictions,
                    unresolved,
                    neighbor_indices,
                    neighbor_scores,
                    PRODUCTION_SHINGLE_CONFIG,
                )
                counters["shingle_hits"] += int(shingle_hits)
                counters["shingle_changes"] += int(shingle_changes)
            result[targets] = predictions
        audit[category] = counters
    return result, audit


def scores(labels: np.ndarray, predictions: np.ndarray, categories: np.ndarray) -> dict[str, float]:
    per_category = {
        category: f1(labels[categories == category], predictions[categories == category])
        for category in (BAD, FLAMMABLE)
    }
    return {
        "bad_f1": float(per_category[BAD]),
        "flammable_f1": float(per_category[FLAMMABLE]),
        "macro_f1": float(np.mean(list(per_category.values()))),
    }


def changed_audit(
    labels: np.ndarray,
    categories: np.ndarray,
    baseline_before: np.ndarray,
    candidate_before: np.ndarray,
    baseline_after: np.ndarray,
    candidate_after: np.ndarray,
) -> dict[str, object]:
    changed_before = baseline_before != candidate_before
    changed_after = baseline_after != candidate_after
    overridden = changed_before & ~changed_after
    survived = changed_before & changed_after
    introduced = ~changed_before & changed_after

    def summarize(mask: np.ndarray) -> dict[str, int]:
        return {
            "rows": int(mask.sum()),
            "bad_rows": int((mask & (categories == BAD)).sum()),
            "flammable_rows": int((mask & (categories == FLAMMABLE)).sum()),
        }

    before_corrected = changed_before & (candidate_before == labels) & (baseline_before != labels)
    before_regressed = changed_before & (candidate_before != labels) & (baseline_before == labels)
    after_corrected = changed_after & (candidate_after == labels) & (baseline_after != labels)
    after_regressed = changed_after & (candidate_after != labels) & (baseline_after == labels)
    return {
        "changed_before_priors": summarize(changed_before),
        "overridden_by_priors": summarize(overridden),
        "survived_after_priors": summarize(survived),
        "introduced_by_different_prior_resolution": summarize(introduced),
        "survival_rate": float(survived.sum() / changed_before.sum()) if changed_before.any() else 0.0,
        "before_priors": {
            "corrected": int(before_corrected.sum()),
            "regressed": int(before_regressed.sum()),
        },
        "after_priors": {
            "corrected": int(after_corrected.sum()),
            "regressed": int(after_regressed.sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--candidate", default="exp260")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    arrays = np.load(args.predictions, allow_pickle=False)
    frame = prepare_frame(args.data, arrays)
    neighbor_indices, neighbor_scores = build_neighbor_graph(frame)
    baseline_before = arrays["baseline_nested_predictions"].astype(np.int8)
    candidate_before = arrays[f"{args.candidate}_nested_predictions"].astype(np.int8)
    baseline_after, baseline_prior_audit = apply_downstream_priors(
        frame, baseline_before, neighbor_indices, neighbor_scores
    )
    candidate_after, candidate_prior_audit = apply_downstream_priors(
        frame, candidate_before, neighbor_indices, neighbor_scores
    )
    labels = frame["label"].to_numpy(np.int8)
    categories = frame["category"].astype(str).to_numpy()
    report = {
        "protocol": "donor-only outer-fold replay of Public-190 exact/name/numeric-family/shingle priors",
        "candidate": args.candidate,
        "shingle_config": list(PRODUCTION_SHINGLE_CONFIG),
        "baseline_before_priors": scores(labels, baseline_before, categories),
        "candidate_before_priors": scores(labels, candidate_before, categories),
        "baseline_after_priors": scores(labels, baseline_after, categories),
        "candidate_after_priors": scores(labels, candidate_after, categories),
        "change_survival": changed_audit(
            labels,
            categories,
            baseline_before,
            candidate_before,
            baseline_after,
            candidate_after,
        ),
        "baseline_prior_hits": baseline_prior_audit,
        "candidate_prior_hits": candidate_prior_audit,
        "limitations": [
            "Donor-only grouped OOF replay estimates downstream masking without hidden labels.",
            "Hidden repeat rate can differ from train, so this audit explains mechanism but cannot reconstruct Public predictions.",
        ],
    }
    report["delta_before_priors"] = {
        key: report["candidate_before_priors"][key] - report["baseline_before_priors"][key]
        for key in ("bad_f1", "flammable_f1", "macro_f1")
    }
    report["delta_after_priors"] = {
        key: report["candidate_after_priors"][key] - report["baseline_after_priors"][key]
        for key in ("bad_f1", "flammable_f1", "macro_f1")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

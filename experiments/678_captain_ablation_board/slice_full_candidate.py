from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FLAMMABLE = "Легковоспламеняющиеся"
CATEGORIES = ("БАД", FLAMMABLE)
REQUIRED_PREDICTION_FIELDS = {"global_index", "id", "fold", "category", "score", "prediction"}
FORBIDDEN_PREDICTION_FIELDS = {"label", "target", "gold", "answer", "sealed"}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_predictions(paths: list[Path], registry: pd.DataFrame) -> pd.DataFrame:
    records = [row for path in paths for row in read_jsonl(path)]
    if any(
        not REQUIRED_PREDICTION_FIELDS <= set(row) or FORBIDDEN_PREDICTION_FIELDS & set(row)
        for row in records
    ):
        raise ValueError("prediction schema or supervision mismatch")
    frame = pd.DataFrame(
        [{key: row[key] for key in REQUIRED_PREDICTION_FIELDS} for row in records]
    ).sort_values("global_index").reset_index(drop=True)
    expected_indices = registry["global_index"].astype(int).tolist()
    if frame["global_index"].astype(int).tolist() != expected_indices:
        raise ValueError("predictions do not exactly cover development rows")
    if frame["id"].astype(str).tolist() != registry["id"].astype(str).tolist():
        raise ValueError("prediction IDs differ from immutable registry")
    if frame["fold"].astype(int).tolist() != registry["development_fold"].astype(int).tolist():
        raise ValueError("prediction folds differ from immutable registry")
    if frame["category"].astype(str).tolist() != registry["category"].astype(str).tolist():
        raise ValueError("prediction categories differ from immutable registry")
    if frame["id"].astype(str).duplicated().any() or not np.isfinite(frame["score"]).all():
        raise ValueError("duplicate IDs or non-finite scores")
    if not np.array_equal(frame["prediction"].astype(int), frame["score"].to_numpy() >= 0.0):
        raise ValueError("prediction differs from frozen zero threshold")
    return frame


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(((labels == 1) & predictions).sum())
    fp = int(((labels == 0) & predictions).sum())
    fn = int(((labels == 1) & ~predictions).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    if len(labels) == 0 or positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    true_positives = np.cumsum(ranked_labels == 1)
    threshold_ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(ranked_scores) - 1]
    precision = true_positives[threshold_ends] / (threshold_ends + 1)
    recall = true_positives[threshold_ends] / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def ocr_cohort_by_id(path: Path) -> dict[str, str]:
    statuses: dict[str, list[str]] = defaultdict(list)
    keys: set[tuple[str, int]] = set()
    for row in read_jsonl(path):
        key = (str(row["id"]), int(row["image_index"]))
        if key in keys:
            raise ValueError("duplicate OCR availability image key")
        keys.add(key)
        status = str(row["status"])
        if status not in {"OCR_AVAILABLE", "OCR_UNAVAILABLE"}:
            raise ValueError("unknown OCR availability status")
        statuses[key[0]].append(status)
    result = {}
    for item_id, values in statuses.items():
        available = sum(value == "OCR_AVAILABLE" for value in values)
        if available == len(values):
            result[item_id] = "ocr_all_available"
        elif available == 0:
            result[item_id] = "ocr_all_unavailable"
        else:
            result[item_id] = "ocr_partially_available"
    return result


def metrics_for_mask(
    *,
    mask: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    baseline_score: np.ndarray,
    candidate_score: np.ndarray,
) -> dict[str, Any]:
    baseline_pred = baseline_score >= 0.0
    candidate_pred = candidate_score >= 0.0
    local_labels = labels[mask]
    local_categories = categories[mask]
    result: dict[str, Any] = {
        "rows": int(mask.sum()),
        "positives": int((local_labels == 1).sum()),
        "baseline_errors": int((baseline_pred[mask] != local_labels).sum()),
        "candidate_errors": int((candidate_pred[mask] != local_labels).sum()),
        "corrected": int(((baseline_pred != labels) & (candidate_pred == labels) & mask).sum()),
        "regressed": int(((baseline_pred == labels) & (candidate_pred != labels) & mask).sum()),
        "categories": {},
    }
    for category in CATEGORIES:
        local = mask & (categories == category)
        if not local.any():
            continue
        result["categories"][category] = {
            "rows": int(local.sum()),
            "positives": int((labels[local] == 1).sum()),
            "baseline_f1": f1(labels[local], baseline_pred[local]),
            "candidate_f1": f1(labels[local], candidate_pred[local]),
        }
        result["categories"][category]["delta"] = (
            result["categories"][category]["candidate_f1"]
            - result["categories"][category]["baseline_f1"]
        )
    flammable = mask & (categories == FLAMMABLE)
    result["flammable_average_precision"] = {
        "baseline": average_precision(labels[flammable], baseline_score[flammable]),
        "candidate": average_precision(labels[flammable], candidate_score[flammable]),
    }
    return result


def evaluate(
    *,
    registry_path: Path,
    baseline_paths: list[Path],
    large_paths: list[Path],
    ocr_availability_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite candidate slice report")
    registry = pd.read_csv(registry_path, dtype={"id": str})
    registry = registry.loc[registry["split"].astype(str).eq("development")].copy()
    registry = registry.sort_values("id", key=lambda values: values.astype(str)).reset_index(drop=True)
    registry["global_index"] = np.arange(len(registry), dtype=np.int64)
    baseline = load_predictions(baseline_paths, registry)
    large = load_predictions(large_paths, registry)

    labels = registry["label"].to_numpy(np.int8)
    categories = registry["category"].astype(str).to_numpy()
    baseline_score = baseline["score"].to_numpy(np.float64)
    large_score = large["score"].to_numpy(np.float64)
    candidate_score = baseline_score.copy()
    flammable = categories == FLAMMABLE
    candidate_score[flammable] = 0.5 * baseline_score[flammable] + 0.5 * large_score[flammable]

    component_label_count = registry.groupby("semantic_component")["label"].nunique().to_dict()
    mixed = registry["semantic_component"].map(component_label_count).astype(int).gt(1).to_numpy()
    singleton = registry["component_size"].astype(int).eq(1).to_numpy()
    ocr_by_id = ocr_cohort_by_id(ocr_availability_path)
    ocr_values = registry["id"].astype(str).map(ocr_by_id).fillna("ocr_no_images").to_numpy()
    folds = registry["development_fold"].astype(int).to_numpy()

    masks: dict[str, np.ndarray] = {
        "all": np.ones(len(registry), dtype=bool),
        "semantic_singleton": singleton,
        "semantic_repeated": ~singleton,
        "mixed_label_component": mixed,
        "consistent_label_component": ~mixed,
        "flammable_positive": flammable & (labels == 1),
        "flammable_negative": flammable & (labels == 0),
    }
    for cohort in (
        "ocr_all_available",
        "ocr_partially_available",
        "ocr_all_unavailable",
        "ocr_no_images",
    ):
        masks[cohort] = ocr_values == cohort
    for fold in range(5):
        masks[f"fold_{fold}"] = folds == fold

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "678",
        "candidate": "fixed_equal_logit_route",
        "weights": {"small": 0.5, "large": 0.5},
        "threshold": 0.0,
        "threshold_tuned": False,
        "registry_sha256": sha256_file(registry_path),
        "baseline_prediction_sha256": [sha256_file(path) for path in baseline_paths],
        "large_prediction_sha256": [sha256_file(path) for path in large_paths],
        "ocr_availability_sha256": sha256_file(ocr_availability_path),
        "slices": {
            name: metrics_for_mask(
                mask=mask,
                labels=labels,
                categories=categories,
                baseline_score=baseline_score,
                candidate_score=candidate_score,
            )
            for name, mask in masks.items()
        },
        "sealed_rows": 0,
        "public_used": False,
        "decision": "DIAGNOSTIC_ONLY_NO_SELECTION",
    }
    if any(not math.isfinite(float(value)) for value in candidate_score):
        raise ValueError("candidate scores are not finite")
    result["contract_sha256"] = canonical_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--baseline-score", type=Path, action="append", required=True)
    result.add_argument("--large-score", type=Path, action="append", required=True)
    result.add_argument("--ocr-availability", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            evaluate(
                registry_path=args.registry,
                baseline_paths=args.baseline_score,
                large_paths=args.large_score,
                ocr_availability_path=args.ocr_availability,
                output_path=args.output,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


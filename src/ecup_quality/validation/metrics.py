from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def binary_f1(labels: Iterable[int], predictions: Iterable[int]) -> float:
    labels_array = np.asarray(labels, dtype=np.int8)
    predictions_array = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels_array == 1) & (predictions_array == 1)).sum())
    fp = int(((labels_array == 0) & (predictions_array == 1)).sum())
    fn = int(((labels_array == 1) & (predictions_array == 0)).sum())
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def rank01(values: Iterable[float]) -> np.ndarray:
    values_array = np.asarray(values)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values_array), dtype=np.float32)
    return ranks


def category_report(labels: np.ndarray, predictions: np.ndarray, categories: np.ndarray) -> dict:
    result = {"categories": {}}
    scores = []
    for category in sorted(np.unique(categories)):
        mask = categories == category
        category_labels = labels[mask]
        category_predictions = predictions[mask]
        score = binary_f1(category_labels, category_predictions)
        scores.append(score)
        result["categories"][str(category)] = {
            "rows": int(mask.sum()),
            "positives": int(category_labels.sum()),
            "predicted_positives": int(category_predictions.sum()),
            "f1": score,
        }
    result["macro_f1"] = float(np.mean(scores))
    return result

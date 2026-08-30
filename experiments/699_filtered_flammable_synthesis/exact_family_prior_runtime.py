from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import soft_cache_runtime as base


SCHEMA = "exp699_exact_family_prior_runtime_v1"
PREREGISTER_SELF_SHA256 = (
    "4eeac3aa45049a5529d0f86585ce5140e925a728aca677684cd009619829e02e"
)
FAMILY_PRIORITY = ("exact_text", "image_exact", "normalized_text")
MINIMUM_DONORS = {
    "exact_text": 1,
    "image_exact": 2,
    "normalized_text": 2,
}
MAXIMUM_DONORS = 20


def _select_unanimous_family(
    channels: dict[str, list[int]], donor_labels: np.ndarray
) -> tuple[str, list[int], int] | None:
    for channel in FAMILY_PRIORITY:
        donors = sorted(set(int(value) for value in channels[channel]))
        if len(donors) < MINIMUM_DONORS[channel]:
            continue
        if len(donors) > MAXIMUM_DONORS:
            return None
        labels = np.asarray(donor_labels, dtype=np.int8)[donors]
        unique = np.unique(labels)
        if len(unique) != 1:
            return None
        return channel, donors, int(unique[0])
    return None


def apply_runtime_exact_family_prior(
    frame: Any,
    baseline_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cache = base.load_cache(cache_path)
    categories = frame["category"].astype(str).to_numpy()
    scores = np.asarray(baseline_scores, dtype=np.float64).copy()
    predictions = np.asarray(baseline_predictions, dtype=np.int8).copy()
    flammable_positions = np.flatnonzero(categories == base.FLAMMABLE)
    if len(flammable_positions) == 0:
        return scores, predictions, []

    texts = [
        base.row_text(
            frame.iloc[index]["name"],
            frame.iloc[index]["description"],
            mask_digits=False,
        )
        for index in flammable_positions
    ]
    normalized = [
        base.row_text(
            frame.iloc[index]["name"],
            frame.iloc[index]["description"],
            mask_digits=True,
        )
        for index in flammable_positions
    ]
    paths: list[Path | None] = []
    for index in flammable_positions:
        values = frame.iloc[index]["image_paths"]
        paths.append(None if not values else Path(values[0]))
    image_exact, _ = base._image_channels(paths, cache)
    exact_text = base._exact_channel(texts, cache["exact_text_groups"])
    normalized_text = base._exact_channel(
        normalized, cache["normalized_text_groups"]
    )
    donor_labels = np.asarray(cache["donor_labels"], dtype=np.int8)
    audit: list[dict[str, Any]] = []
    for local, global_index in enumerate(flammable_positions):
        channels = {
            "exact_text": exact_text[local],
            "image_exact": image_exact[local],
            "normalized_text": normalized_text[local],
        }
        selected = _select_unanimous_family(channels, donor_labels)
        if selected is None:
            continue
        channel, donors, label = selected
        old_score = float(scores[global_index])
        old_prediction = int(predictions[global_index])
        scores[global_index] = float(label)
        predictions[global_index] = label
        audit.append(
            {
                "row_index": int(global_index),
                "channel": channel,
                "donor_count": len(donors),
                "donors": donors,
                "unanimous_label": label,
                "baseline_score": old_score,
                "candidate_score": float(label),
                "baseline_prediction": old_prediction,
                "candidate_prediction": label,
            }
        )
    if not np.array_equal(
        predictions[categories != base.FLAMMABLE],
        np.asarray(baseline_predictions, dtype=np.int8)[categories != base.FLAMMABLE],
    ):
        raise ValueError("exact-family prior changed frozen BAD route")
    return scores, predictions, audit

from __future__ import annotations

import random

import numpy as np


def _fused_scores(oof) -> np.ndarray:
    key = "fused_scores" if "fused_scores" in oof.files else "fused"
    return oof[key].astype(np.float32)


def _thresholds(oof, categories: np.ndarray) -> np.ndarray:
    values = (
        {"БАД": 0.24864045896205267, "Легковоспламеняющиеся": 0.9591804083988902}
        if "fused_scores" in oof.files
        else {"БАД": 0.251901438832283, "Легковоспламеняющиеся": 0.9540719747543336}
    )
    return np.asarray([values[value] for value in categories], dtype=np.float32)


def _hard_random(indices, scores, count, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= count:
        return indices.tolist()
    hard_count = count // 2
    hard = indices[np.argsort(scores[indices])[:hard_count]]
    remaining = np.setdiff1d(indices, hard, assume_unique=False)
    random_part = rng.choice(remaining, size=count - hard_count, replace=False)
    return np.concatenate([hard, random_part]).tolist()


def select_parent_training_records(frame, oof, *, seed: int, holdout_fold: int) -> list[int]:
    rng = np.random.default_rng(seed)
    categories = frame["category"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(np.int8)
    folds = oof["fold_ids"].astype(np.int8)
    uncertainty = np.abs(_fused_scores(oof) - _thresholds(oof, categories))
    train = folds != holdout_fold
    records: list[int] = []
    bad = np.flatnonzero(train & (categories == "БАД"))
    bad_pos, bad_neg = bad[labels[bad] == 1], bad[labels[bad] == 0]
    bad_count = min(1500, len(bad_pos), len(bad_neg))
    records.extend(_hard_random(bad_pos, uncertainty, bad_count, rng))
    records.extend(_hard_random(bad_neg, uncertainty, bad_count, rng))
    flammable = np.flatnonzero(train & (categories == "Легковоспламеняющиеся"))
    flam_pos, flam_neg = flammable[labels[flammable] == 1], flammable[labels[flammable] == 0]
    records.extend(np.repeat(flam_pos, 5).tolist())
    records.extend(_hard_random(flam_neg, uncertainty, min(1600, len(flam_neg)), rng))
    random.Random(seed).shuffle(records)
    return records

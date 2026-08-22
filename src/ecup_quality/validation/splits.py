from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ecup_quality.data.text import compose_text, normalize_text


def product_group(name: object, description: object) -> str:
    return normalize_text(compose_text(name, description))


def assign_grouped_folds(frame: pd.DataFrame, *, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Assign deterministic category-specific StratifiedGroupKFold folds."""
    required = {"category", "label", "name", "description"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    folds = np.full(len(frame), -1, dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int8)
    groups = np.asarray([
        product_group(name, description)
        for name, description in zip(frame["name"], frame["description"])
    ])
    for category in sorted(np.unique(categories)):
        positions = np.flatnonzero(categories == category)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (_, valid_local) in enumerate(
            splitter.split(np.zeros(len(positions)), labels[positions], groups[positions])
        ):
            folds[positions[valid_local]] = fold
    if (folds < 0).any():
        raise RuntimeError("incomplete fold assignment")
    return folds

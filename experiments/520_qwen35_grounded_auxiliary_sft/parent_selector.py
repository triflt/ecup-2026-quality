from __future__ import annotations

import html
import random
import re
import unicodedata

import numpy as np
import pandas as pd

SELECTOR_VERSION = "exp260_fixed_hard_selector_dependency_light_v1"


def fused_oof_scores(oof) -> np.ndarray:
    key = "fused_scores" if "fused_scores" in oof.files else "fused"
    return oof[key].astype(np.float32)


def oof_thresholds(oof, categories: np.ndarray) -> np.ndarray:
    values = (
        {"БАД": 0.24864045896205267, "Легковоспламеняющиеся": 0.9591804083988902}
        if "fused_scores" in oof.files
        else {"БАД": 0.251901438832283, "Легковоспламеняющиеся": 0.9540719747543336}
    )
    return np.asarray([values[category] for category in categories], dtype=np.float32)


def hard_random(indices, scores: np.ndarray, count: int, rng) -> list[int]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= count:
        return indices.tolist()
    hard_count = count // 2
    hard = indices[np.argsort(scores[indices])[:hard_count]]
    remaining = np.setdiff1d(indices, hard, assume_unique=False)
    random_part = rng.choice(remaining, size=count - hard_count, replace=False)
    return np.concatenate([hard, random_part]).tolist()


def normalize_group_text(value) -> str:
    value = "" if pd.isna(value) else html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def family_key(row) -> str:
    name = normalize_group_text(row["name"])
    description = normalize_group_text(row["description"])
    return f"{name}\n{name}\n{description}"


def family_diverse_hard_random(frame, indices, scores: np.ndarray, count: int, rng):
    groups: dict[str, list[int]] = {}
    for index in np.asarray(indices, dtype=np.int64):
        groups.setdefault(family_key(frame.iloc[index]), []).append(int(index))
    if len(groups) < count:
        raise ValueError(f"need {count} distinct negative families, found only {len(groups)}")
    representatives = {
        key: min(local, key=lambda index: float(scores[index])) for key, local in groups.items()
    }
    hard_count = count // 2
    ordered = sorted(representatives, key=lambda key: float(scores[representatives[key]]))
    hard_keys = ordered[:hard_count]
    remaining_keys = np.asarray(ordered[hard_count:], dtype=object)
    random_keys = rng.choice(remaining_keys, size=count - hard_count, replace=False).tolist()
    selected = [representatives[key] for key in hard_keys]
    selected.extend(int(rng.choice(groups[key])) for key in random_keys)
    return selected, {
        "eligible_rows": len(indices),
        "eligible_families": len(groups),
        "selected_rows": len(selected),
        "selected_families": len({family_key(frame.iloc[index]) for index in selected}),
        "hard_families": len(hard_keys),
        "random_families": len(random_keys),
        "max_family_exposures": 1,
    }


def select_parent_training_records(
    frame: pd.DataFrame,
    oof,
    *,
    seed: int,
    holdout_fold: int,
    full_train: bool,
) -> tuple[list[int], dict]:
    rng = np.random.default_rng(seed)
    categories = frame["category"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int8)
    folds = oof["fold_ids"].astype(np.int8)
    uncertainty = np.abs(fused_oof_scores(oof) - oof_thresholds(oof, categories))
    train_mask = np.ones(len(folds), dtype=bool) if full_train else folds != holdout_fold
    records: list[int] = []

    bad = np.flatnonzero(train_mask & (categories == "БАД"))
    bad_pos = bad[labels[bad] == 1]
    bad_neg = bad[labels[bad] == 0]
    bad_count = min(1900 if full_train else 1500, len(bad_neg), len(bad_pos))
    hard_random(bad_pos, uncertainty, bad_count, rng)
    family_rng = np.random.default_rng(seed + 26001)
    selected_bad_positives, bad_positive_selection_audit = family_diverse_hard_random(
        frame, bad_pos, uncertainty, bad_count, family_rng
    )
    bad_positive_selection_audit["family_diverse"] = True
    bad_positive_selection_audit["control_rng_consumed"] = True
    records.extend(selected_bad_positives)
    records.extend(hard_random(bad_neg, uncertainty, bad_count, rng))

    flam = np.flatnonzero(train_mask & (categories == "Легковоспламеняющиеся"))
    flam_pos = flam[labels[flam] == 1]
    flam_neg = flam[labels[flam] == 0]
    records.extend(np.repeat(flam_pos, 5).tolist())
    selection_audit = {
        "positive_rows": len(flam_pos),
        "positive_families": None,
        "total_exposures": int(len(flam_pos) * 5),
        "min_family_exposures": None,
        "max_family_exposures": None,
    }
    flam_neg_count = min(2000 if full_train else 1600, len(flam_neg))
    records.extend(hard_random(flam_neg, uncertainty, flam_neg_count, rng))
    random.Random(seed).shuffle(records)
    selection_audit["family_balanced"] = False
    selection_audit["bad_positive_selection"] = bad_positive_selection_audit
    selection_audit["negative_selection"] = {
        "family_diverse": False,
        "selected_rows": int(flam_neg_count),
    }
    selection_audit["selector_version"] = SELECTOR_VERSION
    return records, selection_audit

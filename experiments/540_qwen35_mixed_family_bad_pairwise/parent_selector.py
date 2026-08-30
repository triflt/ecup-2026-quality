from __future__ import annotations

import html
import random
import re
import unicodedata

import numpy as np
import pandas as pd


def fused_scores(oof):
    return oof["fused_scores" if "fused_scores" in oof.files else "fused"].astype(np.float32)


def thresholds(oof, categories):
    values = (
        {"БАД": 0.24864045896205267, "Легковоспламеняющиеся": 0.9591804083988902}
        if "fused_scores" in oof.files
        else {"БАД": 0.251901438832283, "Легковоспламеняющиеся": 0.9540719747543336}
    )
    return np.asarray([values[value] for value in categories], dtype=np.float32)


def hard_random(indices, scores, count, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= count:
        return indices.tolist()
    hard_count = count // 2
    hard = indices[np.argsort(scores[indices])[:hard_count]]
    remaining = np.setdiff1d(indices, hard, assume_unique=False)
    return np.concatenate(
        [hard, rng.choice(remaining, size=count - hard_count, replace=False)]
    ).tolist()


def normalize_group_text(value):
    value = "" if pd.isna(value) else html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def family_key(row):
    name = normalize_group_text(row["name"])
    return f"{name}\n{name}\n{normalize_group_text(row['description'])}"


def family_diverse(frame, indices, scores, count, rng):
    groups = {}
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
    random_keys = rng.choice(
        np.asarray(ordered[hard_count:], dtype=object),
        size=count - hard_count,
        replace=False,
    ).tolist()
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
        "family_diverse": True,
        "control_rng_consumed": True,
    }


def select_parent_training_records(frame, oof, *, seed, holdout_fold):
    rng = np.random.default_rng(seed)
    categories = frame["category"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(np.int8)
    folds = oof["fold_ids"].astype(np.int8)
    uncertainty = np.abs(fused_scores(oof) - thresholds(oof, categories))
    train = folds != holdout_fold
    records = []

    bad = np.flatnonzero(train & (categories == "БАД"))
    bad_pos = bad[labels[bad] == 1]
    bad_neg = bad[labels[bad] == 0]
    count = min(1500, len(bad_pos), len(bad_neg))
    hard_random(bad_pos, uncertainty, count, rng)
    selected_pos, bad_audit = family_diverse(
        frame, bad_pos, uncertainty, count, np.random.default_rng(seed + 26001)
    )
    records.extend(selected_pos)
    records.extend(hard_random(bad_neg, uncertainty, count, rng))

    flam = np.flatnonzero(train & (categories == "Легковоспламеняющиеся"))
    flam_pos = flam[labels[flam] == 1]
    flam_neg = flam[labels[flam] == 0]
    records.extend(np.repeat(flam_pos, 5).tolist())
    records.extend(hard_random(flam_neg, uncertainty, min(1600, len(flam_neg)), rng))
    random.Random(seed).shuffle(records)
    return records, {
        "positive_rows": len(flam_pos),
        "total_exposures": int(len(flam_pos) * 5),
        "family_balanced": False,
        "bad_positive_selection": bad_audit,
        "negative_selection": {
            "family_diverse": False,
            "selected_rows": int(min(1600, len(flam_neg))),
        },
        "selector_version": "exp260_fixed_hard_selector_dependency_light_v1",
    }

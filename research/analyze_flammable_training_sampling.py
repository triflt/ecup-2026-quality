from __future__ import annotations

import argparse
import html
import json
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


SIGNALS = {
    "gas": r"(?iu)\b(?:газ|пропан|бутан|баллон|картридж)\w*\b",
    "burner": r"(?iu)\b(?:горелк|плит|печ|примус)\w*\b",
    "lighter": r"(?iu)\b(?:зажигалк|спич|огнив|факел)\w*\b",
    "fuel_liquid": r"(?iu)\b(?:топлив|бензин|керосин|жидкост\w*\s+для\s+розжиг)\w*\b",
    "candle": r"(?iu)\b(?:свеч|воск|парафин)\w*\b",
    "kit": r"(?iu)\b(?:комплект|набор|входит|поставк)\w*\b",
    "charcoal": r"(?iu)\b(?:угол|уголь|брик|дров)\w*\b",
}


def normalize_group_text(value: object) -> str:
    value = "" if pd.isna(value) else html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def family_key(row: pd.Series) -> str:
    name = normalize_group_text(row["name"])
    description = normalize_group_text(row["description"])
    return f"{name}\n{name}\n{description}"


def hard_random(indices, uncertainty, count, rng):
    indices = np.asarray(indices, dtype=np.int64)
    hard_count = count // 2
    hard = indices[np.argsort(uncertainty[indices])[:hard_count]]
    remaining = np.setdiff1d(indices, hard, assume_unique=False)
    random_part = rng.choice(remaining, size=count - hard_count, replace=False)
    return np.concatenate([hard, random_part]), hard


def consume_family_balanced_positive_rng(frame, indices, total_count, rng):
    groups = {}
    for index in np.asarray(indices, dtype=np.int64):
        groups.setdefault(family_key(frame.iloc[index]), []).append(int(index))
    keys = list(groups)
    rng.shuffle(keys)
    base, remainder = divmod(total_count, len(keys))
    for position, key in enumerate(keys):
        count = base + int(position < remainder)
        rng.choice(groups[key], size=count, replace=True)


def rng_at_negative_selection(frame, labels, categories, fold_ids, uncertainty, holdout):
    rng = np.random.default_rng(42)
    train_mask = fold_ids != holdout
    bad = np.flatnonzero(train_mask & (categories == "БАД"))
    bad_pos = bad[labels[bad] == 1]
    bad_neg = bad[labels[bad] == 0]
    bad_count = min(1500, len(bad_neg), len(bad_pos))
    hard_random(bad_pos, uncertainty, bad_count, rng)
    hard_random(bad_neg, uncertainty, bad_count, rng)
    flammable = np.flatnonzero(train_mask & (categories == "Легковоспламеняющиеся"))
    # Experiment 241 was rejected, so experiment 250 inherits the original
    # row-repeated positive sampler. It consumes no NumPy RNG state.
    return rng, flammable[labels[flammable] == 0]


def family_diverse(indices, frame, uncertainty, count, rng):
    groups = {}
    for index in np.asarray(indices, dtype=np.int64):
        groups.setdefault(family_key(frame.iloc[index]), []).append(int(index))
    representatives = {
        key: min(local, key=lambda index: float(uncertainty[index]))
        for key, local in groups.items()
    }
    ordered = sorted(representatives, key=lambda key: float(uncertainty[representatives[key]]))
    hard_count = count // 2
    hard_keys = ordered[:hard_count]
    random_keys = rng.choice(
        np.asarray(ordered[hard_count:], dtype=object),
        size=count - hard_count,
        replace=False,
    ).tolist()
    hard = np.asarray([representatives[key] for key in hard_keys], dtype=np.int64)
    random_rows = np.asarray(
        [int(rng.choice(groups[key])) for key in random_keys], dtype=np.int64
    )
    return np.concatenate([hard, random_rows]), hard


def selection_row(name, selected, hard, frame, uncertainty):
    keys = pd.Series([family_key(frame.iloc[index]) for index in selected])
    counts = keys.value_counts()
    text = (
        frame.iloc[selected]["name"].fillna("").astype(str)
        + "\n"
        + frame.iloc[selected]["description"].fillna("").astype(str)
    )
    result = {
        "selection": name,
        "selected_rows": int(len(selected)),
        "selected_families": int(counts.size),
        "duplicate_exposures": int((counts - 1).clip(lower=0).sum()),
        "families_with_at_least_3_rows": int((counts >= 3).sum()),
        "max_family_exposures": int(counts.max()),
        "hard_rows": int(len(hard)),
        "hard_families": int(pd.Series([family_key(frame.iloc[index]) for index in hard]).nunique()),
        "mean_uncertainty": float(np.mean(uncertainty[selected])),
        "median_uncertainty": float(np.median(uncertainty[selected])),
        "hard_mean_uncertainty": float(np.mean(uncertainty[hard])),
    }
    for signal, pattern in SIGNALS.items():
        result[f"signal_{signal}_rows"] = int(text.str.contains(pattern, regex=True, na=False).sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("research/data.csv"))
    parser.add_argument(
        "--oof",
        type=Path,
        default=Path("research/four-head-r2-extracted/four_head_oof.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/250_flammable_family_diverse_negatives/analysis"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.data, dtype={"id": str})
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    oof = np.load(args.oof, allow_pickle=True)
    if not np.array_equal(frame.id.to_numpy(), oof["ids"].astype(str)):
        raise ValueError("data/OOF id mismatch")
    labels = frame.label.to_numpy(np.int8)
    categories = frame.category.astype(str).to_numpy()
    fold_ids = oof["fold_ids"].astype(np.int8)
    fused = oof["fused_scores" if "fused_scores" in oof.files else "fused"].astype(np.float32)
    thresholds = np.asarray(
        [
            0.24864045896205267 if category == "БАД" else 0.9591804083988902
            for category in categories
        ],
        dtype=np.float32,
    )
    uncertainty = np.abs(fused - thresholds)

    rows = []
    for holdout in range(5):
        old_rng, negatives = rng_at_negative_selection(
            frame, labels, categories, fold_ids, uncertainty, holdout
        )
        new_rng, new_negatives = rng_at_negative_selection(
            frame, labels, categories, fold_ids, uncertainty, holdout
        )
        if not np.array_equal(negatives, new_negatives):
            raise ValueError("selection setup is not deterministic")
        old_selected, old_hard = hard_random(negatives, uncertainty, 1600, old_rng)
        new_selected, new_hard = family_diverse(
            new_negatives, frame, uncertainty, 1600, new_rng
        )
        for item in (
            selection_row("row_hard_random", old_selected, old_hard, frame, uncertainty),
            selection_row("family_hard_random", new_selected, new_hard, frame, uncertainty),
        ):
            item["fold"] = holdout
            item["eligible_rows"] = int(len(negatives))
            item["eligible_families"] = int(
                pd.Series([family_key(frame.iloc[index]) for index in negatives]).nunique()
            )
            rows.append(item)

    audit = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "sampling_by_fold.csv", index=False)
    old = audit[audit.selection == "row_hard_random"]
    new = audit[audit.selection == "family_hard_random"]
    report = {
        "data_version": "competition_train_v1",
        "fold_version": "grouped_text_v1",
        "old_selected_family_range": [
            int(old.selected_families.min()), int(old.selected_families.max())
        ],
        "old_duplicate_exposure_range": [
            int(old.duplicate_exposures.min()), int(old.duplicate_exposures.max())
        ],
        "old_hard_duplicate_exposure_range": [
            int((old.hard_rows - old.hard_families).min()),
            int((old.hard_rows - old.hard_families).max()),
        ],
        "old_max_family_exposure": int(old.max_family_exposures.max()),
        "new_selected_families": int(new.selected_families.min()),
        "new_max_family_exposure": int(new.max_family_exposures.max()),
        "mean_uncertainty_change": float(
            new.mean_uncertainty.mean() - old.mean_uncertainty.mean()
        ),
        "hard_mean_uncertainty_change": float(
            new.hard_mean_uncertainty.mean() - old.hard_mean_uncertainty.mean()
        ),
        "interpretation": (
            "The family-level selector preserves 800 hard and 800 random negatives, "
            "but replaces repeated rows with independent normalized product families."
        ),
    }
    (args.output_dir / "sampling_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

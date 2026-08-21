from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("research")
DATA = Path(os.environ.get("ECUP_DATA", ROOT / "data.csv"))
FUSION = ROOT / "qwen3vl-qwen35-5fold-nested-fusion.npz"
REPORT = ROOT / "qwen3vl-qwen35-fuzzy-prior-report.json"
BASE_PRIOR = ROOT / "qwen3vl-qwen35-annotator-prior.json"
PRIOR = ROOT / "qwen3vl-qwen35-fuzzy-annotator-prior.json"

BASE_CONFIGS = {
    "БАД": (2, 2 / 3, 2, 0.999),
    "Легковоспламеняющиеся": (1, 0.999, 999, 0.999),
}


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def compose_text(name: object, description: object) -> str:
    normalized_name = normalize(name)
    return f"{normalized_name}\n{normalized_name}\n{normalize(description)}"


def fingerprint(value: object) -> str:
    return hashlib.sha1(normalize(value).encode("utf-8")).hexdigest()


def canonical(value: object, *, mask_digits: bool = False) -> str:
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    if mask_digits:
        value = re.sub(r"\d+(?:[.,]\d+)?", " # ", value)
    value = re.sub(r"[^0-9a-zа-я#]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def token_signature(value: object) -> str:
    tokens = [token for token in canonical(value).split() if len(token) >= 3]
    return " ".join(sorted(set(tokens)))


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def build_mapping(
    frame: pd.DataFrame,
    donors: np.ndarray,
    key: str,
    minimum: int,
    confidence_min: float,
) -> dict[str, int]:
    groups = frame.iloc[donors].groupby(key, sort=False).label.agg(["count", "mean"])
    confidence = np.maximum(groups["mean"], 1 - groups["mean"])
    keep = (
        (groups["count"] >= minimum)
        & (confidence >= confidence_min)
        & (groups["mean"] != 0.5)
        & (groups.index.astype(str) != "")
    )
    selected = groups.loc[keep, "mean"]
    return {str(key_value): int(mean >= 0.5) for key_value, mean in selected.items()}


def apply_mapping(
    frame: pd.DataFrame,
    targets: np.ndarray,
    predictions: np.ndarray,
    unresolved: np.ndarray,
    mapping: dict[str, int],
    key: str,
) -> int:
    hits = 0
    values = frame.iloc[targets][key].astype(str).to_numpy()
    for local_index, value in enumerate(values):
        if unresolved[local_index] and value in mapping:
            predictions[local_index] = mapping[value]
            unresolved[local_index] = False
            hits += 1
    return hits


def predict(
    frame: pd.DataFrame,
    base: np.ndarray,
    donors: np.ndarray,
    targets: np.ndarray,
    category: str,
    extra: tuple[str, int, float] | None,
) -> tuple[np.ndarray, dict[str, int]]:
    exact_min, exact_conf, name_min, name_conf = BASE_CONFIGS[category]
    predictions = base[targets].copy()
    unresolved = np.ones(len(targets), dtype=bool)
    hits: dict[str, int] = {}
    exact = build_mapping(frame, donors, "text_hash", exact_min, exact_conf)
    hits["exact"] = apply_mapping(
        frame, targets, predictions, unresolved, exact, "text_hash"
    )
    if name_min < 999:
        name = build_mapping(frame, donors, "normalized_name", name_min, name_conf)
        hits["name"] = apply_mapping(
            frame, targets, predictions, unresolved, name, "normalized_name"
        )
    else:
        hits["name"] = 0
    if extra is not None:
        key, minimum, confidence = extra
        mapping = build_mapping(frame, donors, key, minimum, confidence)
        hits[key] = apply_mapping(frame, targets, predictions, unresolved, mapping, key)
    return predictions, hits


def candidate_grid(category: str) -> list[tuple[str, int, float]]:
    candidates: list[tuple[str, int, float]] = []
    for key in ["canonical_text", "canonical_text_mask_digits", "token_signature"]:
        for minimum in [1, 2, 3, 4, 6]:
            for confidence in [2 / 3, 0.75, 0.9, 0.999]:
                candidates.append((key, minimum, confidence))
    for key in ["canonical_name", "canonical_name_mask_digits"]:
        for minimum in [2, 3, 5, 8]:
            for confidence in [0.75, 0.9, 0.999]:
                candidates.append((key, minimum, confidence))
    if category == "Легковоспламеняющиеся":
        # Positive examples are scarce; conservative unanimity is safer for broad keys.
        candidates = [row for row in candidates if row[2] >= 0.9]
    return candidates


def choose_extra_nested(
    frame: pd.DataFrame,
    base: np.ndarray,
    category: str,
    outer_fold: int,
) -> tuple[tuple[str, int, float], float, int]:
    labels = frame.label.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy(dtype=np.int8)
    category_mask = frame.category.to_numpy() == category
    outer_train = category_mask & (folds != outer_fold)
    best: tuple[tuple[float, int, int, float, str], tuple[str, int, float], float, int] | None = None
    for config in candidate_grid(category):
        inner_labels: list[np.ndarray] = []
        inner_predictions: list[np.ndarray] = []
        extra_hits = 0
        for inner_fold in sorted(frame.fold.unique()):
            if inner_fold == outer_fold:
                continue
            donors = np.flatnonzero(outer_train & (folds != inner_fold))
            targets = np.flatnonzero(outer_train & (folds == inner_fold))
            predictions, hits = predict(frame, base, donors, targets, category, config)
            inner_labels.append(labels[targets])
            inner_predictions.append(predictions)
            extra_hits += hits.get(config[0], 0)
        value = f1(np.concatenate(inner_labels), np.concatenate(inner_predictions))
        preference = (value, -extra_hits, config[1], config[2], config[0])
        if best is None or preference > best[0]:
            best = (preference, config, value, extra_hits)
    assert best is not None
    return best[1], best[2], best[3]


def evaluate_fixed_oof(
    frame: pd.DataFrame,
    base: np.ndarray,
    category: str,
    config: tuple[str, int, float] | None,
) -> tuple[float, int, np.ndarray]:
    labels = frame.label.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy(dtype=np.int8)
    category_mask = frame.category.to_numpy() == category
    predictions = base.copy()
    extra_hits = 0
    for fold in sorted(frame.fold.unique()):
        donors = np.flatnonzero(category_mask & (folds != fold))
        targets = np.flatnonzero(category_mask & (folds == fold))
        predicted, hits = predict(frame, base, donors, targets, category, config)
        predictions[targets] = predicted
        if config is not None:
            extra_hits += hits.get(config[0], 0)
    positions = np.flatnonzero(category_mask)
    return f1(labels[positions], predictions[positions]), extra_hits, predictions


def repeated_random_split(
    frame: pd.DataFrame,
    base: np.ndarray,
    category: str,
    config: tuple[str, int, float],
) -> dict[str, object]:
    labels = frame.label.to_numpy(dtype=np.int8)
    category_positions = np.flatnonzero(frame.category.to_numpy() == category)
    rng = np.random.default_rng(20260821)
    rows = []
    for repeat in range(20):
        train_parts, valid_parts = [], []
        for label in [0, 1]:
            positions = category_positions[labels[category_positions] == label].copy()
            rng.shuffle(positions)
            cut = int(round(0.70 * len(positions)))
            train_parts.append(positions[:cut])
            valid_parts.append(positions[cut:])
        donors = np.concatenate(train_parts)
        targets = np.concatenate(valid_parts)
        baseline, _ = predict(frame, base, donors, targets, category, None)
        candidate, hits = predict(frame, base, donors, targets, category, config)
        rows.append({
            "repeat": repeat,
            "rows": int(len(targets)),
            "baseline_f1": f1(labels[targets], baseline),
            "candidate_f1": f1(labels[targets], candidate),
            "extra_hits": hits.get(config[0], 0),
        })
    deltas = np.asarray([row["candidate_f1"] - row["baseline_f1"] for row in rows])
    return {
        "baseline_mean_f1": float(np.mean([row["baseline_f1"] for row in rows])),
        "candidate_mean_f1": float(np.mean([row["candidate_f1"] for row in rows])),
        "mean_delta": float(deltas.mean()),
        "delta_std": float(deltas.std()),
        "positive_repeats": int((deltas > 0).sum()),
        "negative_repeats": int((deltas < 0).sum()),
        "mean_extra_hits": float(np.mean([row["extra_hits"] for row in rows])),
        "repeats": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", default=str(FUSION))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--prior", default=str(PRIOR))
    args = parser.parse_args()

    frame = pd.read_csv(DATA)
    frame["name"] = frame.name.fillna("").astype(str)
    frame["description"] = frame.description.fillna("").astype(str)
    frame["normalized_name"] = frame.name.map(normalize)
    texts = [compose_text(name, description) for name, description in zip(frame.name, frame.description)]
    frame["text_hash"] = [fingerprint(text) for text in texts]
    frame["canonical_text"] = [canonical(text) for text in texts]
    frame["canonical_text_mask_digits"] = [canonical(text, mask_digits=True) for text in texts]
    frame["canonical_name"] = frame.name.map(canonical)
    frame["canonical_name_mask_digits"] = frame.name.map(
        lambda value: canonical(value, mask_digits=True)
    )
    frame["token_signature"] = [token_signature(text) for text in texts]

    fusion = np.load(args.fusion, allow_pickle=True)
    if not np.array_equal(frame.id.astype(str).to_numpy(), fusion["ids"].astype(str)):
        raise ValueError("id mismatch")
    frame["fold"] = fusion["folds"].astype(np.int8)
    base = fusion["full_oof_predictions"].astype(np.int8)
    labels = frame.label.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy(dtype=np.int8)

    report: dict[str, object] = {
        "protocol": "honest outer-fold donor mappings with inner-fold config selection; 20 repeated stratified 70/30 checks",
        "categories": {},
    }
    final_prior = json.loads(BASE_PRIOR.read_text(encoding="utf-8"))
    final_prior["fuzzy"] = {"configs": {}, "mappings": {}}
    macro_baseline, macro_nested, macro_fixed = [], [], []
    for category in sorted(frame.category.unique()):
        category_mask = frame.category.to_numpy() == category
        nested_predictions = base.copy()
        choices: list[tuple[str, int, float]] = []
        fold_rows = []
        baseline_f1, _, baseline_predictions = evaluate_fixed_oof(
            frame, base, category, None
        )
        for outer_fold in sorted(frame.fold.unique()):
            config, inner_f1, inner_hits = choose_extra_nested(
                frame, base, category, int(outer_fold)
            )
            donors = np.flatnonzero(category_mask & (folds != outer_fold))
            targets = np.flatnonzero(category_mask & (folds == outer_fold))
            predicted, hits = predict(frame, base, donors, targets, category, config)
            nested_predictions[targets] = predicted
            choices.append(config)
            fold_rows.append({
                "fold": int(outer_fold),
                "selected": list(config),
                "inner_f1": inner_f1,
                "inner_extra_hits": inner_hits,
                "validation_f1": f1(labels[targets], predicted),
                "validation_extra_hits": hits.get(config[0], 0),
            })
        category_positions = np.flatnonzero(category_mask)
        nested_f1 = f1(labels[category_positions], nested_predictions[category_positions])
        fixed_grid_rows = []
        for grid_config in candidate_grid(category):
            grid_f1, grid_hits, grid_predictions = evaluate_fixed_oof(
                frame, base, category, grid_config
            )
            changed = int(
                (grid_predictions[category_positions] != baseline_predictions[category_positions]).sum()
            )
            fixed_grid_rows.append({
                "config": list(grid_config),
                "f1": grid_f1,
                "delta": grid_f1 - baseline_f1,
                "extra_hits": grid_hits,
                "changed_predictions": changed,
            })
        fixed_grid_rows.sort(
            key=lambda row: (row["f1"], -row["changed_predictions"], row["config"][1]),
            reverse=True,
        )
        choice_counts = Counter(choices)
        selected = max(
            choice_counts,
            key=lambda row: (choice_counts[row], row[1], row[2], row[0]),
        )
        fixed_f1, fixed_hits, _ = evaluate_fixed_oof(frame, base, category, selected)
        random_split = repeated_random_split(frame, base, category, selected)

        all_positions = np.flatnonzero(category_mask)
        mapping = build_mapping(frame, all_positions, selected[0], selected[1], selected[2])
        active = (
            fixed_f1 > baseline_f1
            and random_split["mean_delta"] > 0
            and random_split["positive_repeats"] >= 15
        )
        final_prior["fuzzy"]["configs"][category] = list(selected) if active else None
        final_prior["fuzzy"]["mappings"][category] = (
            {hashlib.sha1(key.encode("utf-8")).hexdigest(): value for key, value in mapping.items()}
            if active
            else {}
        )
        report["categories"][category] = {
            "baseline_existing_prior_honest_oof_f1": baseline_f1,
            "nested_selected_fuzzy_f1": nested_f1,
            "nested_delta": nested_f1 - baseline_f1,
            "selected_full_config": list(selected),
            "choice_counts": {str(list(key)): value for key, value in choice_counts.items()},
            "fixed_config_honest_oof_f1": fixed_f1,
            "fixed_config_delta": fixed_f1 - baseline_f1,
            "fixed_config_extra_hits": fixed_hits,
            "fixed_grid_top": fixed_grid_rows[:15],
            "full_mapping_entries": len(mapping),
            "activated_for_submission": active,
            "folds": fold_rows,
            "random_split": random_split,
        }
        macro_baseline.append(baseline_f1)
        macro_nested.append(nested_f1)
        macro_fixed.append(fixed_f1)

    report["macro"] = {
        "baseline_existing_prior_honest_oof_f1": float(np.mean(macro_baseline)),
        "nested_selected_fuzzy_f1": float(np.mean(macro_nested)),
        "nested_delta": float(np.mean(macro_nested) - np.mean(macro_baseline)),
        "fixed_config_honest_oof_f1": float(np.mean(macro_fixed)),
        "fixed_config_delta": float(np.mean(macro_fixed) - np.mean(macro_baseline)),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.prior).write_text(json.dumps(final_prior, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print({"prior_bytes": Path(args.prior).stat().st_size})


if __name__ == "__main__":
    main()

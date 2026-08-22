from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("research")
DATA = Path(os.environ.get("ECUP_DATA", ROOT / "data.csv"))
LORA = ROOT / "lora-hard-5fold-robust-fusion-report.npz"
REPORT = ROOT / "annotator_prior_report_v2.json"
PRIOR = ROOT / "annotator_prior_v2.json"


def normalize(value):
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def compose_text(name, description):
    name = normalize(name)
    return f"{name}\n{name}\n{normalize(description)}"


def fingerprint(value):
    return hashlib.sha1(normalize(value).encode("utf-8")).hexdigest()


def f1(labels, predictions):
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def group_leave_one_out(frame, key):
    counts = frame.groupby(["category", key]).label.transform("count").to_numpy(dtype=np.int32)
    sums = frame.groupby(["category", key]).label.transform("sum").to_numpy(dtype=np.float32)
    other_count = counts - 1
    other_mean = np.divide(
        sums - frame.label.to_numpy(dtype=np.float32), other_count,
        out=np.full(len(frame), np.nan, dtype=np.float32), where=other_count > 0,
    )
    confidence = np.maximum(other_mean, 1 - other_mean)
    prediction = (other_mean >= 0.5).astype(np.int8)
    return other_count, confidence, prediction


def apply_config(base, arrays, config, positions=None):
    exact_min, exact_conf, name_min, name_conf = config
    if positions is None:
        positions = np.arange(len(base))
    result = base[positions].copy()
    exact_count, exact_confidence, exact_prediction, name_count, name_confidence, name_prediction = arrays
    exact = (exact_count[positions] >= exact_min) & (exact_confidence[positions] >= exact_conf)
    name = (~exact) & (name_count[positions] >= name_min) & (name_confidence[positions] >= name_conf)
    result[exact] = exact_prediction[positions][exact]
    result[name] = name_prediction[positions][name]
    return result, int(exact.sum()), int(name.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", default=str(LORA))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--prior", default=str(PRIOR))
    args = parser.parse_args()
    frame = pd.read_csv(DATA)
    frame["name"] = frame.name.fillna("").astype(str)
    frame["description"] = frame.description.fillna("").astype(str)
    frame["normalized_name"] = frame.name.map(normalize)
    frame["text_hash"] = [fingerprint(compose_text(a, b)) for a, b in zip(frame.name, frame.description)]
    lora = np.load(args.fusion, allow_pickle=True)
    if not np.array_equal(frame.id.astype(str).to_numpy(), lora["ids"].astype(str)):
        raise ValueError("id mismatch")
    frame["fold"] = lora["folds"]
    if "full_oof_predictions" in lora.files:
        base = lora["full_oof_predictions"].astype(np.int8)
    else:
        base = np.zeros(len(frame), dtype=np.int8)
        configs_by_category = {
            "БАД": (0.40, 0.60, 0.25814030990600584),
            "Легковоспламеняющиеся": (0.75, 0.25, 0.9654546632766724),
        }
        for category, (weight_base, weight_lora, threshold) in configs_by_category.items():
            mask = frame.category.to_numpy() == category
            score = weight_base * lora["base_rank"][mask] + weight_lora * lora["lora_rank"][mask]
            base[mask] = score >= threshold
    exact = group_leave_one_out(frame, "text_hash")
    name = group_leave_one_out(frame, "normalized_name")
    arrays = (*exact, *name)
    grid = list(product(
        [1, 2, 3, 4, 999], [0.55, 0.60, 2 / 3, 0.75, 0.80, 0.90, 0.999],
        [2, 3, 4, 5, 8, 999], [0.55, 0.60, 2 / 3, 0.75, 0.80, 0.90, 0.999],
    ))
    labels = frame.label.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy(dtype=np.int8)
    report = {"source": "organizer says test uses the same ambiguous annotation process", "categories": {}}
    selected_for_mapping = {}
    macro = []
    for category in sorted(frame.category.unique()):
        category_mask = frame.category.to_numpy() == category
        nested_predictions = base.copy()
        fold_rows = []
        choices = []
        for outer_fold in sorted(frame.fold.unique()):
            train = np.flatnonzero(category_mask & (folds != outer_fold))
            valid = np.flatnonzero(category_mask & (folds == outer_fold))
            best = None
            for config in grid:
                predictions, exact_changed, name_changed = apply_config(base, arrays, config, train)
                value = f1(labels[train], predictions)
                changed = exact_changed + name_changed
                preference = (value, -changed, config[0], config[2], config[1], config[3])
                if best is None or preference > best[0]:
                    best = (preference, config)
            config = best[1]
            predictions, exact_changed, name_changed = apply_config(base, arrays, config, valid)
            nested_predictions[valid] = predictions
            choices.append(config)
            fold_rows.append({
                "fold": int(outer_fold), "config": list(config),
                "validation_f1": f1(labels[valid], predictions),
                "exact_overrides": exact_changed, "name_overrides": name_changed,
            })
        category_f1 = f1(labels[category_mask], nested_predictions[category_mask])
        baseline_f1 = f1(labels[category_mask], base[category_mask])
        current_positions = np.flatnonzero(category_mask)
        current_predictions, current_exact, current_name = apply_config(
            base, arrays, (2, 0.999, 3, 0.999), current_positions
        )
        current_lookup_f1 = f1(labels[category_mask], current_predictions)
        choice_counts = Counter(choices)
        chosen = max(choice_counts, key=lambda config: (choice_counts[config], config[0], config[2], config[1], config[3]))
        selected_for_mapping[category] = chosen
        full_predictions, exact_changed, name_changed = apply_config(
            base, arrays, chosen, np.flatnonzero(category_mask)
        )
        report["categories"][category] = {
            "baseline_f1": baseline_f1,
            "current_consistent_lookup_f1": current_lookup_f1,
            "current_consistent_exact_overrides": current_exact,
            "current_consistent_name_overrides": current_name,
            "nested_prior_f1": category_f1,
            "increment_over_current_lookup": category_f1 - current_lookup_f1,
            "chosen_config": list(chosen),
            "chosen_config_loo_f1": f1(labels[category_mask], full_predictions),
            "exact_overrides": exact_changed,
            "name_overrides": name_changed,
            "folds": fold_rows,
        }
        macro.append(category_f1)
    report["nested_macro_f1"] = float(np.mean(macro))

    mappings = {"exact": {}, "name": {}, "configs": {}}
    for category, config in selected_for_mapping.items():
        exact_min, exact_conf, name_min, name_conf = config
        mappings["configs"][category] = list(config)
        local = frame[frame.category == category]
        for key, group in local.groupby("text_hash"):
            count, mean = len(group), float(group.label.mean())
            confidence = max(mean, 1 - mean)
            if count >= exact_min and confidence >= exact_conf and mean != 0.5:
                mappings["exact"][f"{category}\t{key}"] = int(mean >= 0.5)
        for key, group in local.groupby("normalized_name"):
            count, mean = len(group), float(group.label.mean())
            confidence = max(mean, 1 - mean)
            if count >= name_min and confidence >= name_conf and mean != 0.5:
                mappings["name"][f"{category}\t{key}"] = int(mean >= 0.5)
    report_path = Path(args.report)
    prior_path = Path(args.prior)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    prior_path.write_text(json.dumps(mappings, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print({"exact_mappings": len(mappings["exact"]), "name_mappings": len(mappings["name"]), "bytes": prior_path.stat().st_size})


if __name__ == "__main__":
    main()

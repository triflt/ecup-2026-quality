from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("research")
DATA = Path(os.environ.get("ECUP_DATA", ROOT / "data.csv"))
LORA = ROOT / "lora-hard-5fold-robust-fusion-report.npz"
OUTPUT = ROOT / "annotator_prior_random_split_report.json"

PRIOR_CONFIGS = {
    "БАД": (2, 2 / 3, 2, 0.999),
    "Легковоспламеняющиеся": (1, 0.999, 999, 0.999),
}
CURRENT_CONFIG = (2, 0.999, 3, 0.999)


def normalize(value):
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def fingerprint(name, description):
    name = normalize(name)
    value = normalize(f"{name}\n{name}\n{normalize(description)}")
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def f1(labels, predictions):
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def build_mapping(train, key, minimum, confidence_min):
    result = {}
    for value, group in train.groupby(key):
        count = len(group)
        mean = float(group.label.mean())
        confidence = max(mean, 1 - mean)
        if count >= minimum and confidence >= confidence_min and mean != 0.5:
            result[value] = int(mean >= 0.5)
    return result


def apply(frame, train_positions, valid_positions, base, config):
    exact_min, exact_conf, name_min, name_conf = config
    train = frame.iloc[train_positions]
    valid = frame.iloc[valid_positions]
    exact = build_mapping(train, "text_hash", exact_min, exact_conf)
    name = build_mapping(train, "normalized_name", name_min, name_conf)
    predictions = base[valid_positions].copy()
    exact_hits = valid.text_hash.isin(exact).to_numpy()
    name_hits = (~exact_hits) & valid.normalized_name.isin(name).to_numpy()
    for local_index, value in enumerate(valid.text_hash):
        if value in exact:
            predictions[local_index] = exact[value]
        elif valid.normalized_name.iloc[local_index] in name:
            predictions[local_index] = name[valid.normalized_name.iloc[local_index]]
    return predictions, int(exact_hits.sum()), int(name_hits.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", default=str(LORA))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    frame = pd.read_csv(DATA)
    frame["name"] = frame.name.fillna("").astype(str)
    frame["description"] = frame.description.fillna("").astype(str)
    frame["normalized_name"] = frame.name.map(normalize)
    frame["text_hash"] = [fingerprint(a, b) for a, b in zip(frame.name, frame.description)]
    lora = np.load(args.fusion, allow_pickle=True)
    if "full_oof_predictions" in lora.files:
        base = lora["full_oof_predictions"].astype(np.int8)
    else:
        base = np.zeros(len(frame), dtype=np.int8)
        score_configs = {
            "БАД": (0.40, 0.60, 0.25814030990600584),
            "Легковоспламеняющиеся": (0.75, 0.25, 0.9654546632766724),
        }
        for category, (weight_base, weight_lora, threshold) in score_configs.items():
            mask = frame.category.to_numpy() == category
            scores = weight_base * lora["base_rank"][mask] + weight_lora * lora["lora_rank"][mask]
            base[mask] = scores >= threshold
    rng = np.random.default_rng(20260821)
    report = {"protocol": "20 repeated stratified row splits, 70% prior-train / 30% simulated hidden", "categories": {}}
    for category in sorted(frame.category.unique()):
        category_positions = np.flatnonzero(frame.category.to_numpy() == category)
        rows = []
        for repeat in range(20):
            train_parts, valid_parts = [], []
            for label in [0, 1]:
                positions = category_positions[frame.label.to_numpy()[category_positions] == label].copy()
                rng.shuffle(positions)
                cut = int(round(0.70 * len(positions)))
                train_parts.append(positions[:cut])
                valid_parts.append(positions[cut:])
            train = np.concatenate(train_parts)
            valid = np.concatenate(valid_parts)
            current, current_exact, current_name = apply(frame, train, valid, base, CURRENT_CONFIG)
            prior, prior_exact, prior_name = apply(frame, train, valid, base, PRIOR_CONFIGS[category])
            labels = frame.label.to_numpy(dtype=np.int8)[valid]
            rows.append({
                "repeat": repeat,
                "rows": len(valid),
                "base_f1": f1(labels, base[valid]),
                "current_lookup_f1": f1(labels, current),
                "prior_f1": f1(labels, prior),
                "current_exact_hits": current_exact,
                "current_name_hits": current_name,
                "prior_exact_hits": prior_exact,
                "prior_name_hits": prior_name,
            })
        summary = {}
        for key in ["base_f1", "current_lookup_f1", "prior_f1", "current_exact_hits", "current_name_hits", "prior_exact_hits", "prior_name_hits"]:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            summary[key] = {"mean": float(values.mean()), "std": float(values.std())}
        summary["prior_minus_current_f1"] = float(
            np.mean([row["prior_f1"] - row["current_lookup_f1"] for row in rows])
        )
        summary["rows_per_split"] = rows[0]["rows"]
        report["categories"][category] = {"summary": summary, "repeats": rows}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({category: value["summary"] for category, value in report["categories"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

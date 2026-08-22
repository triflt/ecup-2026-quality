from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from soft_annotator_prior_cv import (
    DATA,
    MODEL_CONFIG,
    QWEN35,
    QWEN3VL,
    apply_soft,
    build_split_cache,
    calibrate_at_threshold,
    canonical,
    compose_text,
    f1,
    fingerprint,
    normalize,
    policies,
)


REPORT = Path("research/qwen3vl-qwen35-soft-annotator-prior-random-split-report.json")
HARD_CONFIG = {
    "БАД": (2, 2 / 3, 2, 0.999),
    "Легковоспламеняющиеся": (1, 0.999, 999, 0.999),
}


def hard_mapping(train: pd.DataFrame, key: str, minimum: int, confidence_minimum: float) -> dict[str, int]:
    stats = train.groupby(key, sort=False).label.agg(["count", "mean"])
    confidence = np.maximum(stats["mean"], 1 - stats["mean"])
    keep = (
        (stats["count"] >= minimum)
        & (confidence >= confidence_minimum)
        & (stats["mean"] != 0.5)
    )
    return {str(key): int(value >= 0.5) for key, value in stats.loc[keep, "mean"].items()}


def apply_hard(
    frame: pd.DataFrame,
    donors: np.ndarray,
    targets: np.ndarray,
    base: np.ndarray,
    config: tuple[int, float, int, float],
) -> np.ndarray:
    exact_min, exact_conf, name_min, name_conf = config
    train = frame.iloc[donors]
    exact = hard_mapping(train, "text_hash", exact_min, exact_conf)
    name = hard_mapping(train, "normalized_name", name_min, name_conf)
    result = base[targets].copy()
    target = frame.iloc[targets]
    for index, (exact_key, name_key) in enumerate(zip(target.text_hash, target.normalized_name)):
        if exact_key in exact:
            result[index] = exact[exact_key]
        elif name_key in name:
            result[index] = name[name_key]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()

    frame = pd.read_csv(DATA)
    frame["name"] = frame.name.fillna("").astype(str)
    frame["description"] = frame.description.fillna("").astype(str)
    frame["normalized_name"] = frame.name.map(normalize)
    texts = [compose_text(a, b) for a, b in zip(frame.name, frame.description)]
    frame["text_hash"] = [fingerprint(text) for text in texts]
    frame["canonical_text_mask_digits"] = [canonical(text, mask_digits=True) for text in texts]
    labels = frame.label.to_numpy(dtype=np.int8)
    categories = frame.category.astype(str).to_numpy()

    qwen35 = np.load(QWEN35, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    ids = frame.id.astype(str).to_numpy()
    if not np.array_equal(ids, qwen35["ids"].astype(str)) or not np.array_equal(ids, qwen3vl["ids"].astype(str)):
        raise ValueError("id mismatch")

    alphas = [0.0, 1.0, 3.0, 10.0]
    weights = [0.25, 0.50, 0.75, 1.0]
    thresholds = [0.45, 0.475, 0.50, 0.525, 0.55]
    rng = np.random.default_rng(20260821)
    report: dict[str, object] = {
        "protocol": "20 repeated stratified row splits (70% train/30% hidden); soft config selected on repeats 0-9 and confirmed without retuning on repeats 10-19",
        "categories": {},
    }

    for category in sorted(frame.category.unique()):
        category_positions = np.flatnonzero(categories == category)
        (base_weight, vl_weight, q35_weight), model_threshold = MODEL_CONFIG[category]
        model_score = (
            base_weight * qwen35["base_rank"].astype(np.float32)
            + vl_weight * qwen3vl["lora_rank"].astype(np.float32)
            + q35_weight * qwen35["lora_rank"].astype(np.float32)
        )
        model_probability = calibrate_at_threshold(model_score, model_threshold)
        base_predictions = (model_score >= model_threshold).astype(np.int8)

        splits = []
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
            splits.append({
                "repeat": repeat,
                "donors": donors,
                "targets": targets,
                "cache": build_split_cache(frame, donors, targets, alphas),
                "base_f1": f1(labels[targets], base_predictions[targets]),
                "hard_f1": f1(
                    labels[targets],
                    apply_hard(frame, donors, targets, base_predictions, HARD_CONFIG[category]),
                ),
            })

        best = None
        candidate_policies = policies(category)
        for alpha, weight, threshold, policy in product(alphas, weights, thresholds, candidate_policies):
            values, covered = [], 0
            config = (weight, alpha, threshold, policy)
            for split in splits[:10]:
                targets = split["targets"]
                predictions, hits, _ = apply_soft(
                    model_probability[targets], split["cache"][alpha], config
                )
                values.append(f1(labels[targets], predictions))
                covered += int(hits.sum())
            mean = float(np.mean(values))
            preference = (mean, -weight, -covered, -abs(threshold - 0.5), -alpha)
            if best is None or preference > best[0]:
                best = (preference, config, values)
        assert best is not None
        config = best[1]
        rows = []
        for split in splits:
            targets = split["targets"]
            predictions, covered, hits = apply_soft(
                model_probability[targets], split["cache"][config[1]], config
            )
            rows.append({
                "repeat": split["repeat"],
                "partition": "selection" if split["repeat"] < 10 else "confirmation",
                "rows": int(len(targets)),
                "base_f1": split["base_f1"],
                "hard_f1": split["hard_f1"],
                "soft_f1": f1(labels[targets], predictions),
                "covered": int(covered.sum()),
                "hits": hits,
            })

        def summarize(selected: list[dict[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {"repeats": len(selected)}
            for key in ["base_f1", "hard_f1", "soft_f1", "covered"]:
                values = np.asarray([row[key] for row in selected], dtype=np.float64)
                result[key] = {"mean": float(values.mean()), "std": float(values.std())}
            result["soft_minus_hard_mean"] = float(np.mean([
                row["soft_f1"] - row["hard_f1"] for row in selected
            ]))
            result["soft_wins_vs_hard"] = int(sum(row["soft_f1"] > row["hard_f1"] for row in selected))
            return result

        report["categories"][category] = {
            "selected_config": [config[0], config[1], config[2], list(config[3])],
            "selection": summarize(rows[:10]),
            "confirmation": summarize(rows[10:]),
            "all": summarize(rows),
            "rows": rows,
        }
        print(
            f"category={category} config={config} confirmation={report['categories'][category]['confirmation']}",
            flush=True,
        )

    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

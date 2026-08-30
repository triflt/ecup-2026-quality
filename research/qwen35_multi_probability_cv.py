from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from qwen35_seed_ensemble_cv import (
    BASE,
    QWEN3VL,
    QWEN35_SEED_A,
    evaluate_matrix,
    fold_category_ranks,
    load_seed_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-set",
        action="append",
        nargs="+",
        required=True,
        metavar="NAME_OR_CSV",
        help="Repeat as: --prediction-set NAME fold0.csv ... fold4.csv",
    )
    parser.add_argument(
        "--primary-combination",
        required=True,
        help="Comma-separated prediction-set names fixed before evaluation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = np.load(BASE, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    reference = np.load(QWEN35_SEED_A, allow_pickle=True)
    ids = base["ids"].astype(str)
    folds = base["fold_ids"].astype(np.int8)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    probabilities = {}
    for specification in args.prediction_set:
        if len(specification) < 2:
            raise ValueError("each --prediction-set needs a name and prediction CSVs")
        name, paths = specification[0], specification[1:]
        if name in probabilities:
            raise ValueError(f"duplicate prediction-set name: {name}")
        logits = load_seed_predictions(paths, base)
        probabilities[name] = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))

    primary_names = tuple(value.strip() for value in args.primary_combination.split(","))
    if not primary_names or not set(primary_names).issubset(probabilities):
        raise ValueError(f"invalid primary combination: {primary_names}")
    base_matrix = np.column_stack(
        [
            reference["base_rank"].astype(np.float32),
            qwen3vl["lora_rank"].astype(np.float32),
        ]
    )
    reports = {}
    primary_predictions = None
    primary_full_predictions = None
    for count in range(1, len(probabilities) + 1):
        for names in itertools.combinations(probabilities, count):
            mean_probability = np.mean(
                np.column_stack([probabilities[name] for name in names]), axis=1
            ).astype(np.float32)
            ranks = fold_category_ranks(mean_probability, folds, categories)
            report, nested, full = evaluate_matrix(
                ["robust_base", "qwen3vl", "qwen35_mean_probability"],
                np.column_stack([base_matrix, ranks]),
                labels,
                categories,
                folds,
            )
            key = "+".join(names)
            reports[key] = report
            if set(names) == set(primary_names) and len(names) == len(primary_names):
                primary_predictions = nested
                primary_full_predictions = full
    if primary_predictions is None:
        raise ValueError("primary combination was not evaluated")

    primary_key = "+".join(name for name in probabilities if name in primary_names)
    result = {
        "primary_combination": primary_key,
        "primary_was_fixed_before_evaluation": True,
        "diagnostic_combinations_are_not_selection_evidence": True,
        "combinations": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        ids=ids,
        folds=folds,
        labels=labels,
        categories=categories,
        nested_predictions=primary_predictions,
        full_oof_predictions=primary_full_predictions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

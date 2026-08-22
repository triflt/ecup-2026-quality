from __future__ import annotations

import argparse
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
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = np.load(BASE, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    reference = np.load(QWEN35_SEED_A, allow_pickle=True)
    ids = base["ids"].astype(str)
    folds = base["fold_ids"].astype(np.int8)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    for name, source in (("qwen3vl", qwen3vl), ("reference", reference)):
        if not np.array_equal(ids, source["ids"].astype(str)):
            raise ValueError(f"{name} id mismatch")
    logits = load_seed_predictions(args.predictions, base)
    variant_rank = fold_category_ranks(logits, folds, categories)
    report, nested, full = evaluate_matrix(
        ["robust_base", "qwen3vl", args.variant_name],
        np.column_stack(
            [
                reference["base_rank"].astype(np.float32),
                qwen3vl["lora_rank"].astype(np.float32),
                variant_rank,
            ]
        ),
        labels,
        categories,
        folds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        ids=ids,
        folds=folds,
        labels=labels,
        categories=categories,
        variant_rank=variant_rank,
        nested_predictions=nested,
        full_oof_predictions=full,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecup_quality.data.text import compose_text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("validation/grouped_text_v1/folds.csv"))
    parser.add_argument("--basket-output", type=Path, default=Path("validation/grouped_text_v1/basket.csv"))
    parser.add_argument("--manifest-output", type=Path, default=Path("validation/grouped_text_v1/manifest.json"))
    parser.add_argument("--basket-fold", type=int, default=4)
    parser.add_argument("--dataset-version", default="competition_train_v1")
    parser.add_argument("--evaluation-version", default="grouped_text_v1")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing immutable version.")
    parser.add_argument(
        "--reference-oof",
        type=Path,
        help="Optional local NPZ with ids/folds used to reproduce the historical split exactly.",
    )
    args = parser.parse_args()

    targets = (args.output, args.basket_output, args.manifest_output)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite versioned validation files; choose a new version/path "
            f"or pass --force explicitly: {existing}"
        )

    frame = pd.read_csv(args.data)
    if args.reference_oof is None:
        from ecup_quality.validation.splits import assign_grouped_folds

        folds = assign_grouped_folds(frame)
        fold_source = "recomputed"
    else:
        reference = np.load(args.reference_oof, allow_pickle=True)
        if not np.array_equal(frame["id"].astype(str).to_numpy(), reference["ids"].astype(str)):
            raise ValueError("reference OOF id order does not match data")
        folds = reference["folds"].astype(np.int8)
        fold_source = "frozen reference OOF"
    output = frame[["id", "category", "label"]].copy()
    output["fold"] = folds
    output["group_hash"] = [
        hashlib.sha1(compose_text(name, description).encode("utf-8")).hexdigest()
        for name, description in zip(frame["name"], frame["description"])
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    basket = output[output.fold == args.basket_fold].copy()
    basket.to_csv(args.basket_output, index=False)
    manifest = {
        "evaluation_version": args.evaluation_version,
        "dataset_version": args.dataset_version,
        "protocol": "category-specific StratifiedGroupKFold by normalized full text",
        "n_splits": 5,
        "random_state": 42,
        "fold_source": fold_source,
        "data_rows": len(output),
        "basket_fold": args.basket_fold,
        "basket_rows": len(basket),
        "fold_rows": output.fold.value_counts().sort_index().astype(int).to_dict(),
        "fold_positives": output.groupby("fold").label.sum().astype(int).to_dict(),
        "source_data_sha256": sha256(args.data),
        "folds_sha256": sha256(args.output),
        "basket_sha256": sha256(args.basket_output),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

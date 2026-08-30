from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ID = "715"
EXPECTED_DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"
SEED = 42
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_image(images: Path, row_id: str) -> str:
    folder = images / row_id
    preferred = folder / "0.jpg"
    if preferred.is_file():
        return str(preferred.resolve())
    paths = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no image for id={row_id}")
    return str(paths[0].resolve())


def sampled_indices(frame: pd.DataFrame) -> tuple[list[int], dict]:
    labels = frame["label"].to_numpy(np.int8)
    categories = frame["category"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED)
    selected: list[int] = []

    bad = np.flatnonzero(categories == "БАД")
    bad_pos = bad[labels[bad] == 1]
    bad_neg = bad[labels[bad] == 0]
    bad_count = min(1900, len(bad_pos), len(bad_neg))
    selected.extend(rng.choice(bad_pos, bad_count, replace=False).tolist())
    selected.extend(rng.choice(bad_neg, bad_count, replace=False).tolist())

    flammable = np.flatnonzero(categories == "Легковоспламеняющиеся")
    flammable_pos = flammable[labels[flammable] == 1]
    flammable_neg = flammable[labels[flammable] == 0]
    selected.extend(np.repeat(flammable_pos, 5).tolist())
    flammable_neg_count = min(2000, len(flammable_neg))
    selected.extend(rng.choice(flammable_neg, flammable_neg_count, replace=False).tolist())
    random.Random(SEED).shuffle(selected)
    return selected, {
        "bad_negative": bad_count,
        "bad_positive": bad_count,
        "flammable_negative": flammable_neg_count,
        "flammable_positive_unique": len(flammable_pos),
        "flammable_positive_occurrences": 5 * len(flammable_pos),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("canonical data checksum mismatch")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty runtime")
    data = pd.read_csv(args.data, dtype={"id": str})
    folds = pd.read_csv(args.folds, dtype={"id": str})
    if data["id"].duplicated().any() or folds["id"].duplicated().any():
        raise ValueError("duplicate source IDs")
    fold_map = dict(zip(folds["id"], folds["fold"].astype(int), strict=True))
    if set(data["id"]) != set(fold_map):
        raise ValueError("data/folds ID mismatch")
    data["fold"] = [fold_map[row_id] for row_id in data["id"]]
    data["name"] = data["name"].fillna("").astype(str)
    data["description"] = data["description"].fillna("").astype(str)
    image_by_id = {row_id: first_image(args.images, row_id) for row_id in data["id"]}
    selected, strata = sampled_indices(data)
    rows = []
    for occurrence_index, index in enumerate(selected):
        row = data.iloc[index]
        rows.append(
            {
                "occurrence_index": occurrence_index,
                "id": str(row.id),
                "fold": int(row.fold),
                "category": str(row.category),
                "name": str(row["name"])[:320],
                "description": str(row.description)[:1800],
                "image_path": image_by_id[str(row.id)],
                "label": int(row.label),
            }
        )
    if len({row["id"] for row in rows}) == len(rows):
        raise ValueError("full runtime unexpectedly lacks oversampled occurrences")
    args.output_dir.mkdir(parents=True)
    train_path = args.output_dir / "train.jsonl"
    write_jsonl(train_path, rows)
    audit = {
        "schema_version": "exp715_full_runtime_v1",
        "experiment_id": EXPERIMENT_ID,
        "source_sampler": "exp697_label_only_v1_full_scale",
        "seed": SEED,
        "full_data": True,
        "data_sha256": sha256(args.data),
        "folds_sha256": sha256(args.folds),
        "train_occurrences": len(rows),
        "train_unique_ids": len({row["id"] for row in rows}),
        "strata": strata,
        "train_sha256": sha256(train_path),
        "decision": "FULL_RUNTIME_FROZEN",
    }
    (args.output_dir / "runtime_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

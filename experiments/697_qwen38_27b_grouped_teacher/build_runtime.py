from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/data.csv")
IMAGES = Path("/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/images/images")
FOLDS = Path(__file__).resolve().parents[2] / "validation/grouped_text_v1/folds.csv"
EXPECTED_DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"
SEED = 42
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_image(row_id: str) -> str:
    folder = IMAGES / row_id
    preferred = folder / "0.jpg"
    if preferred.is_file():
        return str(preferred.resolve())
    paths = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no image for id={row_id}")
    return str(paths[0].resolve())


def sampled_indices(frame: pd.DataFrame, outer_fold: int) -> list[int]:
    labels = frame["label"].to_numpy(np.int8)
    categories = frame["category"].astype(str).to_numpy()
    folds = frame["fold"].to_numpy(np.int8)
    train = folds != outer_fold
    rng = np.random.default_rng(SEED)
    selected: list[int] = []

    bad = np.flatnonzero(train & (categories == "БАД"))
    bad_pos = bad[labels[bad] == 1]
    bad_neg = bad[labels[bad] == 0]
    count = min(1500, len(bad_pos), len(bad_neg))
    selected.extend(rng.choice(bad_pos, count, replace=False).tolist())
    selected.extend(rng.choice(bad_neg, count, replace=False).tolist())

    flammable = np.flatnonzero(train & (categories == "Легковоспламеняющиеся"))
    flammable_pos = flammable[labels[flammable] == 1]
    flammable_neg = flammable[labels[flammable] == 0]
    selected.extend(np.repeat(flammable_pos, 5).tolist())
    selected.extend(
        rng.choice(flammable_neg, min(1600, len(flammable_neg)), replace=False).tolist()
    )
    random.Random(SEED).shuffle(selected)
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha256(DATA) != EXPECTED_DATA_SHA256:
        raise ValueError("canonical data checksum mismatch")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty runtime")

    data = pd.read_csv(DATA, dtype={"id": str})
    folds = pd.read_csv(FOLDS, dtype={"id": str})
    fold_map = dict(zip(folds["id"], folds["fold"].astype(int), strict=True))
    if set(data["id"]) != set(fold_map):
        raise ValueError("data/folds ID mismatch")
    data["fold"] = [fold_map[row_id] for row_id in data["id"]]
    data["name"] = data["name"].fillna("").astype(str)
    data["description"] = data["description"].fillna("").astype(str)
    images = {row_id: first_image(row_id) for row_id in data["id"]}

    def record(index: int, with_label: bool) -> dict:
        row = data.iloc[index]
        item = {
            "id": str(row.id),
            "fold": int(row.fold),
            "category": str(row.category),
            "name": str(row["name"])[:320],
            "description": str(row.description)[:1800],
            "image_path": images[str(row.id)],
        }
        if with_label:
            item["label"] = int(row.label)
        return item

    selected = sampled_indices(data, args.fold)
    train_rows = [record(index, True) for index in selected]
    valid_indices = np.flatnonzero(data["fold"].to_numpy() == args.fold)
    validation_rows = [record(int(index), False) for index in valid_indices]
    if any(row["fold"] == args.fold for row in train_rows):
        raise ValueError("outer validation leakage")

    args.output_dir.mkdir(parents=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    audit = {
        "experiment_id": "697",
        "fold": args.fold,
        "data_sha256": sha256(DATA),
        "folds_sha256": sha256(FOLDS),
        "sampler": "label_only_v1",
        "seed": SEED,
        "train_occurrences": len(train_rows),
        "train_unique_ids": len({row["id"] for row in train_rows}),
        "validation_rows": len(validation_rows),
        "validation_labels_written": 0,
        "train_sha256": sha256(train_path),
        "validation_sha256": sha256(validation_path),
    }
    (args.output_dir / "runtime_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

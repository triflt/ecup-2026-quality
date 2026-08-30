from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: object, limit: int) -> str:
    text = "" if pd.isna(value) else html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head].rstrip() + " … " + text[-(limit - head) :].lstrip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite runtime files")

    frame = pd.read_csv(args.data, usecols=["id", "category", "name", "description"])
    folds = pd.read_csv(args.folds, usecols=["id", "split", "development_fold"])
    development = folds.loc[
        folds["split"].astype(str) == "development", ["id", "development_fold"]
    ].rename(columns={"development_fold": "fold"})
    if len(development) != 11118:
        raise ValueError(f"unexpected development scope: {len(development)}")
    merged = development.merge(frame, on="id", how="left", validate="one_to_one")
    if merged["category"].isna().any():
        raise ValueError("development rows do not map one-to-one to source data")
    with gzip.open(args.image_manifest, "rt", encoding="utf-8", newline="") as stream:
        urls = {str(row["id"]): row["image_url"] for row in csv.DictReader(stream, delimiter="\t")}
    if set(merged["id"].astype(str)) != set(urls):
        raise ValueError("image manifest does not exactly cover development rows")
    merged = merged.sort_values("id", key=lambda values: values.astype(str)).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for global_index, row in merged.iterrows():
            record = {
                "global_index": int(global_index),
                "id": str(row["id"]),
                "fold": int(row["fold"]),
                "category": str(row["category"]),
                "name": clean(row["name"], 320),
                "description": clean(row["description"], 1800),
                "image_url": urls[str(row["id"])],
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": 1,
        "rows": len(merged),
        "labels_read": 0,
        "sealed_rows": 0,
        "data_sha256": sha256_file(args.data),
        "folds_sha256": sha256_file(args.folds),
        "image_manifest_sha256": sha256_file(args.image_manifest),
        "output_sha256": sha256_file(args.output),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

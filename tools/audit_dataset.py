from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from ecup_quality.data.text import normalize_text

IMAGE_PATTERN = re.compile(r"^(?P<id>[^/]+?)/(?P<index>\d+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    required = {"id", "category", "label", "name", "description"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    ids = frame["id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("data contains duplicate ids")

    names = frame["name"].fillna("").astype(str).map(normalize_text)
    descriptions = frame["description"].fillna("").astype(str).map(normalize_text)
    full_text = names + " " + descriptions
    name_stats = frame.assign(group=names).groupby("group").label.agg(["count", "nunique"])
    text_stats = frame.assign(group=full_text).groupby("group").label.agg(["count", "nunique"])

    image_indices: dict[str, list[int]] = {}
    invalid_names = []
    for path in args.images.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(args.images).as_posix()
        match = IMAGE_PATTERN.match(relative)
        if match is None:
            invalid_names.append(relative)
            continue
        image_indices.setdefault(match.group("id"), []).append(int(match.group("index")))

    expected_ids = set(ids)
    actual_ids = set(image_indices)
    non_contiguous = {
        item_id: sorted(indices)
        for item_id, indices in image_indices.items()
        if sorted(indices) != list(range(len(indices)))
    }
    distribution = Counter(len(image_indices[item_id]) for item_id in expected_ids & actual_ids)
    report = {
        "data_sha256": file_sha256(args.data),
        "rows": len(frame),
        "category_rows": frame.category.value_counts().sort_index().to_dict(),
        "category_positives": frame.groupby("category").label.sum().astype(int).to_dict(),
        "repeated_name_groups": int((name_stats["count"] > 1).sum()),
        "rows_in_repeated_name_groups": int(name_stats.loc[name_stats["count"] > 1, "count"].sum()),
        "conflicting_name_groups": int((name_stats["nunique"] > 1).sum()),
        "repeated_text_groups": int((text_stats["count"] > 1).sum()),
        "rows_in_repeated_text_groups": int(text_stats.loc[text_stats["count"] > 1, "count"].sum()),
        "conflicting_text_groups": int((text_stats["nunique"] > 1).sum()),
        "image_files": int(sum(len(values) for values in image_indices.values())),
        "image_count_distribution": {str(key): value for key, value in sorted(distribution.items())},
        "missing_image_ids": sorted(expected_ids - actual_ids),
        "extra_image_ids": sorted(actual_ids - expected_ids),
        "non_contiguous_image_indices": non_contiguous,
        "unrecognized_image_paths": invalid_names[:100],
    }
    report["status"] = "PASS" if not any([
        report["missing_image_ids"], report["extra_image_ids"],
        report["non_contiguous_image_indices"], report["unrecognized_image_paths"],
    ]) else "FAIL"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

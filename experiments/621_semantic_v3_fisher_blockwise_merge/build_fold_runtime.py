"""Build a development-only runtime for one semantic-v3 outer fold.

The output deliberately has separate train and validation files.  The
validation file is written from an allow-list that does not contain ``label``;
the GPU worker therefore cannot accidentally read validation supervision.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, TextIO

DEVELOPMENT_FOLDS = (0, 1, 2, 3, 4)
SAFE_MEMBERSHIP_COLUMNS = (
    "id",
    "category",
    "semantic_component",
    "component_size",
    "split",
    "development_fold",
)
BASE_DATA_COLUMNS = (
    "id",
    "name",
    "description",
    "category",
    "semantic_component",
    "development_fold",
)
TRAIN_DATA_COLUMNS = (*BASE_DATA_COLUMNS, "label")
VALIDATION_DATA_COLUMNS = (*BASE_DATA_COLUMNS,)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        rows = [{str(key): str(value) for key, value in raw.items()} for raw in reader]
        return [str(value) for value in reader.fieldnames], rows


def _read_membership(path: Path) -> list[dict[str, str]]:
    header, rows = _read_csv_rows(path)
    if not set(SAFE_MEMBERSHIP_COLUMNS).issubset(header):
        raise ValueError(f"{path} lacks semantic-v3 membership columns")
    safe = [{column: row[column] for column in SAFE_MEMBERSHIP_COLUMNS} for row in rows]
    if any(not row["id"] for row in safe) or len({row["id"] for row in safe}) != len(safe):
        raise ValueError("membership IDs must be non-empty and unique")
    return safe


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def _read_image_manifest(
    path: Path, development_ids: set[str]
) -> tuple[list[str], list[dict[str, str]]]:
    with _open_text(path) as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError("image manifest must be a TSV with an id column")
        fields = [str(value) for value in reader.fieldnames]
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for raw in reader:
            item_id = str(raw["id"])
            if item_id not in development_ids:
                continue
            if item_id in seen:
                raise ValueError("development image manifest contains duplicate IDs")
            seen.add(item_id)
            rows.append({field: str(raw[field]) for field in fields})
    if seen != development_ids:
        raise ValueError("image manifest does not cover every development ID")
    rows.sort(key=lambda row: row["id"])
    return fields, rows


def build_fold_runtime(
    *,
    data_path: Path,
    folds_path: Path,
    image_manifest_path: Path,
    output_dir: Path,
    outer_fold: int,
) -> dict[str, Any]:
    if outer_fold not in DEVELOPMENT_FOLDS:
        raise ValueError(f"outer_fold must be one of {DEVELOPMENT_FOLDS}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite fold runtime")

    source_header, source_rows = _read_csv_rows(data_path)
    required_source = {"id", "name", "description", "category", "label"}
    if not required_source.issubset(source_header):
        raise ValueError("source data lacks required product and label columns")
    source_by_id: dict[str, dict[str, str]] = {}
    for row in source_rows:
        item_id = row["id"]
        if not item_id or item_id in source_by_id:
            raise ValueError("source data IDs must be non-empty and unique")
        source_by_id[item_id] = row

    membership = _read_membership(folds_path)
    membership_by_id = {row["id"]: row for row in membership}
    if set(source_by_id) != set(membership_by_id):
        raise ValueError("source data and semantic-v3 membership IDs differ")
    if any(row["split"] not in {"development", "sealed_holdout"} for row in membership):
        raise ValueError("membership contains an unexpected split")
    sealed = [row for row in membership if row["split"] == "sealed_holdout"]
    if any(row["development_fold"] != "-1" for row in sealed):
        raise ValueError("sealed membership rows must have fold=-1")
    development = [row for row in membership if row["split"] == "development"]
    if {int(row["development_fold"]) for row in development} != set(DEVELOPMENT_FOLDS):
        raise ValueError("development folds must be exactly 0..4")
    components: dict[str, int] = {}
    for row in development:
        fold = int(row["development_fold"])
        previous = components.setdefault(row["semantic_component"], fold)
        if previous != fold:
            raise ValueError("semantic components cross development folds")
        source = source_by_id[row["id"]]
        if source["category"] != row["category"] or int(source["label"]) not in (0, 1):
            raise ValueError(f"source/membership target mismatch for {row['id']}")

    development_ids = {row["id"] for row in development}
    image_fields, image_rows = _read_image_manifest(image_manifest_path, development_ids)
    output_dir.mkdir(parents=True, exist_ok=False)

    def data_row(membership_row: dict[str, str]) -> dict[str, str]:
        source = source_by_id[membership_row["id"]]
        return {
            "id": membership_row["id"],
            "name": source["name"],
            "description": source["description"],
            "category": membership_row["category"],
            "semantic_component": membership_row["semantic_component"],
            "development_fold": membership_row["development_fold"],
            "label": source["label"],
        }

    all_development = [data_row(row) for row in development]
    train = [row for row in all_development if int(row["development_fold"]) != outer_fold]
    validation = [row for row in all_development if int(row["development_fold"]) == outer_fold]
    train_path = output_dir / "train_data.csv"
    validation_path = output_dir / "validation_data.csv"
    development_path = output_dir / "development_data_label_free.csv"
    membership_path = output_dir / "development_membership.csv"
    image_path = output_dir / "development_image_manifest.tsv"
    _write_csv(train_path, TRAIN_DATA_COLUMNS, train)
    _write_csv(validation_path, VALIDATION_DATA_COLUMNS, validation)
    _write_csv(development_path, VALIDATION_DATA_COLUMNS, all_development)
    _write_csv(membership_path, SAFE_MEMBERSHIP_COLUMNS, development)
    with image_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=image_fields, delimiter="\t", extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(image_rows)

    files = {
        path.name: sha256_file(path)
        for path in (train_path, validation_path, development_path, membership_path, image_path)
    }
    audit = {
        "protocol": "621_semantic_v3_fold_runtime_v1",
        "outer_fold": outer_fold,
        "development_rows": len(all_development),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "sealed_rows_written": 0,
        "validation_label_column_present": False,
        "validation_labels_written": False,
        "development_label_free": True,
        "image_rows": len(image_rows),
        "files_sha256": files,
        "decision": "GO",
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    (output_dir / "runtime_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a label-isolated semantic-v3 fold runtime.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--image-manifest", required=True, type=Path)
    parser.add_argument("--outer-fold", required=True, type=int, choices=DEVELOPMENT_FOLDS)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build_fold_runtime(
                data_path=args.data,
                folds_path=args.folds,
                image_manifest_path=args.image_manifest,
                output_dir=args.output_dir,
                outer_fold=args.outer_fold,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

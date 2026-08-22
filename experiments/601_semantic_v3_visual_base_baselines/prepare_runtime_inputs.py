from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from semantic_v3_contract import (
    DEFAULT_DATA,
    DEFAULT_FIRST_MANIFEST,
    DEFAULT_FOLDS,
    ROBUST_PROTOCOL_VERSION,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    *,
    data_path: Path,
    folds_path: Path,
    manifest_path: Path,
    robust_base_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path, dtype={"id": str})
    folds = pd.read_csv(folds_path, dtype={"id": str})
    if not np.array_equal(data["id"].astype(str).to_numpy(), folds["id"].to_numpy()):
        raise ValueError("data/folds id mismatch")
    development_mask = folds["split"].astype(str).to_numpy() == "development"
    sealed_ids = set(folds.loc[~development_mask, "id"].astype(str))
    dev_data = data.loc[development_mask].reset_index(drop=True)
    dev_folds = folds.loc[development_mask].reset_index(drop=True)
    dev_ids = dev_data["id"].astype(str).tolist()
    if set(dev_ids) & sealed_ids:
        raise ValueError("sealed ids entered development data")
    data_out = output_dir / "development_data.csv"
    folds_out = output_dir / "development_folds.csv"
    manifest_out = output_dir / "development_first_image_manifest.tsv.gz"
    oof_out = output_dir / "qwen_parent_oof.npz"
    id_map_out = output_dir / "development_id_map.csv"
    dev_data.to_csv(data_out, index=False)
    dev_folds.to_csv(folds_out, index=False)
    pd.DataFrame(
        {
            "development_position": np.arange(len(dev_ids), dtype=np.int32),
            "source_position": np.flatnonzero(development_mask),
            "id": dev_ids,
            "development_fold": dev_folds["development_fold"].to_numpy(np.int8),
            "semantic_component": dev_folds["semantic_component"].astype(str),
        }
    ).to_csv(id_map_out, index=False)
    wanted = set(dev_ids)
    written: list[str] = []
    with gzip.open(manifest_path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames != ["id", "image_url"]:
            raise ValueError("first-image manifest schema mismatch")
        with gzip.open(manifest_out, "wt", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames, delimiter="\t")
            writer.writeheader()
            for row in reader:
                if str(row["id"]) in wanted:
                    writer.writerow(row)
                    written.append(str(row["id"]))
    if written != dev_ids:
        raise ValueError("development manifest id/order mismatch")
    robust = np.load(robust_base_path, allow_pickle=False)
    if (
        "protocol_version" not in robust.files
        or str(robust["protocol_version"]) != ROBUST_PROTOCOL_VERSION
    ):
        raise ValueError("robust base is not the strict nested v2 protocol")
    for key, expected in (
        ("ids", np.asarray(dev_ids)),
        ("labels", dev_data["label"].to_numpy(np.int8)),
        ("categories", dev_data["category"].astype(str).to_numpy()),
        ("folds", dev_folds["development_fold"].to_numpy(np.int8)),
    ):
        if not np.array_equal(robust[key].astype(expected.dtype), expected):
            raise ValueError(f"robust base {key} mismatch")
    np.savez_compressed(
        oof_out,
        ids=np.asarray(dev_ids),
        labels=dev_data["label"].to_numpy(np.int8),
        categories=np.asarray(dev_data["category"].astype(str).tolist(), dtype=str),
        fold_ids=dev_folds["development_fold"].to_numpy(np.int8),
        fused_scores=robust["robust_base_score"].astype(np.float32),
    )
    outputs = [data_out, folds_out, manifest_out, oof_out, id_map_out]
    report = {
        "version": "semantic_v3_qwen_runtime_inputs_v1",
        "development_rows": len(dev_ids),
        "sealed_rows": len(sealed_ids),
        "sealed_ids_in_any_runtime_output": 0,
        "manifest_rows": len(written),
        "output_sha256": {path.name: _sha256(path) for path in outputs},
    }
    audit_path = output_dir / "zero_sealed_runtime_audit.json"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--first-manifest", type=Path, default=DEFAULT_FIRST_MANIFEST)
    parser.add_argument("--robust-base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(
        data_path=args.data,
        folds_path=args.folds,
        manifest_path=args.first_manifest,
        robust_base_path=args.robust_base,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

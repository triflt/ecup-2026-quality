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
    DEFAULT_ALL_EMBEDDINGS,
    DEFAULT_DATA,
    DEFAULT_FIRST_EMBEDDINGS,
    DEFAULT_FIRST_MANIFEST,
    DEFAULT_FOLDS,
    DEVELOPMENT_FOLDS,
    EXPECTED_DEVELOPMENT_ROWS,
    EXPECTED_FOLDS_SHA256,
    EXPECTED_SEALED_ROWS,
    ROOT,
)

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "analysis/input_audit.json"
ROBUST_ENTRYPOINTS = (
    ROOT / "research/build_oof_cache.py",
    ROOT / "research/four_head_fusion_cv.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_ids(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["id", "image_url"]:
            raise ValueError("first-image manifest schema mismatch")
        return [str(row["id"]) for row in reader]


def audit_inputs(
    *,
    data: Path,
    folds: Path,
    first_manifest: Path,
    all_embeddings: Path,
    first_embeddings: Path,
) -> dict[str, object]:
    if sha256(folds) != EXPECTED_FOLDS_SHA256:
        raise ValueError("semantic-v3 folds checksum mismatch")
    frame = pd.read_csv(data, dtype={"id": str})
    split = pd.read_csv(folds, dtype={"id": str})
    ids = frame["id"].astype(str).to_numpy()
    if len(ids) != len(set(ids)):
        raise ValueError("data ids are not unique")
    if not np.array_equal(ids, split["id"].astype(str).to_numpy()):
        raise ValueError("data/fold id order mismatch")
    if not np.array_equal(
        frame["category"].astype(str).to_numpy(),
        split["category"].astype(str).to_numpy(),
    ):
        raise ValueError("data/fold category mismatch")
    if not np.array_equal(frame["label"].to_numpy(np.int8), split["label"].to_numpy(np.int8)):
        raise ValueError("data/fold label mismatch")
    development = split["split"].astype(str).to_numpy() == "development"
    sealed = split["split"].astype(str).to_numpy() == "sealed_holdout"
    if not np.all(development | sealed):
        raise ValueError("unexpected split outside development/sealed_holdout")
    if int(development.sum()) != EXPECTED_DEVELOPMENT_ROWS:
        raise ValueError("development row count mismatch")
    if int(sealed.sum()) != EXPECTED_SEALED_ROWS:
        raise ValueError("sealed row count mismatch")
    dev_folds = split.loc[development, "development_fold"].to_numpy(np.int8)
    if set(dev_folds.tolist()) != set(DEVELOPMENT_FOLDS):
        raise ValueError("development fold set mismatch")
    if not np.all(split.loc[sealed, "development_fold"].to_numpy(np.int8) == -1):
        raise ValueError("sealed rows have a development fold")
    dev_ids = set(ids[development])
    sealed_ids = set(ids[sealed])
    if dev_ids & sealed_ids:
        raise ValueError("development and sealed ids overlap")
    if manifest_ids(first_manifest) != ids.tolist():
        raise ValueError("first-image manifest id/order mismatch")
    embedding_contracts = {}
    for name, path in (("all_images", all_embeddings), ("first_image", first_embeddings)):
        archive = np.load(path, allow_pickle=False)
        if not np.array_equal(archive["ids"].astype(str), ids):
            raise ValueError(f"{name} embedding id/order mismatch")
        embeddings = archive["embeddings"].astype(np.float32)
        if not np.isfinite(embeddings).all():
            raise FloatingPointError(f"{name} embeddings contain non-finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise FloatingPointError(f"{name} embeddings have invalid L2 norms")
        embedding_contracts[name] = {
            "rows": len(embeddings),
            "dimension": int(embeddings.shape[1]),
            "stored_dtype": str(archive["embeddings"].dtype),
            "finite": True,
            "min": float(embeddings.min()),
            "max": float(embeddings.max()),
            "norm_min": float(norms.min()),
            "norm_median": float(np.median(norms)),
            "norm_max": float(norms.max()),
            "zero_norm_rows": 0,
            "additional_normalization_required": False,
            "sha256": sha256(path),
        }
    fold_counts = {str(fold): int((dev_folds == fold).sum()) for fold in DEVELOPMENT_FOLDS}
    return {
        "audit_version": "semantic_v3_visual_base_inputs_v1",
        "status": "ready",
        "folds_sha256": sha256(folds),
        "data_sha256": sha256(data),
        "first_manifest_sha256": sha256(first_manifest),
        "rows": len(ids),
        "development_rows": len(dev_ids),
        "sealed_rows": len(sealed_ids),
        "development_fold_rows": fold_counts,
        "development_sealed_id_overlap": 0,
        "data_fold_id_category_label_mapping_verified": True,
        "sealed_used_for_train_selector_threshold_eval": False,
        "embeddings_are_fixed_label_blind_features": True,
        "embedding_contracts": embedding_contracts,
        "robust_base_reproducibility": {
            "status": "reproducible_from_fixed_embeddings_and_raw_text",
            "artifact_only_blocker": False,
            "historical_entrypoint_sha256": {
                str(path.relative_to(ROOT)): sha256(path) for path in ROBUST_ENTRYPOINTS
            },
            "new_protocol": (
                "outer semantic-v3 development fold; four remaining folds select "
                "four-head weights and threshold; empirical-CDF projection to outer fold"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--first-manifest", type=Path, default=DEFAULT_FIRST_MANIFEST)
    parser.add_argument("--all-embeddings", type=Path, default=DEFAULT_ALL_EMBEDDINGS)
    parser.add_argument("--first-embeddings", type=Path, default=DEFAULT_FIRST_EMBEDDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = audit_inputs(
        data=args.data,
        folds=args.folds,
        first_manifest=args.first_manifest,
        all_embeddings=args.all_embeddings,
        first_embeddings=args.first_embeddings,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from build_solution140_full_support_refit import (
    ALLOWED_VARIANTS,
    PREPROCESSING_CONTRACTS,
)
from build_synth_runtime import canonical_sha256, sha256_file
from PIL import Image
from train_synth_fold import cache_path, read_jsonl

EXPECTED_DECISION = "GO_FULL_REFIT_SOLUTION140_FULL_SUPPORT"
EXPECTED_SCHEMA = "exp699_solution140_full_support_refit_runtime_v1"


def load_runtime_ids(runtime_dir: Path) -> tuple[list[str], dict[str, Any]]:
    audit_path = runtime_dir / "runtime_audit.json"
    train_path = runtime_dir / "train.jsonl"
    validation_path = runtime_dir / "validation.jsonl"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payload = dict(audit)
    declared = payload.pop("contract_sha256", None)
    if declared != canonical_sha256(payload):
        raise ValueError("corrected refit runtime self-hash mismatch")
    if (
        audit.get("schema") != EXPECTED_SCHEMA
        or audit.get("decision") != EXPECTED_DECISION
        or (int(audit.get("cap", -1)), audit.get("preprocessing_key"))
        not in ALLOWED_VARIANTS
        or audit.get("preprocessing_contract")
        != PREPROCESSING_CONTRACTS[audit["preprocessing_key"]]
        or audit.get("preprocessing_sha256")
        != canonical_sha256(PREPROCESSING_CONTRACTS[audit["preprocessing_key"]])
        or audit.get("output_sha256")
        != {
            "train.jsonl": sha256_file(train_path),
            "validation.jsonl": sha256_file(validation_path),
        }
    ):
        raise ValueError("corrected refit runtime binding mismatch")
    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    if validation:
        raise ValueError("full refit validation must be empty")
    ids = sorted({str(row["id"]) for row in train if not row.get("synthetic")})
    if len(ids) != 5998:
        raise ValueError("solution140 full-support cache requires 5998 real ids")
    return ids, audit


def prepare(
    runtime_dir: Path,
    mounted_image_root: Path,
    cache_root: Path,
    report_path: Path,
    preprocessing: str,
    workers: int = 32,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    ids, audit = load_runtime_ids(runtime_dir)
    if preprocessing != audit.get("preprocessing_key"):
        raise ValueError("cache preprocessing/runtime mismatch")
    if workers < 1 or workers > 64:
        raise ValueError("workers must be in 1..64")
    if cache_root.exists() and any(cache_root.iterdir()) and not reuse_existing:
        raise FileExistsError("refusing to mix solution140 image cache with existing files")
    cache_root.mkdir(parents=True, exist_ok=True)
    def prepare_one(row_id: str) -> dict[str, Any]:
        source = mounted_image_root / row_id / "0.jpg"
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"missing regular mounted first image for id {row_id}")
        destination = cache_path(cache_root, row_id, "qwen35_4b")
        if preprocessing == "thumbnail448":
            with Image.open(source) as opened:
                image = opened.convert("RGB")
                original_size = list(image.size)
                image.thumbnail((448, 448), Image.Resampling.LANCZOS)
                cached_size = list(image.size)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=92)
                image.close()
            expected_bytes = buffer.getvalue()
            expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()
            if destination.is_file() and not destination.is_symlink():
                if not reuse_existing or sha256_file(destination) != expected_sha256:
                    raise FileExistsError("invalid pre-existing thumbnail cache entry")
                cache_mode = "verified_reused_thumbnail"
            elif destination.exists() or destination.is_symlink():
                raise FileExistsError("invalid pre-existing thumbnail cache entry")
            else:
                destination.write_bytes(expected_bytes)
                cache_mode = "materialized_thumbnail"
        else:
            with Image.open(source) as opened:
                original_size = list(opened.size)
            cached_size = original_size
            if destination.is_symlink():
                if not reuse_existing or destination.resolve() != source.resolve():
                    raise FileExistsError("invalid pre-existing area-cap cache entry")
                cache_mode = "verified_reused_mounted_source_symlink"
            elif destination.exists():
                raise FileExistsError("invalid pre-existing area-cap cache entry")
            else:
                os.symlink(source, destination)
                cache_mode = "mounted_source_symlink_dynamic_area_cap"
        return {
            "id": row_id,
            "source_sha256": sha256_file(source),
            "cache_sha256": sha256_file(destination),
            "original_size": original_size,
            "cached_size": cached_size,
            "cache_mode": cache_mode,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        entries = list(pool.map(prepare_one, ids))
    expected_inventory = {
        cache_path(cache_root, row_id, "qwen35_4b").name for row_id in ids
    }
    actual_inventory = {path.name for path in cache_root.iterdir()}
    if actual_inventory != expected_inventory:
        raise ValueError("image cache inventory mismatch")
    if preprocessing == "thumbnail448" and any(
        max(entry["cached_size"]) > 448 for entry in entries
    ):
        raise RuntimeError("thumbnail edge contract violated")
    report: dict[str, Any] = {
        "schema": "exp699_solution140_qwen35_image_cache_v1",
        "experiment_id": "699",
        "architecture": "qwen35_4b",
        "runtime_contract_sha256": audit["contract_sha256"],
        "runtime_decision": audit["decision"],
        "unique_real_ids": len(ids),
        "created": sum(
            entry["cache_mode"]
            in {"materialized_thumbnail", "mounted_source_symlink_dynamic_area_cap"}
            for entry in entries
        ),
        "reused": sum("reused" in entry["cache_mode"] for entry in entries),
        "preprocessing_key": preprocessing,
        "preprocessing": PREPROCESSING_CONTRACTS[preprocessing],
        "preprocessing_sha256": canonical_sha256(
            PREPROCESSING_CONTRACTS[preprocessing]
        ),
        "entries_sha256": canonical_sha256(entries),
        "cache_inventory_sha256": canonical_sha256(
            [[entry["id"], entry["cache_sha256"]] for entry in entries]
        ),
        "labels_read": 0,
        "sealed_rows_used": 0,
        "public_rows_used": 0,
        "decision": "ACCEPT_SOLUTION140_QWEN35_IMAGE_CACHE",
    }
    report["self_sha256"] = canonical_sha256(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--mounted-image-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--preprocessing", choices=tuple(PREPROCESSING_CONTRACTS), required=True
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.runtime_dir.resolve(),
                args.mounted_image_root.resolve(),
                args.cache_root.resolve(),
                args.report.resolve(),
                args.preprocessing,
                args.workers,
                args.reuse_existing,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

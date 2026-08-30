from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from build_refit_runtime import consensus_synthetic, synthetic_row
from build_synth_runtime import canonical_sha256, sha256_file, write_jsonl

BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"
FOLDS = (0, 1, 2, 3, 4)
SEED = 42
EXPECTED_REAL_COUNTS = {
    "БАД:0": 1900,
    "БАД:1": 1900,
    "Легковоспламеняющиеся:0": 2000,
    "Легковоспламеняющиеся:1": 990,
}
EXPECTED_REAL_OCCURRENCES = 6790
EXPECTED_REAL_UNIQUE = 5998
EXPECTED_OPTIMIZER_STEPS = 425
PREPROCESSING_CONTRACTS = {
    "thumbnail448": {
        "name": "original_solution140_first_image_thumbnail_448",
        "mode": "PIL.Image.thumbnail",
        "maximum_size": [448, 448],
        "resampling": "LANCZOS",
        "color": "RGB",
        "output_format": "JPEG",
        "jpeg_quality": 92,
        "aspect_ratio_preserved": True,
        "upscale": False,
    },
    "area_cap262144": {
        "name": "area_cap_262144_train_infer",
        "mode": "area_cap",
        "maximum_pixels": 262144,
        "minimum_resized_edge": 28,
        "resampling": "LANCZOS",
        "color": "RGB",
        "source_cache": "regular_mounted_first_image_symlink",
        "aspect_ratio_preserved_except_minimum_edge_guard": True,
        "upscale": False,
    },
}
ALLOWED_VARIANTS = {
    (10, "thumbnail448"),
    (5, "thumbnail448"),
    (10, "area_cap262144"),
}
EXPECTED_INPUT_SHA256 = {
    "data": "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510",
    "oof": "d78bb7df897e244e030fa388e1b9c770d15434cc33aa1f56b238c7d057646116",
    "reference_code": "f9f508757458c6e6efa38f5e1f12913bdbf1db3acedc3b5ffbbcda3cc5234cf3",
    "reference_report": "44a90c62781d5d51c0916f517b5ea3a11f817c835cdebcdf93a271be269bcc55",
}
EXPECTED_SELECTION_SHA256 = {
    "ordered_indices": "2006a56d7499cf754fe9db34781117350a738fae513eec334eaa6864ad1f309d",
    "index_multiset": "0a31883f4e0b68a5fd9f6f803f660d02d67300cde30dd4f60564253e90a4af63",
    "unique_ids": "3e9e0bbdad4d2f832521375ab1a305a42de0e8b31e12bae29bf869b87df47ba1",
}


def hard_random(
    indices: np.ndarray,
    scores: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    """Byte-for-byte selection semantics from qwen35_family_balanced_lora.py."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= count:
        return indices.tolist()
    hard_count = count // 2
    # NumPy's default quicksort may reorder exact score ties differently across
    # versions.  The training runtime is materialized on remote compute, so make the
    # frozen selector order portable instead of binding it to one NumPy build.
    hard = indices[np.argsort(scores[indices], kind="stable")[:hard_count]]
    remaining = np.setdiff1d(indices, hard, assume_unique=False)
    random_part = rng.choice(remaining, size=count - hard_count, replace=False)
    return np.concatenate([hard, random_part]).tolist()


def select_solution140_full_support(frame: Any, oof: Any) -> tuple[list[int], dict[str, Any]]:
    """Reproduce the accepted 6,790-occurrence/5,998-unique full selector."""
    rng = np.random.default_rng(SEED)
    categories = frame["category"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int8)
    fused_key = "fused_scores" if "fused_scores" in oof.files else "fused"
    threshold_map = (
        {BAD: 0.24864045896205267, FLAMMABLE: 0.9591804083988902}
        if "fused_scores" in oof.files
        else {BAD: 0.251901438832283, FLAMMABLE: 0.9540719747543336}
    )
    thresholds = np.asarray([threshold_map[category] for category in categories], dtype=np.float32)
    uncertainty = np.abs(oof[fused_key].astype(np.float32) - thresholds)
    records: list[int] = []

    bad = np.flatnonzero(categories == BAD)
    bad_positive = bad[labels[bad] == 1]
    bad_negative = bad[labels[bad] == 0]
    bad_count = min(1900, len(bad_negative), len(bad_positive))
    records.extend(hard_random(bad_positive, uncertainty, bad_count, rng))
    records.extend(hard_random(bad_negative, uncertainty, bad_count, rng))

    flammable = np.flatnonzero(categories == FLAMMABLE)
    flammable_positive = flammable[labels[flammable] == 1]
    flammable_negative = flammable[labels[flammable] == 0]
    # The accepted full_train_report has 5,998 unique source rows. That is only
    # possible when all 198 flammable positives are retained and repeated 5x;
    # family resampling can omit rows within duplicate-text families. Preserve
    # the accepted artifact semantics rather than the later experimental
    # FAMILY_BALANCE_FLAMMABLE default in the reference source file.
    records.extend(np.repeat(flammable_positive, 5).tolist())
    selection_audit: dict[str, Any] = {
        "positive_rows": len(flammable_positive),
        "positive_families": None,
        "total_exposures": int(len(flammable_positive) * 5),
        "min_family_exposures": 5,
        "max_family_exposures": 5,
        "family_balanced": False,
        "accepted_report_unique_support": EXPECTED_REAL_UNIQUE,
    }
    records.extend(
        hard_random(
            flammable_negative,
            uncertainty,
            min(2000, len(flammable_negative)),
            rng,
        )
    )
    random.Random(SEED).shuffle(records)
    return records, selection_audit


def _load_reference_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "full_train": True,
        "training_mode": "hard",
        "train_records": EXPECTED_REAL_OCCURRENCES,
        "train_unique": EXPECTED_REAL_UNIQUE,
        "download_failures": 0,
        "optimizer_updates": EXPECTED_OPTIMIZER_STEPS,
    }
    mismatch = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatch:
        raise ValueError(f"solution140 reference report mismatch: {mismatch}")
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    import pandas as pd

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite corrected full-refit runtime")
    input_sha256 = {
        "data": sha256_file(args.data),
        "oof": sha256_file(args.oof),
        "reference_code": sha256_file(args.reference_code),
        "reference_report": sha256_file(args.reference_report),
    }
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise ValueError(f"solution140 frozen input SHA mismatch: {input_sha256}")
    frame = pd.read_csv(args.data)
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    oof = np.load(args.oof, allow_pickle=True)
    ids = frame["id"].astype(str).to_numpy()
    if not np.array_equal(ids, oof["ids"].astype(str)):
        raise ValueError("solution140 data/OOF id mismatch")
    labels = frame["label"].to_numpy(dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy()
    if not np.array_equal(labels, oof["labels"].astype(np.int8)):
        raise ValueError("solution140 data/OOF label mismatch")
    if not np.array_equal(categories, oof["categories"].astype(str)):
        raise ValueError("solution140 data/OOF category mismatch")
    folds = oof["fold_ids"].astype(np.int8)
    if set(folds.tolist()) != set(FOLDS):
        raise ValueError("solution140 OOF must contain exact folds0..4")
    reference_report = _load_reference_report(args.reference_report)

    selected_indices, flammable_selection = select_solution140_full_support(frame, oof)
    real_counts = Counter(f"{categories[index]}:{int(labels[index])}" for index in selected_indices)
    if dict(sorted(real_counts.items())) != EXPECTED_REAL_COUNTS:
        raise ValueError(f"solution140 selector count mismatch: {real_counts}")
    if (
        len(selected_indices) != EXPECTED_REAL_OCCURRENCES
        or len(set(selected_indices)) != EXPECTED_REAL_UNIQUE
    ):
        raise ValueError("solution140 selector support mismatch")
    selection_sha256 = {
        "ordered_indices": canonical_sha256(selected_indices),
        "index_multiset": canonical_sha256(sorted(Counter(selected_indices).items())),
        "unique_ids": canonical_sha256(
            sorted({str(frame.iloc[index]["id"]) for index in selected_indices})
        ),
    }
    if selection_sha256 != EXPECTED_SELECTION_SHA256:
        raise ValueError(f"solution140 exact selector SHA mismatch: {selection_sha256}")

    occurrence_counts: Counter[int] = Counter()
    real_train: list[dict[str, Any]] = []
    for index in selected_indices:
        occurrence_index = occurrence_counts[index]
        occurrence_counts[index] += 1
        source = frame.iloc[index]
        real_train.append(
            {
                "category": str(source["category"]),
                "description": str(source["description"]),
                "fold": int(folds[index]),
                "global_index": int(index),
                "id": str(source["id"]),
                "label": int(source["label"]),
                "name": str(source["name"]),
                "occurrence_index": int(occurrence_index),
                "ocr_images": [],
                "semantic_component": hashlib.sha256(
                    f"solution140-full-support:{index}".encode()
                ).hexdigest(),
                "synthetic": False,
            }
        )

    if (args.synth_cap, args.preprocessing) not in ALLOWED_VARIANTS:
        raise ValueError("refit variant is not preregistered")
    selected_synth, ranked_hashes = consensus_synthetic(
        {fold: path for fold, path in args.ranked}, source="v2", cap=args.synth_cap
    )
    extras = [synthetic_row(row, index) for index, row in enumerate(selected_synth)]
    train = real_train + extras
    expected_train_occurrences = EXPECTED_REAL_OCCURRENCES + args.synth_cap
    if len(train) != expected_train_occurrences:
        raise RuntimeError("corrected refit train occurrence mismatch")
    if math.ceil(len(train) / 16) != EXPECTED_OPTIMIZER_STEPS:
        raise RuntimeError("corrected refit must preserve 425 optimizer updates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, [])
    preprocessing_contract = PREPROCESSING_CONTRACTS[args.preprocessing]
    report: dict[str, Any] = {
        "schema": "exp699_solution140_full_support_refit_runtime_v1",
        "schema_version": 1,
        "experiment_id": "699",
        "stage": "full_refit",
        "outer_fold": -1,
        "parent_experiment_id": "140",
        "source": "v2",
        "mode": "positive_only_append",
        "cap": args.synth_cap,
        "augmentation_arm": "synth_append",
        "selection": "solution140_qwen35_full_train_hard_all_positive_5x_exact",
        "image_policy": preprocessing_contract["name"],
        "preprocessing_key": args.preprocessing,
        "preprocessing_contract": preprocessing_contract,
        "preprocessing_sha256": canonical_sha256(preprocessing_contract),
        "original_real_occurrences": len(real_train),
        "original_real_unique": len(set(selected_indices)),
        "real_selector_counts": dict(sorted(real_counts.items())),
        "train_occurrences": len(train),
        "synthetic_occurrences": len(extras),
        "appended_occurrences": len(extras),
        "appended_positive_occurrences": len(extras),
        "appended_negative_occurrences": 0,
        "removed_real_occurrences": 0,
        "original_real_rows_changed": 0,
        "bad_rows_changed": 0,
        "validation_rows": 0,
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "public_rows_used": 0,
        "effective_batch": 16,
        "epochs": args.epochs,
        "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS * args.epochs,
        "selected_real_ordered_indices_sha256": selection_sha256["ordered_indices"],
        "selected_real_multiset_sha256": selection_sha256["index_multiset"],
        "selected_real_unique_ids_sha256": selection_sha256["unique_ids"],
        "selected_synth_ids_sha256": canonical_sha256(
            [str(row["candidate_id"]) for row in selected_synth]
        ),
        "selected_synth_payload_sha256": canonical_sha256(
            [
                {
                    "candidate_id": str(row["candidate_id"]),
                    "category": str(row["category"]),
                    "description": str(row["description"]),
                    "label": int(row["label"]),
                    "name": str(row["name"]),
                    "source": str(row["source"]),
                }
                for row in selected_synth
            ]
        ),
        "flammable_family_selection": flammable_selection,
        "reference_report": {
            "file_sha256": sha256_file(args.reference_report),
            "train_records": reference_report["train_records"],
            "train_unique": reference_report["train_unique"],
            "optimizer_updates": reference_report["optimizer_updates"],
        },
        "input_sha256": {
            **input_sha256,
            "ranked_folds": ranked_hashes,
        },
        "output_sha256": {
            "train.jsonl": sha256_file(train_path),
            "validation.jsonl": sha256_file(validation_path),
        },
        "decision": "GO_FULL_REFIT_SOLUTION140_FULL_SUPPORT",
    }
    report["contract_sha256"] = canonical_sha256(report)
    (args.output_dir / "runtime_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_ranked(value: str) -> tuple[int, Path]:
    fold_text, separator, path_text = value.partition("=")
    if not separator:
        raise ValueError("ranked input must be FOLD=PATH")
    fold = int(fold_text)
    if fold not in FOLDS:
        raise ValueError("ranked fold must be 0..4")
    return fold, Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--reference-code", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--ranked", action="append", type=parse_ranked, required=True)
    parser.add_argument("--synth-cap", type=int, choices=(5, 10), required=True)
    parser.add_argument(
        "--preprocessing",
        choices=tuple(PREPROCESSING_CONTRACTS),
        required=True,
    )
    parser.add_argument("--epochs", type=int, choices=(1,), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.ranked) != 5 or len(dict(args.ranked)) != 5:
        raise ValueError("exact unique ranked folds0..4 are required")
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

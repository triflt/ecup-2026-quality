from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from build_synth_runtime import FLAMMABLE, canonical_sha256, sha256_file, write_jsonl

BAD = "БАД"
FOLDS = (0, 1, 2, 3, 4)
SEED = 42


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def hard_random(
    indices: np.ndarray,
    uncertainty: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= count:
        return indices.tolist()
    hard_count = count // 2
    order = np.lexsort((indices, uncertainty[indices]))
    hard = indices[order[:hard_count]]
    remaining = np.setdiff1d(indices, hard, assume_unique=False)
    random_part = rng.choice(remaining, size=count - hard_count, replace=False)
    return np.concatenate([hard, random_part]).tolist()


def select_full_refit_indices(
    labels: np.ndarray,
    categories: np.ndarray,
    fused_scores: np.ndarray,
    *,
    seed: int = SEED,
) -> list[int]:
    threshold_map = {BAD: 0.24864045896205267, FLAMMABLE: 0.9591804083988902}
    thresholds = np.asarray([threshold_map[str(value)] for value in categories])
    uncertainty = np.abs(fused_scores.astype(np.float32) - thresholds)
    rng = np.random.default_rng(seed)
    records: list[int] = []

    bad = np.flatnonzero(categories == BAD)
    bad_pos = bad[labels[bad] == 1]
    bad_neg = bad[labels[bad] == 0]
    bad_count = min(1500, len(bad_pos), len(bad_neg))
    records.extend(hard_random(bad_pos, uncertainty, bad_count, rng))
    records.extend(hard_random(bad_neg, uncertainty, bad_count, rng))

    flammable = np.flatnonzero(categories == FLAMMABLE)
    flammable_pos = flammable[labels[flammable] == 1]
    flammable_neg = flammable[labels[flammable] == 0]
    records.extend(np.repeat(flammable_pos, 5).tolist())
    records.extend(
        hard_random(flammable_neg, uncertainty, min(1600, len(flammable_neg)), rng)
    )
    random.Random(seed).shuffle(records)
    return records


def consensus_synthetic(
    ranked_paths: dict[int, Path], *, source: str, cap: int
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if set(ranked_paths) != set(FOLDS):
        raise ValueError("exact ranked folds0..4 are required")
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hashes = {}
    for fold, path in sorted(ranked_paths.items()):
        rows = read_jsonl(path)
        if not rows or any(int(row.get("selector_fold", -1)) != fold for row in rows):
            raise ValueError(f"ranked fold{fold} binding mismatch")
        hashes[str(fold)] = sha256_file(path)
        for row in rows:
            if (
                row.get("source") == source
                and row.get("category") == FLAMMABLE
                and type(row.get("label")) is int
                and int(row["label"]) == 1
            ):
                by_id[str(row["candidate_id"])].append(row)
    candidates = []
    for candidate_id, rows in by_id.items():
        if len(rows) != len(FOLDS):
            # Fold-local eligibility removes some candidates from some ranked
            # manifests. Full refit may only use the exact five-fold
            # intersection; partial candidates are ineligible, not a malformed
            # packet.
            continue
        if {int(row["selector_fold"]) for row in rows} != set(FOLDS):
            raise ValueError("synthetic candidate selector folds are duplicated")
        invariant = {
            (str(row["source"]), str(row["category"]), int(row["label"]), str(row["name"]), str(row["description"]))
            for row in rows
        }
        if len(invariant) != 1:
            raise ValueError("synthetic candidate payload differs across folds")
        candidates.append(
            (
                float(np.mean([int(row["label_rank"]) for row in rows])),
                candidate_id,
                rows[0],
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < cap:
        raise ValueError("insufficient consensus synthetic positives")
    return [dict(item[2]) for item in candidates[:cap]], hashes


def synthetic_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    candidate_id = str(row["candidate_id"])
    result = {
        "category": FLAMMABLE,
        "description": str(row["description"]),
        "fold": -1,
        "global_index": -(index + 1),
        "id": f"synth-refit-v2-{candidate_id}",
        "label": 1,
        "name": str(row["name"]),
        "occurrence_index": 0,
        "ocr_images": [],
        "semantic_component": hashlib.sha256(
            f"exp699:refit:v2:{candidate_id}".encode()
        ).hexdigest(),
        "synthetic": True,
        "augmentation_arm": "synth_append",
        "image_policy": "text_only",
        "synthetic_candidate_id": candidate_id,
        "synthetic_source": "v2",
    }
    return result


def load_development_rows(
    parent_root: Path,
    runtime_map_path: Path,
    runtime_contract_path: Path,
    oof_path: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    runtime_contract = json.loads(runtime_contract_path.read_text(encoding="utf-8"))
    payload = dict(runtime_contract)
    declared = payload.pop("self_sha256", None)
    if declared != canonical_sha256(payload):
        raise ValueError("runtime map contract self-hash mismatch")
    if (
        runtime_contract.get("map_sha256") != sha256_file(runtime_map_path)
        or runtime_contract.get("labels_read") != 0
        or runtime_contract.get("sealed_rows") != 0
        or runtime_contract.get("public_used") is not False
    ):
        raise ValueError("runtime map contract gate mismatch")
    map_rows = read_jsonl(runtime_map_path)
    if len(map_rows) != 11118:
        raise ValueError("development runtime map must contain 11118 rows")
    map_by_key = {
        (str(row["id"]), int(row["fold"]), str(row["category"])): row
        for row in map_rows
    }
    if len(map_by_key) != len(map_rows):
        raise ValueError("runtime map key is duplicated")

    rows = []
    parent_bindings = {}
    for fold in FOLDS:
        root = parent_root / f"fold{fold}"
        audit_path = root / "runtime_audit.json"
        validation_path = root / "validation.jsonl"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_payload = dict(audit)
        audit_declared = audit_payload.pop("contract_sha256", None)
        if (
            audit_declared != canonical_sha256(audit_payload)
            or int(audit.get("outer_fold", -1)) != fold
            or audit.get("decision") != "GO"
            or audit.get("output_sha256", {}).get("validation.jsonl")
            != sha256_file(validation_path)
            or audit.get("validation_labels_written") != 0
            or audit.get("sealed_rows_written") != 0
        ):
            raise ValueError(f"parent fold{fold} validation binding mismatch")
        local = read_jsonl(validation_path)
        if any("label" in row for row in local):
            raise ValueError("parent validation labels are forbidden")
        rows.extend(local)
        parent_bindings[str(fold)] = {
            "contract_sha256": audit_declared,
            "validation_sha256": sha256_file(validation_path),
        }
    rows.sort(key=lambda row: int(row["global_index"]))
    if len(rows) != 11118 or len({str(row["id"]) for row in rows}) != 11118:
        raise ValueError("parent validation union is not exact development")
    for row in rows:
        key = (str(row["id"]), int(row["fold"]), str(row["category"]))
        mapped = map_by_key.get(key)
        if mapped is None or int(mapped["global_index"]) != int(row["global_index"]):
            raise ValueError("parent validation/runtime map mismatch")

    oof = np.load(oof_path, allow_pickle=True)
    oof_ids = oof["ids"].astype(str)
    oof_categories = oof["categories"].astype(str)
    position = {
        (str(row_id), str(category)): index
        for index, (row_id, category) in enumerate(
            zip(oof_ids, oof_categories, strict=True)
        )
    }
    indices = np.asarray(
        [position[(str(row["id"]), str(row["category"]))] for row in rows],
        dtype=np.int64,
    )
    labels = oof["labels"].astype(np.int8)[indices]
    categories = oof_categories[indices]
    fused = oof["fused"].astype(np.float32)[indices]
    return rows, labels, categories, fused, parent_bindings


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite full-refit runtime")
    rows, labels, categories, fused, parent_bindings = load_development_rows(
        args.parent_root,
        args.runtime_map,
        args.runtime_map_contract,
        args.oof,
    )
    selected_indices = select_full_refit_indices(labels, categories, fused)
    real_train = []
    for occurrence_index, index in enumerate(selected_indices):
        real_train.append(
            {**rows[index], "occurrence_index": occurrence_index, "label": int(labels[index])}
        )
    selected_synth, ranked_hashes = consensus_synthetic(
        {fold: path for fold, path in args.ranked}, source="v2", cap=10
    )
    extras = [synthetic_row(row, index) for index, row in enumerate(selected_synth)]
    train = real_train + extras
    before_bad = Counter(
        (str(row["id"]), int(row["label"]))
        for row in real_train
        if row["category"] == BAD
    )
    after_bad = Counter(
        (str(row["id"]), int(row["label"]))
        for row in train
        if row["category"] == BAD
    )
    if before_bad != after_bad:
        raise RuntimeError("BAD refit multiset changed while appending synth")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, [])
    expected_real_counts = {"БАД:0": 1500, "БАД:1": 1500, "Легковоспламеняющиеся:0": 1600, "Легковоспламеняющиеся:1": 850}
    actual_real_counts = {
        f"{category}:{label}": int(
            sum(row["category"] == category and int(row["label"]) == label for row in real_train)
        )
        for category in (BAD, FLAMMABLE)
        for label in (0, 1)
    }
    if actual_real_counts != expected_real_counts:
        raise ValueError(f"unexpected full-refit selector counts: {actual_real_counts}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "699",
        "stage": "full_refit",
        "outer_fold": -1,
        "parent_experiment_id": "641",
        "source": "v2",
        "mode": "positive_only_append",
        "cap": 10,
        "augmentation_arm": "synth_append",
        "image_policy": "all_real_unchanged_plus_text_only_extra",
        "selection": "full_development_641_selector_semantics",
        "original_real_occurrences": len(real_train),
        "real_selector_counts": actual_real_counts,
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
        "expected_optimizer_steps": math.ceil(len(train) / 16) * args.epochs,
        "selected_real_multiset_sha256": canonical_sha256(
            sorted(Counter(str(row["id"]) for row in real_train).items())
        ),
        "selected_synth_ids_sha256": canonical_sha256(
            [str(row["candidate_id"]) for row in selected_synth]
        ),
        "input_sha256": {
            "oof": sha256_file(args.oof),
            "runtime_map": sha256_file(args.runtime_map),
            "runtime_map_contract": sha256_file(args.runtime_map_contract),
            "ranked_folds": ranked_hashes,
            "parent_folds": parent_bindings,
        },
        "output_sha256": {
            "train.jsonl": sha256_file(train_path),
            "validation.jsonl": sha256_file(validation_path),
        },
        "decision": "GO_FULL_REFIT",
    }
    report["contract_sha256"] = canonical_sha256(report)
    (args.output_dir / "runtime_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_ranked(value: str) -> tuple[int, Path]:
    fold_text, sep, path_text = value.partition("=")
    if not sep:
        raise ValueError("ranked input must be FOLD=PATH")
    fold = int(fold_text)
    if fold not in FOLDS:
        raise ValueError("ranked fold must be 0..4")
    return fold, Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--runtime-map", type=Path, required=True)
    parser.add_argument("--runtime-map-contract", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--ranked", action="append", type=parse_ranked, required=True)
    parser.add_argument("--epochs", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.ranked) != 5 or len(dict(args.ranked)) != 5:
        raise ValueError("exact unique ranked folds0..4 are required")
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

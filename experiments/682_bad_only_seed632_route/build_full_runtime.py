"""Build the exact development-only seed-632 full-refit runtime on CPU.

The policy is validated before any supervised source file is opened.  The
builder never accepts competition data or the sealed registry.  It reconstructs
the 11,118 development rows only from five checksum-bound, already label-
isolated experiment-623 outer-train runtimes: every development row must occur
with one identical supervised payload in exactly four of those runtimes.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contract import (
    add_self_hash,
    canonical_sha256,
    load_json,
    load_spec,
    require_sha,
    sha256_file,
    verify_self_hash,
)

PARENT_SHA256 = "c30e690ad260af72fcc625c8d3e6d9ab9c5a096d8443d6d9f5f7adbcaa52123c"
SELECTOR_FUNCTIONS = {
    "rank01",
    "fused_oof_scores",
    "oof_thresholds",
    "hard_random",
    "select_training",
}


def verify_policy(*, policy_path: Path, spec_path: Path) -> dict[str, Any]:
    """Gate before data access; deliberately has no source-root argument."""

    spec = load_spec(spec_path)
    policy = load_json(policy_path)
    verify_self_hash(policy, "policy_sha256")
    expected = {
        "schema_version": spec["full_refit"]["required_policy_schema"],
        "experiment_id": "682",
        "decision": "GO_DEVELOPMENT_ONLY_POST_VALIDATION_REFIT",
        "frozen_before_gpu_launch": True,
        "public_feedback_used": False,
        "sealed_rows_allowed": 0,
        "permitted_splits": ["development"],
        "development_rows": 11118,
        "seed": 31415,
        "selected_records": 6116,
        "selected_unique_rows": 5436,
        "selected_id_multiset_sha256": "f4734df62119ca67d86e22681fb09544b8c56761c6ffaf7617875cccdb983e9f",
        "optimizer_updates": 383,
        "parent_runner_sha256": PARENT_SHA256,
    }
    mismatch = {
        key: {"expected": value, "actual": policy.get(key)}
        for key, value in expected.items()
        if policy.get(key) != value
    }
    if mismatch:
        raise ValueError(f"post-validation refit policy mismatch: {mismatch}")
    if spec["full_refit"]["enabled"] is not True:
        raise PermissionError("frozen spec does not authorize development-only refit")
    return policy


def _exact_selector(parent_path: Path):
    require_sha(parent_path, PARENT_SHA256, name="exact experiment-600 parent")
    tree = ast.parse(parent_path.read_text(encoding="utf-8"), filename=str(parent_path))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in SELECTOR_FUNCTIONS
    ]
    if {node.name for node in nodes} != SELECTOR_FUNCTIONS:
        raise ValueError("exact parent selector function set changed")
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "np": np,
        "random": random,
        "SEED": 31415,
        "FULL_TRAIN": True,
        "HOLDOUT_FOLD": -1,
        "TRAINING_MODE": "hard",
    }
    exec(compile(module, str(parent_path), "exec"), namespace)
    return namespace["select_training"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"JSON object required in {path}")
            rows.append(item)
    return rows


def _load_development_rows(
    *, source_root: Path, source_manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    copies: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    observed_hashes: dict[str, str] = {}
    frozen_inputs = source_manifest["frozen_input_sha256"]
    for fold in range(5):
        fold_dir = source_root / f"fold{fold}"
        record = source_manifest["folds"][str(fold)]
        audit_path = fold_dir / "runtime_audit.json"
        train_path = fold_dir / "train.jsonl"
        require_sha(audit_path, record["runtime_audit_sha256"], name=f"fold{fold} audit")
        require_sha(train_path, record["train_jsonl_sha256"], name=f"fold{fold} train")
        audit = load_json(audit_path)
        expected_audit = {
            "experiment_id": "623",
            "outer_fold": fold,
            "decision": "GO",
            "frozen_input_hashes_enforced": True,
            "input_sha256": frozen_inputs,
            "train_rows": record["train_rows"],
            "validation_rows": record["validation_rows"],
            "sealed_rows_written": 0,
            "validation_label_columns": [],
            "validation_rationale_columns": [],
        }
        mismatch = {
            key: {"expected": value, "actual": audit.get(key)}
            for key, value in expected_audit.items()
            if audit.get(key) != value
        }
        if mismatch:
            raise ValueError(f"fold{fold} label-isolation audit mismatch: {mismatch}")
        if audit.get("output_sha256", {}).get("train.jsonl") != sha256_file(train_path):
            raise ValueError(f"fold{fold} train output is not bound by its audit")
        rows = _read_jsonl(train_path)
        if len(rows) != record["train_rows"]:
            raise ValueError(f"fold{fold} train row count mismatch")
        for row in rows:
            if int(row.get("label", -1)) not in (0, 1) or not isinstance(
                row.get("rationale"), dict
            ):
                raise ValueError(f"fold{fold} supervised row schema mismatch")
            copies[str(row["id"])].append(row)
        observed_hashes[f"fold{fold}/runtime_audit.json"] = sha256_file(audit_path)
        observed_hashes[f"fold{fold}/train.jsonl"] = sha256_file(train_path)

    if len(copies) != source_manifest["development_rows"]:
        raise ValueError("outer-train union is not the exact development scope")
    result: list[dict[str, Any]] = []
    for item_id, payloads in copies.items():
        canonical = {json.dumps(row, ensure_ascii=False, sort_keys=True) for row in payloads}
        if len(payloads) != 4 or len(canonical) != 1:
            raise ValueError(f"development supervision is not 4x identical: {item_id}")
        result.append(payloads[0])
    result.sort(key=lambda row: int(row["row_index"]))
    if [int(row["row_index"]) for row in result] != list(range(len(result))):
        raise ValueError("development row indices are not complete and ordered")
    if len({str(row["id"]) for row in result}) != len(result):
        raise ValueError("duplicate development ID")
    return result, observed_hashes


def _validate_selector_and_images(
    *,
    source_root: Path,
    source_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[Path, Path, Any]:
    selector_path = (
        source_root
        / f"fold{source_manifest['selector_source_fold']}"
        / "development_selector_oof.npz"
    )
    image_path = (
        source_root
        / f"fold{source_manifest['image_manifest_source_fold']}"
        / "development_image_manifest.tsv.gz"
    )
    require_sha(selector_path, source_manifest["selector_file_sha256"], name="selector")
    require_sha(image_path, source_manifest["image_manifest_file_sha256"], name="image manifest")
    selector = np.load(selector_path, allow_pickle=False)
    required = {"ids", "fold_ids"}
    if not required.issubset(selector.files):
        raise ValueError("selector schema mismatch")
    ids = [str(row["id"]) for row in rows]
    if selector["ids"].astype(str).tolist() != ids:
        raise ValueError("selector IDs/order differ from development runtime")
    if len(selector["fold_ids"]) != len(rows) or set(selector["fold_ids"].astype(int)) != set(
        range(5)
    ):
        raise ValueError("selector fold scope mismatch")
    with gzip.open(image_path, "rt", encoding="utf-8", newline="") as stream:
        image_rows = list(csv.DictReader(stream, delimiter="\t"))
    if [str(row["id"]) for row in image_rows] != ids:
        raise ValueError("image manifest IDs/order differ from development runtime")
    if any(not str(row["image_url"]).strip() for row in image_rows):
        raise ValueError("empty image URL")
    return selector_path, image_path, selector


def build(args: argparse.Namespace) -> dict[str, Any]:
    # The policy gate is intentionally first: no supervised source is touched before it.
    policy = verify_policy(policy_path=args.policy, spec_path=args.spec)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty full-runtime directory")
    source_manifest = load_json(args.source_manifest)
    verify_self_hash(source_manifest, "manifest_sha256")
    if source_manifest.get("sealed_rows") != 0:
        raise ValueError("source manifest admits sealed rows")
    rows, observed_hashes = _load_development_rows(
        source_root=args.fold_runtime_root,
        source_manifest=source_manifest,
    )
    selector_path, image_path, selector = _validate_selector_and_images(
        source_root=args.fold_runtime_root,
        source_manifest=source_manifest,
        rows=rows,
    )
    frame = pd.DataFrame(
        [
            {
                key: row[key]
                for key in ("id", "category", "name", "description", "label")
            }
            for row in rows
        ]
    )
    select_training = _exact_selector(args.parent)
    selected = [int(value) for value in select_training(frame, selector)]
    selected_ids = frame.iloc[selected]["id"].astype(str).tolist()
    multiset_sha = canonical_sha256(sorted(Counter(selected_ids).items()))
    counts = Counter(
        f"{frame.iloc[index]['category']}|{int(frame.iloc[index]['label'])}"
        for index in selected
    )
    batches = math.ceil(len(selected) / 4)
    updates = math.ceil(batches / 4)
    observed = {
        "selected_records": len(selected),
        "selected_unique_rows": len(set(selected_ids)),
        "selected_id_multiset_sha256": multiset_sha,
        "selected_indices_sha256": canonical_sha256(selected),
        "selected_class_occurrences": dict(sorted(counts.items())),
        "microbatches": batches,
        "optimizer_updates": updates,
    }
    expected = {
        key: policy[key]
        for key in (
            "selected_records",
            "selected_unique_rows",
            "selected_id_multiset_sha256",
            "optimizer_updates",
        )
    }
    mismatch = {
        key: {"expected": value, "actual": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if observed["selected_class_occurrences"] != policy["selected_class_occurrences"]:
        mismatch["selected_class_occurrences"] = {
            "expected": policy["selected_class_occurrences"],
            "actual": observed["selected_class_occurrences"],
        }
    if mismatch:
        raise ValueError(f"exact full selector differs from frozen policy: {mismatch}")

    args.output.mkdir(parents=True, exist_ok=True)
    train_path = args.output / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    selector_output = args.output / "development_selector_oof.npz"
    image_output = args.output / "development_image_manifest.tsv.gz"
    shutil.copyfile(selector_path, selector_output)
    shutil.copyfile(image_path, image_output)
    selected_output = args.output / "selected_indices.npy"
    np.save(selected_output, np.asarray(selected, dtype=np.int64), allow_pickle=False)
    report = add_self_hash(
        {
            "schema_version": "exp682_full_runtime_v1",
            "experiment_id": "682",
            "mode": "development_only_FULL_TRAIN_1",
            "decision": "GO",
            "policy_sha256": sha256_file(args.policy),
            "frozen_spec_sha256": sha256_file(args.spec),
            "source_manifest_sha256": sha256_file(args.source_manifest),
            "parent_runner_sha256": sha256_file(args.parent),
            "development_rows": len(rows),
            "development_labels_written": len(rows),
            "sealed_rows_read": 0,
            "sealed_labels_read": 0,
            "sealed_rows_written": 0,
            "public_feedback_used": False,
            **observed,
            "source_file_sha256": observed_hashes,
            "output_sha256": {
                "train.jsonl": sha256_file(train_path),
                "development_selector_oof.npz": sha256_file(selector_output),
                "development_image_manifest.tsv.gz": sha256_file(image_output),
                "selected_indices.npy": sha256_file(selected_output),
            },
        },
        "runtime_sha256",
    )
    (args.output / "runtime_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=here / "frozen_spec.json")
    parser.add_argument("--policy", type=Path, default=here / "post_validation_refit_policy.json")
    parser.add_argument(
        "--source-manifest", type=Path, default=here / "runtime_source_manifest.json"
    )
    parser.add_argument("--fold-runtime-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))

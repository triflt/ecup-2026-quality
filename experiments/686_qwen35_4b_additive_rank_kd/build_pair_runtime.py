from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from typing import Any

EXPERIMENT_ID = "685"
TEACHER_EXPERIMENT_ID = "662"
CONTROL_EXPERIMENT_ID = "680"
FLAMMABLE = "Легковоспламеняющиеся"
KEY_FIELDS = ("global_index", "id", "category", "fold", "occurrence_index")
HARD_NEGATIVES = 4
UNIFORM_NEGATIVES = 4
PAIR_TEMPERATURE = 0.5
NORMAL_CLIP = 2.5
TARGET_MIN = 0.05
TARGET_MAX = 0.95
MAX_ITEM_WEIGHT_SHARE = 0.01
MAX_CLIPPED_TARGET_FRACTION = 0.80
MIN_HARD_PAIR_FRACTION = 0.30
EXPECTED_AGGREGATE_SHA256 = (
    "7995429aedc2b0d90c82ef2ed2f31af113b0614b8b2b8fb730dde4d87d35c0fd"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = jsonl_bytes(rows)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def occurrence_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in KEY_FIELDS)


def tie_aware_ranks(values: list[float]) -> list[float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("rank input must be nonempty and finite")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def normal_ranks(values: list[float]) -> tuple[list[float], list[float]]:
    ranks = tie_aware_ranks(values)
    normal = NormalDist()
    size = len(values)
    transformed = [
        max(
            -NORMAL_CLIP,
            min(NORMAL_CLIP, normal.inv_cdf((rank - 0.5) / size)),
        )
        for rank in ranks
    ]
    return ranks, transformed


def deterministic_order(
    candidates: list[int], *, fold: int, positive_key: tuple[Any, ...], rows: list[dict[str, Any]]
) -> list[int]:
    def digest(index: int) -> bytes:
        payload = json.dumps(
            [fold, list(positive_key), list(occurrence_key(rows[index])), "uniform"],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).digest()

    return sorted(candidates, key=lambda index: (digest(index), occurrence_key(rows[index])))


def build_pair_records(
    rows: list[dict[str, Any]],
    teacher_scores: list[float],
    *,
    fold: int,
    max_endpoint_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != len(teacher_scores):
        raise ValueError("row/teacher-score length mismatch")
    if any(row.get("category") != FLAMMABLE for row in rows):
        raise ValueError("pair builder received a non-flammable row")
    if any(int(row.get("fold", fold)) == fold for row in rows):
        raise ValueError("outer-validation row entered pair generation")
    if max_endpoint_count <= 0:
        raise ValueError("max endpoint count must be positive")
    if any("semantic_component" not in row for row in rows):
        raise ValueError("semantic component is required for pair generation")

    labels = [int(row["label"]) for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("both hard classes are required")
    positives = [index for index, label in enumerate(labels) if label == 1]
    negatives = [index for index, label in enumerate(labels) if label == 0]
    ranks, transformed = normal_ranks(teacher_scores)
    pairs_per_positive = HARD_NEGATIVES + UNIFORM_NEGATIVES
    total_pairs = len(positives) * pairs_per_positive

    id_endpoint_counts: Counter[str] = Counter()
    component_endpoint_counts: Counter[str] = Counter()
    for index in positives:
        id_endpoint_counts[str(rows[index]["id"])] += pairs_per_positive
        component_endpoint_counts[str(rows[index]["semantic_component"])] += pairs_per_positive
    if max(id_endpoint_counts.values()) > max_endpoint_count:
        raise ValueError("positive item alone exceeds the frozen item-weight cap")

    records: list[dict[str, Any]] = []

    def eligible(positive: int, negative: int, selected: set[int]) -> bool:
        if negative in selected:
            return False
        if str(rows[positive]["id"]) == str(rows[negative]["id"]):
            return False
        if str(rows[positive]["semantic_component"]) == str(
            rows[negative]["semantic_component"]
        ):
            return False
        return id_endpoint_counts[str(rows[negative]["id"])] < max_endpoint_count

    def append_pair(positive: int, negative: int, pair_kind: str) -> None:
        raw_target = 1.0 / (
            1.0
            + math.exp(
                -max(
                    -50.0,
                    min(
                        50.0,
                        (transformed[positive] - transformed[negative])
                        / PAIR_TEMPERATURE,
                    ),
                )
            )
        )
        target = max(TARGET_MIN, min(TARGET_MAX, raw_target))
        record = {
            "pair_index": len(records),
            "pair_kind": pair_kind,
            "positive_key": list(occurrence_key(rows[positive])),
            "negative_key": list(occurrence_key(rows[negative])),
            "positive_semantic_component": str(rows[positive]["semantic_component"]),
            "negative_semantic_component": str(rows[negative]["semantic_component"]),
            "positive_teacher_score": float(teacher_scores[positive]),
            "negative_teacher_score": float(teacher_scores[negative]),
            "positive_teacher_rank": float(ranks[positive]),
            "negative_teacher_rank": float(ranks[negative]),
            "positive_normal_rank": float(transformed[positive]),
            "negative_normal_rank": float(transformed[negative]),
            "raw_pair_target": float(raw_target),
            "pair_target": float(target),
        }
        records.append(record)
        id_endpoint_counts[str(rows[negative]["id"])] += 1
        component_endpoint_counts[str(rows[negative]["semantic_component"])] += 1

    positive_order = sorted(positives, key=lambda index: occurrence_key(rows[index]))
    for positive in positive_order:
        selected: set[int] = set()
        nearest = sorted(
            negatives,
            key=lambda negative: (
                abs(ranks[positive] - ranks[negative]),
                occurrence_key(rows[negative]),
            ),
        )
        hard = [negative for negative in nearest if eligible(positive, negative, selected)][
            :HARD_NEGATIVES
        ]
        if len(hard) != HARD_NEGATIVES:
            raise ValueError("item-weight cap prevented hard-negative coverage")
        for negative in hard:
            append_pair(positive, negative, "teacher_hard_balanced")
            selected.add(negative)

        uniform_candidates = deterministic_order(
            negatives,
            fold=fold,
            positive_key=occurrence_key(rows[positive]),
            rows=rows,
        )
        uniform = [
            negative
            for negative in uniform_candidates
            if eligible(positive, negative, selected)
        ][:UNIFORM_NEGATIVES]
        if len(uniform) != UNIFORM_NEGATIVES:
            raise ValueError("item-weight cap prevented uniform-negative coverage")
        for negative in uniform:
            append_pair(positive, negative, "hash_uniform_balanced")
            selected.add(negative)

    if len(records) != total_pairs:
        raise RuntimeError("pair count drifted")
    represented = Counter(tuple(record["positive_key"]) for record in records)
    if set(represented.values()) != {pairs_per_positive} or len(represented) != len(positives):
        raise RuntimeError("positive pair coverage drifted")

    clipped = sum(
        record["pair_target"] in {TARGET_MIN, TARGET_MAX} for record in records
    )
    hard = sum(record["pair_kind"] == "teacher_hard_balanced" for record in records)
    inversions = sum(
        record["positive_teacher_score"] <= record["negative_teacher_score"]
        for record in records
    )
    max_id_share = max(id_endpoint_counts.values()) / (2 * total_pairs)
    max_component_share = max(component_endpoint_counts.values()) / (2 * total_pairs)
    summary = {
        "rows": len(rows),
        "positive_occurrences": len(positives),
        "negative_occurrences": len(negatives),
        "pairs": len(records),
        "pairs_per_positive": pairs_per_positive,
        "teacher_hard_pairs": hard,
        "teacher_hard_fraction": hard / len(records),
        "clipped_pair_targets": clipped,
        "clipped_pair_target_fraction": clipped / len(records),
        "ambiguous_pair_target_fraction": sum(
            0.1 < record["pair_target"] < 0.9 for record in records
        )
        / len(records),
        "teacher_pair_inversions": inversions,
        "teacher_pair_inversion_fraction": inversions / len(records),
        "max_item_endpoint_count": max(id_endpoint_counts.values()),
        "max_component_endpoint_count": max(component_endpoint_counts.values()),
        "max_item_weight_share": max_id_share,
        "max_component_weight_share": max_component_share,
        "item_endpoint_cap": max_endpoint_count,
    }
    return records, summary


def build(
    source_runtime: Path,
    teacher_runtime: Path,
    teacher_artifact: Path,
    teacher_aggregate: Path,
    output: Path,
    fold: int,
) -> dict[str, Any]:
    if fold not in range(5):
        raise ValueError("fold must be 0..4")
    if output.exists():
        raise FileExistsError("refusing to replace an existing pair runtime")
    if sha256_file(teacher_aggregate) != EXPECTED_AGGREGATE_SHA256:
        raise ValueError("teacher aggregate SHA mismatch")
    aggregate = json.loads(teacher_aggregate.read_text(encoding="utf-8"))
    if (
        aggregate.get("decision") != "ACCEPT_FULL_TARGET_SET"
        or aggregate.get("folds") != [0, 1, 2, 3, 4]
        or int(aggregate.get("labels_read", -1)) != 0
        or int(aggregate.get("validation_labels_read", -1)) != 0
        or int(aggregate.get("sealed_rows", -1)) != 0
        or aggregate.get("public_used") is not False
    ):
        raise ValueError("teacher aggregate is not accepted and label-free")

    root = Path(__file__).resolve().parents[2]
    control_builder = load_module(
        root / "experiments/680_qwen35_4b_flammable_only_hard_bce/build_runtime.py",
        "exp680_runtime_for_685",
    )
    teacher_builder = load_module(
        root / "experiments/681_qwen35_4b_flammable_logit_distillation/build_runtime.py",
        "exp681_runtime_for_685",
    )
    with tempfile.TemporaryDirectory(prefix=f"exp685_base_f{fold}_") as directory:
        filtered = Path(directory) / "runtime"
        base_audit = control_builder.build(source_runtime, filtered, fold)
        train_rows = read_jsonl(filtered / "train.jsonl")
        validation_rows = read_jsonl(filtered / "validation.jsonl")

    teacher_rows, teacher_manifest = teacher_builder.load_teacher(
        teacher_artifact, teacher_runtime, fold
    )
    if teacher_manifest["archive_sha256"] != aggregate["artifact_sha256"][str(fold)]:
        raise ValueError("teacher artifact differs from accepted aggregate")
    if (
        teacher_manifest["runtime_contract_sha256"]
        != aggregate["runtime_contract_sha256"][str(fold)]
    ):
        raise ValueError("teacher runtime differs from accepted aggregate")
    if (
        teacher_manifest["teacher_scores_sha256"]
        != aggregate["teacher_scores_sha256"][str(fold)]
    ):
        raise ValueError("teacher score payload differs from accepted aggregate")
    teacher_rows = [row for row in teacher_rows if row["category"] == FLAMMABLE]
    if [occurrence_key(row) for row in train_rows] != [
        occurrence_key(row) for row in teacher_rows
    ]:
        raise ValueError("teacher scores do not bind to filtered student occurrences")

    teacher_scores = [float(row["score"]) for row in teacher_rows]
    total_pairs = sum(int(row["label"]) == 1 for row in train_rows) * (
        HARD_NEGATIVES + UNIFORM_NEGATIVES
    )
    max_endpoint_count = math.floor(2 * total_pairs * MAX_ITEM_WEIGHT_SHARE)
    pairs, pair_summary = build_pair_records(
        train_rows,
        teacher_scores,
        fold=fold,
        max_endpoint_count=max_endpoint_count,
    )
    if pair_summary["teacher_hard_fraction"] < MIN_HARD_PAIR_FRACTION:
        raise ValueError("teacher-hard pair fraction is below the frozen gate")
    if pair_summary["clipped_pair_target_fraction"] > MAX_CLIPPED_TARGET_FRACTION:
        raise ValueError("rank targets collapse at clipping bounds")
    if pair_summary["max_item_weight_share"] > MAX_ITEM_WEIGHT_SHARE:
        raise ValueError("one item exceeds one percent of total pair weight")

    ranks, transformed = normal_ranks(teacher_scores)
    train_targets = []
    for row, score, rank, transformed_rank in zip(
        train_rows, teacher_scores, ranks, transformed, strict=True
    ):
        train_targets.append(
            {
                **{field: row[field] for field in KEY_FIELDS},
                "semantic_component": str(row["semantic_component"]),
                "label": int(row["label"]),
                "teacher_score": score,
                "teacher_rank": rank,
                "teacher_normal_rank": transformed_rank,
            }
        )

    output.mkdir(parents=True)
    train_targets_payload = jsonl_bytes(train_targets)
    pairs_payload = jsonl_bytes(pairs)
    (output / "train_targets.jsonl").write_bytes(train_targets_payload)
    (output / "pairs.jsonl").write_bytes(pairs_payload)
    train_targets_sha256 = sha256_bytes(train_targets_payload)
    pairs_sha256 = sha256_bytes(pairs_payload)
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": "685A_R0",
        "outer_fold": fold,
        "control_experiment_id": CONTROL_EXPERIMENT_ID,
        "teacher_experiment_id": TEACHER_EXPERIMENT_ID,
        "source_641_runtime_contract_sha256": base_audit[
            "source_runtime_contract_sha256"
        ],
        "derived_680_runtime_contract_sha256": base_audit["contract_sha256"],
        "teacher_aggregate_sha256": EXPECTED_AGGREGATE_SHA256,
        "teacher_manifest": teacher_manifest,
        "teacher_target_scope": "outer_train_in_sample_outer_validation_unread",
        "ordinary_oof_merge_used": False,
        "derived_680_output_sha256": base_audit["output_sha256"],
        "source_validation_rows": len(validation_rows),
        "pair_sampler": {
            "hard_negatives_per_positive": HARD_NEGATIVES,
            "uniform_negatives_per_positive": UNIFORM_NEGATIVES,
            "hard_sampler": "nearest_teacher_rank_subject_to_frozen_item_cap_and_same_component_exclusion",
            "uniform_sampler": "sha256_order_subject_to_frozen_item_cap_and_same_component_exclusion",
            "same_id_pairs": 0,
            "same_semantic_component_pairs": 0,
            "max_item_weight_share": MAX_ITEM_WEIGHT_SHARE,
            "semantic_component_weight_cap": None,
        },
        "rank_target": {
            "tie_handling": "average_rank",
            "normal_transform": "normal_ppf((rank-0.5)/n)",
            "normal_clip": NORMAL_CLIP,
            "temperature": PAIR_TEMPERATURE,
            "target_min": TARGET_MIN,
            "target_max": TARGET_MAX,
        },
        "pair_summary": pair_summary,
        "validation_labels_written": 0,
        "outer_validation_teacher_overlap": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "output_sha256": {
            "train_targets.jsonl": train_targets_sha256,
            "pairs.jsonl": pairs_sha256,
        },
        "decision": "GO_TECHNICAL_SMOKE",
    }
    report["contract_sha256"] = canonical_sha256(report)
    audit_payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    (output / "runtime_audit.json").write_bytes(audit_payload)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-runtime", type=Path, required=True)
    result.add_argument("--teacher-runtime", type=Path, required=True)
    result.add_argument("--teacher-artifact", type=Path, required=True)
    result.add_argument("--teacher-aggregate", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            build(
                args.source_runtime,
                args.teacher_runtime,
                args.teacher_artifact,
                args.teacher_aggregate,
                args.output,
                args.fold,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

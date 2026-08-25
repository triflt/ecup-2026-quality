from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from build_pair_runtime import (
    EXPERIMENT_ID,
    FLAMMABLE,
    HARD_NEGATIVES,
    KEY_FIELDS,
    MAX_CLIPPED_TARGET_FRACTION,
    MAX_ITEM_WEIGHT_SHARE,
    MIN_HARD_PAIR_FRACTION,
    PAIR_TEMPERATURE,
    TARGET_MAX,
    TARGET_MIN,
    UNIFORM_NEGATIVES,
    canonical_sha256,
    occurrence_key,
    sha256_bytes,
)


def read_jsonl_bytes(payload: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in payload.decode().splitlines()]


def verify_payloads(
    audit_payload: bytes, train_payload: bytes, pairs_payload: bytes
) -> dict[str, Any]:
    audit = json.loads(audit_payload)
    body = dict(audit)
    digest = body.pop("contract_sha256", None)
    if digest != canonical_sha256(body):
        raise ValueError("runtime audit self-hash mismatch")
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "685A_R0",
        "decision": "GO_TECHNICAL_SMOKE",
        "ordinary_oof_merge_used": False,
        "outer_validation_teacher_overlap": 0,
        "validation_labels_written": 0,
        "sealed_rows_used": 0,
        "public_used": False,
    }
    mismatches = {
        key: {"expected": value, "actual": audit.get(key)}
        for key, value in expected.items()
        if audit.get(key) != value
    }
    if mismatches:
        raise ValueError(f"runtime contract mismatch: {mismatches}")
    if audit.get("output_sha256") != {
        "train_targets.jsonl": sha256_bytes(train_payload),
        "pairs.jsonl": sha256_bytes(pairs_payload),
    }:
        raise ValueError("runtime payload SHA mismatch")

    train = read_jsonl_bytes(train_payload)
    pairs = read_jsonl_bytes(pairs_payload)
    expected_target_fields = {
        *KEY_FIELDS,
        "semantic_component",
        "label",
        "teacher_score",
        "teacher_rank",
        "teacher_normal_rank",
    }
    if any(set(row) != expected_target_fields for row in train):
        raise ValueError("sanitized train-target schema mismatch")
    if any(row.get("category") != FLAMMABLE for row in train):
        raise ValueError("train runtime is not flammable-only")
    if any(int(row["fold"]) == int(audit["outer_fold"]) for row in train):
        raise ValueError("outer-validation row entered train runtime")
    if set(audit.get("derived_680_output_sha256", {})) != {
        "train.jsonl",
        "validation.jsonl",
    }:
        raise ValueError("source runtime payload binding is incomplete")
    lookup = {occurrence_key(row): row for row in train}
    if len(lookup) != len(train):
        raise ValueError("duplicate occurrence key in train runtime")

    expected_pair_fields = {
        "pair_index",
        "pair_kind",
        "positive_key",
        "negative_key",
        "positive_semantic_component",
        "negative_semantic_component",
        "positive_teacher_score",
        "negative_teacher_score",
        "positive_teacher_rank",
        "negative_teacher_rank",
        "positive_normal_rank",
        "negative_normal_rank",
        "raw_pair_target",
        "pair_target",
    }
    positive_counts: Counter[tuple[Any, ...]] = Counter()
    kind_counts: Counter[tuple[Any, ...]] = Counter()
    id_endpoints: Counter[str] = Counter()
    component_endpoints: Counter[str] = Counter()
    clipped = 0
    inversions = 0
    for index, pair in enumerate(pairs):
        if set(pair) != expected_pair_fields or int(pair["pair_index"]) != index:
            raise ValueError("pair schema or ordering mismatch")
        positive_key = tuple(pair["positive_key"])
        negative_key = tuple(pair["negative_key"])
        if positive_key not in lookup or negative_key not in lookup:
            raise ValueError("pair key is absent from train runtime")
        positive = lookup[positive_key]
        negative = lookup[negative_key]
        if int(positive["label"]) != 1 or int(negative["label"]) != 0:
            raise ValueError("pair hard labels are reversed")
        if str(positive["id"]) == str(negative["id"]):
            raise ValueError("same product id appears on both sides of a pair")
        if str(positive["semantic_component"]) == str(
            negative["semantic_component"]
        ):
            raise ValueError("same semantic component appears on both sides of a pair")
        if pair["positive_semantic_component"] != str(positive["semantic_component"]):
            raise ValueError("positive semantic component binding mismatch")
        if pair["negative_semantic_component"] != str(negative["semantic_component"]):
            raise ValueError("negative semantic component binding mismatch")
        if float(pair["positive_teacher_score"]) != float(positive["teacher_score"]):
            raise ValueError("positive teacher-score binding mismatch")
        if float(pair["negative_teacher_score"]) != float(negative["teacher_score"]):
            raise ValueError("negative teacher-score binding mismatch")
        raw = 1.0 / (
            1.0
            + math.exp(
                -max(
                    -50.0,
                    min(
                        50.0,
                        (
                            float(pair["positive_normal_rank"])
                            - float(pair["negative_normal_rank"])
                        )
                        / PAIR_TEMPERATURE,
                    ),
                )
            )
        )
        target = max(TARGET_MIN, min(TARGET_MAX, raw))
        if not math.isclose(raw, float(pair["raw_pair_target"]), abs_tol=1e-15):
            raise ValueError("raw pair target mismatch")
        if not math.isclose(target, float(pair["pair_target"]), abs_tol=1e-15):
            raise ValueError("clipped pair target mismatch")
        if target in {TARGET_MIN, TARGET_MAX}:
            clipped += 1
        if float(pair["positive_teacher_score"]) <= float(
            pair["negative_teacher_score"]
        ):
            inversions += 1
        positive_counts[positive_key] += 1
        kind_counts[(positive_key, str(pair["pair_kind"]))] += 1
        for row in (positive, negative):
            id_endpoints[str(row["id"])] += 1
            component_endpoints[str(row["semantic_component"])] += 1

    pairs_per_positive = HARD_NEGATIVES + UNIFORM_NEGATIVES
    positives = [row for row in train if int(row["label"]) == 1]
    if len(positive_counts) != len(positives) or set(positive_counts.values()) != {
        pairs_per_positive
    }:
        raise ValueError("positive pair coverage mismatch")
    for key in positive_counts:
        if kind_counts[(key, "teacher_hard_balanced")] != HARD_NEGATIVES:
            raise ValueError("hard-negative count mismatch")
        if kind_counts[(key, "hash_uniform_balanced")] != UNIFORM_NEGATIVES:
            raise ValueError("uniform-negative count mismatch")
    hard_fraction = sum(
        pair["pair_kind"] == "teacher_hard_balanced" for pair in pairs
    ) / len(pairs)
    clipped_fraction = clipped / len(pairs)
    max_item_share = max(id_endpoints.values()) / (2 * len(pairs))
    if hard_fraction < MIN_HARD_PAIR_FRACTION:
        raise ValueError("teacher-hard fraction is below gate")
    if clipped_fraction > MAX_CLIPPED_TARGET_FRACTION:
        raise ValueError("pair targets collapsed at clipping bounds")
    if max_item_share > MAX_ITEM_WEIGHT_SHARE:
        raise ValueError("one product exceeds the frozen pair-weight cap")

    recomputed = {
        "rows": len(train),
        "positive_occurrences": len(positives),
        "negative_occurrences": len(train) - len(positives),
        "pairs": len(pairs),
        "pairs_per_positive": pairs_per_positive,
        "teacher_hard_pairs": int(hard_fraction * len(pairs)),
        "teacher_hard_fraction": hard_fraction,
        "clipped_pair_targets": clipped,
        "clipped_pair_target_fraction": clipped_fraction,
        "ambiguous_pair_target_fraction": sum(
            0.1 < float(pair["pair_target"]) < 0.9 for pair in pairs
        )
        / len(pairs),
        "teacher_pair_inversions": inversions,
        "teacher_pair_inversion_fraction": inversions / len(pairs),
        "max_item_endpoint_count": max(id_endpoints.values()),
        "max_component_endpoint_count": max(component_endpoints.values()),
        "max_item_weight_share": max_item_share,
        "max_component_weight_share": max(component_endpoints.values())
        / (2 * len(pairs)),
        "item_endpoint_cap": math.floor(
            2 * len(pairs) * MAX_ITEM_WEIGHT_SHARE
        ),
    }
    if audit.get("pair_summary") != recomputed:
        raise ValueError("pair summary does not match payload")
    acceptance = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": "685A_R0",
        "outer_fold": int(audit["outer_fold"]),
        "runtime_contract_sha256": digest,
        "output_sha256": audit["output_sha256"],
        "pair_summary": recomputed,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "decision": "ACCEPT_R0_OPEN_TECHNICAL_SMOKE",
    }
    acceptance["acceptance_sha256"] = canonical_sha256(acceptance)
    return acceptance


def verify(runtime: Path) -> dict[str, Any]:
    audit_path = runtime / "runtime_audit.json"
    train_path = runtime / "train_targets.jsonl"
    pairs_path = runtime / "pairs.jsonl"
    for path in (audit_path, train_path, pairs_path):
        if not path.is_file():
            raise FileNotFoundError(f"runtime payload is missing: {path.name}")
    return verify_payloads(
        audit_path.read_bytes(), train_path.read_bytes(), pairs_path.read_bytes()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.runtime)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError("refusing to replace an existing acceptance report")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from contract import (
    ALLOWED_STAGES,
    BLIND_AUDIT_SIZE,
    DRAFT_VERSION,
    FORBIDDEN_STAGES,
    GRAPH_VERSION,
    MAX_KEY_DEGREE,
    MIN_ELIGIBLE_COMPONENTS,
    MIN_ELIGIBLE_PAIRS,
    MIN_KAPPA,
    MIN_SAME_PRODUCT_PRECISION,
    RECOMBINATION_FRACTION_OF_REPEATS,
    SEED,
)


@dataclass(frozen=True)
class StrongPair:
    left_id: str
    right_id: str
    semantic_component: str
    category: str
    label: int
    stage: str
    key_hash: str
    key_degree: int
    corroboration: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _strict_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"invalid strict boolean {value!r}")


def load_frozen_provenance(
    sealed_dir: Path, draft_dir: Path, data: pd.DataFrame
) -> tuple[pd.DataFrame, list[StrongPair], dict[str, Any]]:
    sealed_manifest_path = sealed_dir / "manifest.json"
    sealed_manifest = json.loads(sealed_manifest_path.read_text(encoding="utf-8"))
    if (
        sealed_manifest.get("version") != GRAPH_VERSION
        or sealed_manifest.get("sealed") is not True
        or sealed_manifest.get("valid_for_candidate_scoring") is not True
    ):
        raise ValueError("semantic-family v3 is not a sealed scoring graph")
    folds_path = sealed_dir / "folds.csv"
    if file_sha256(folds_path) != sealed_manifest["output_sha256"]["folds"]:
        raise ValueError("sealed folds checksum mismatch")
    draft_manifest_path = draft_dir / "manifest.json"
    draft_manifest = json.loads(draft_manifest_path.read_text(encoding="utf-8"))
    if draft_manifest.get("graph_version") != DRAFT_VERSION:
        raise ValueError("draft provenance graph version mismatch")
    edges_path = draft_dir / "edges.csv"
    if file_sha256(edges_path) != draft_manifest["output_sha256"]["edges"]:
        raise ValueError("draft edge provenance checksum mismatch")
    sealed = pd.read_csv(folds_path, dtype={"id": str, "semantic_component": str})
    if len(sealed) != len(data) or set(sealed.id) != set(data.id.astype(str)):
        raise ValueError("sealed graph row membership differs from training data")
    aligned = sealed.set_index("id").loc[data.id.astype(str)].reset_index()
    if not np.array_equal(aligned.category.astype(str), data.category.astype(str)):
        raise ValueError("sealed category mismatch")
    if not np.array_equal(aligned.label.to_numpy(np.int8), data.label.to_numpy(np.int8)):
        raise ValueError("sealed label mismatch")
    mixed = set(
        aligned.groupby("semantic_component").label.nunique().loc[lambda values: values > 1].index
    )
    meta = aligned.set_index("id")
    precedence = {"exact_full_text": 0, "exact_first_image": 1, "exact_name_corroborated": 2}
    selected: dict[tuple[str, str], StrongPair] = {}
    with edges_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            stage = row["stage"]
            if stage not in ALLOWED_STAGES:
                continue
            if stage in FORBIDDEN_STAGES:
                raise ValueError("forbidden stage entered the allowed-stage contract")
            if not all(
                _strict_bool(row[key])
                for key in ("informative_guard", "degree_guard", "label_blind")
            ):
                continue
            degree = int(row["key_degree"])
            if degree > MAX_KEY_DEGREE:
                continue
            if "generic" in str(row["corroboration"]).lower():
                continue
            left, right = str(row["left_id"]), str(row["right_id"])
            if left not in meta.index or right not in meta.index:
                raise ValueError("draft edge references an unknown sealed row")
            left_meta, right_meta = meta.loc[left], meta.loc[right]
            component = str(left_meta.semantic_component)
            if component != str(right_meta.semantic_component) or component in mixed:
                continue
            if left_meta.category != right_meta.category or int(left_meta.label) != int(
                right_meta.label
            ):
                continue
            if left_meta["split"] == "sealed_holdout" or right_meta["split"] == "sealed_holdout":
                continue
            key = tuple(sorted((left, right)))
            item = StrongPair(
                left_id=key[0],
                right_id=key[1],
                semantic_component=component,
                category=str(left_meta.category),
                label=int(left_meta.label),
                stage=stage,
                key_hash=str(row["key_hash"]),
                key_degree=degree,
                corroboration=str(row["corroboration"]),
            )
            previous = selected.get(key)
            if previous is None or precedence[item.stage] < precedence[previous.stage]:
                selected[key] = item
    pairs = sorted(selected.values(), key=lambda item: (item.left_id, item.right_id))
    provenance = {
        "sealed_manifest_sha256": file_sha256(sealed_manifest_path),
        "sealed_folds_sha256": file_sha256(folds_path),
        "draft_manifest_sha256": file_sha256(draft_manifest_path),
        "draft_edges_sha256": file_sha256(edges_path),
        "strong_pairs": len(pairs),
        "mixed_label_components_excluded": len(mixed),
        "allowed_stages": sorted(ALLOWED_STAGES),
        "forbidden_stages": sorted(FORBIDDEN_STAGES),
        "maximum_key_degree": MAX_KEY_DEGREE,
    }
    return aligned, pairs, provenance


def eligible_outer_pairs(
    pairs: list[StrongPair], fold_by_id: dict[str, int], holdout_fold: int
) -> list[StrongPair]:
    return [
        pair
        for pair in pairs
        if fold_by_id[pair.left_id] != holdout_fold and fold_by_id[pair.right_id] != holdout_fold
    ]


def deterministic_blind_sample(pairs: list[StrongPair]) -> list[StrongPair]:
    unique = {(pair.left_id, pair.right_id): pair for pair in pairs}
    if len(unique) < BLIND_AUDIT_SIZE:
        raise ValueError("fewer than 300 unique strong pairs are available for blind audit")
    return sorted(
        unique.values(),
        key=lambda pair: hashlib.sha256(
            f"exp590-blind-v1\0{pair.left_id}\0{pair.right_id}".encode()
        ).hexdigest(),
    )[:BLIND_AUDIT_SIZE]


def evaluate_reviews(sample: list[StrongPair], reviews_path: Path | None) -> dict[str, Any]:
    base = {
        "required_pairs": BLIND_AUDIT_SIZE,
        "reviewed_pairs": 0,
        "same_product_precision": None,
        "required_same_product_precision": MIN_SAME_PRODUCT_PRECISION,
        "cohen_kappa": None,
        "required_cohen_kappa_if_defined": MIN_KAPPA,
        "kappa_defined": False,
        "raw_agreement": None,
        "decision": "NO_GO",
        "reason": "two blinded reviewer decisions are not available",
    }
    if reviews_path is None or not reviews_path.is_file():
        return base
    expected = {f"R{index:03d}": pair for index, pair in enumerate(sample, 1)}
    reviews = pd.read_csv(reviews_path, dtype={"audit_id": str, "left_id": str, "right_id": str})
    required = {
        "audit_id",
        "left_id",
        "right_id",
        "reviewer_a_same_product",
        "reviewer_b_same_product",
    }
    if not required.issubset(reviews.columns) or len(reviews) != BLIND_AUDIT_SIZE:
        raise ValueError("blind review schema/count mismatch")
    if set(reviews.audit_id) != set(expected):
        raise ValueError("blind review audit IDs mismatch")
    values_a, values_b = [], []
    for row in reviews.itertuples(index=False):
        pair = expected[row.audit_id]
        if str(row.left_id) != pair.left_id or str(row.right_id) != pair.right_id:
            raise ValueError("blind review pair identity mismatch")
        a, b = int(row.reviewer_a_same_product), int(row.reviewer_b_same_product)
        if a not in (0, 1) or b not in (0, 1):
            raise ValueError("blind review decisions must be atomic 0/1")
        values_a.append(a)
        values_b.append(b)
    array_a, array_b = np.asarray(values_a), np.asarray(values_b)
    agreement = float(np.mean(array_a == array_b))
    strict_positive = (array_a == 1) & (array_b == 1)
    precision = float(strict_positive.mean())
    observed = agreement
    expected_agreement = float(
        np.mean(array_a) * np.mean(array_b) + (1 - np.mean(array_a)) * (1 - np.mean(array_b))
    )
    kappa_defined = expected_agreement < 1.0
    kappa = (observed - expected_agreement) / (1 - expected_agreement) if kappa_defined else None
    kappa_pass = kappa >= MIN_KAPPA if kappa_defined else agreement == 1.0
    passed = precision >= MIN_SAME_PRODUCT_PRECISION and kappa_pass
    return {
        **base,
        "reviewed_pairs": BLIND_AUDIT_SIZE,
        "same_product_precision": precision,
        "cohen_kappa": float(kappa) if kappa is not None else None,
        "kappa_defined": kappa_defined,
        "raw_agreement": agreement,
        "decision": "GO" if passed else "NO_GO",
        "reason": None if passed else "blind precision/agreement gate failed",
    }


def build_fold_plan(
    frame: pd.DataFrame,
    oof,
    records: list[int],
    pairs: list[StrongPair],
    *,
    holdout_fold: int,
    review_audit: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids = frame.id.astype(str).to_numpy()
    id_to_index = {item_id: index for index, item_id in enumerate(ids)}
    fold_by_id = dict(zip(ids, oof["fold_ids"].astype(np.int8), strict=True))
    eligible = eligible_outer_pairs(pairs, fold_by_id, holdout_fold)
    components = {pair.semantic_component for pair in eligible}
    counts = Counter(records)
    parent_members = set(counts)
    neighbors: dict[int, list[tuple[int, StrongPair]]] = defaultdict(list)
    for pair in eligible:
        left, right = id_to_index[pair.left_id], id_to_index[pair.right_id]
        if left in parent_members and right in parent_members:
            neighbors[left].append((right, pair))
            neighbors[right].append((left, pair))
    epoch_records = list(records)
    random.Random(SEED).shuffle(epoch_records)
    positions: dict[int, list[int]] = defaultdict(list)
    for position, index in enumerate(epoch_records):
        positions[index].append(position)
    manifest = []
    for source in sorted(neighbors, key=lambda index: ids[index]):
        occurrences = positions[source]
        repeated = occurrences[1:]
        if not repeated:
            continue
        exact_count = len(repeated) * RECOMBINATION_FRACTION_OF_REPEATS
        if not exact_count.is_integer():
            raise ValueError("eligible repeated occurrence count cannot realize exact 25%")
        donor, pair = min(
            neighbors[source],
            key=lambda item: hashlib.sha256(
                f"exp590-donor-v1\0{holdout_fold}\0{ids[source]}\0{ids[item[0]]}".encode()
            ).hexdigest(),
        )
        chosen = sorted(
            repeated,
            key=lambda position: hashlib.sha256(
                f"exp590-occ-v1\0{holdout_fold}\0{ids[source]}\0{position}".encode()
            ).hexdigest(),
        )[: int(exact_count)]
        for position in sorted(chosen):
            manifest.append(
                {
                    "occurrence_position": position,
                    "source_index": source,
                    "source_id": ids[source],
                    "donor_index": donor,
                    "donor_id": ids[donor],
                    **asdict(pair),
                }
            )
    repeated_eligible = sum(
        len(positions[index]) - 1 for index in neighbors if len(positions[index]) > 1
    )
    failures = []
    if len(eligible) < MIN_ELIGIBLE_PAIRS:
        failures.append("insufficient_eligible_pairs")
    if len(components) < MIN_ELIGIBLE_COMPONENTS:
        failures.append("insufficient_eligible_components")
    if review_audit["decision"] != "GO":
        failures.append("blind_audit_not_approved")
    if (
        repeated_eligible == 0
        or len(manifest) / repeated_eligible != RECOMBINATION_FRACTION_OF_REPEATS
    ):
        failures.append("recombination_fraction_not_exact")
    audit = {
        "experiment_id": "590",
        "holdout_fold": holdout_fold,
        "parent_training_records": len(records),
        "parent_record_multiset_sha256": canonical_sha256(sorted(counts.items())),
        "candidate_record_multiset_sha256": canonical_sha256(sorted(Counter(records).items())),
        "record_multiset_unchanged": True,
        "record_order_unchanged": True,
        "steps_unchanged": True,
        "text_prompt_labels_weights_unchanged": True,
        "eligible_pairs": len(eligible),
        "eligible_components": len(components),
        "eligible_repeated_occurrences": repeated_eligible,
        "recombined_occurrences": len(manifest),
        "recombined_fraction_of_repeats": len(manifest) / repeated_eligible
        if repeated_eligible
        else 0.0,
        "sealed_holdout_pairs": 0,
        "outer_validation_pairs": 0,
        "mixed_label_component_pairs": 0,
        "forbidden_stage_pairs": 0,
        "generic_provenance_pairs": 0,
        "high_degree_pairs": 0,
        "cross_category_pairs": 0,
        "cross_label_pairs": 0,
        "manifest_sha256": canonical_sha256(manifest),
        "blind_audit": review_audit,
        "failures": failures,
        "decision": "GO" if not failures else "NO_GO",
    }
    return manifest, audit

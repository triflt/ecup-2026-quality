from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP530_ANALYSIS = ROOT / "experiments/530_internvl_pairwise_flammable_ranker/analysis"
EXP530_AUDIT = EXP530_ANALYSIS / "pair_manifest_audit.json"
FLAMMABLE = "Легковоспламеняющиеся"
PAIRING_VERSION = "qwen35_exp530_parent_intersection_v1"
MAX_PAIRS_PER_POSITIVE = 4
MAX_NEGATIVE_REUSE = 8
MIN_REALIZED_PAIRS = 64
MIN_PAIR_BATCHES = 64


@dataclass(frozen=True)
class PairRow:
    batch_index: int
    positive_slot: int
    negative_slot: int
    positive_index: int
    positive_id: str
    negative_index: int
    negative_id: str
    exp530_pair_index: int
    positive_cue_mask: str
    negative_cue_mask: str
    selection_scope: str
    cosine_similarity: float


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"expected a strict boolean value, received {value!r}")


def load_frozen_exp530(fold: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if fold not in (0, 3):
        raise ValueError("only folds 0 and 3 are frozen")
    audit = json.loads(EXP530_AUDIT.read_text(encoding="utf-8"))
    if audit.get("selector_label_blind") is not True or audit.get("selector_rows") != 909:
        raise ValueError("exp530 selector contract mismatch")
    membership_path = EXP530_ANALYSIS / "selector_membership.csv"
    pairs_path = EXP530_ANALYSIS / f"pair_manifest_fold_{fold}.csv"
    if file_sha256(membership_path) != audit["output_sha256"]["selector_membership"]:
        raise ValueError("exp530 selector membership checksum mismatch")
    if file_sha256(pairs_path) != audit["output_sha256"][f"pair_manifest_fold_{fold}"]:
        raise ValueError("exp530 pair manifest checksum mismatch")
    membership = pd.read_csv(
        membership_path, dtype={"id": str, "connected_component": str, "cue_mask": str}
    )
    pairs = pd.read_csv(
        pairs_path,
        dtype={
            "positive_id": str,
            "negative_id": str,
            "positive_cue_mask": str,
            "negative_cue_mask": str,
        },
    )
    return membership, pairs, audit


def build_pair_plan(
    frame: pd.DataFrame,
    fold_ids: np.ndarray,
    training_records: list[int],
    *,
    holdout_fold: int,
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    membership, frozen_pairs, exp530_audit = load_frozen_exp530(holdout_fold)
    id_to_index = {str(item_id): index for index, item_id in enumerate(frame.id.astype(str))}
    if len(id_to_index) != len(frame):
        raise ValueError("data ids are not unique")
    counts = Counter(int(index) for index in training_records)
    remaining = counts.copy()
    labels = frame["label"].to_numpy(np.int8)
    categories = frame["category"].astype(str).to_numpy()
    member = membership.set_index("id")
    candidates = []
    for row in frozen_pairs.itertuples(index=False):
        positive = id_to_index.get(str(row.positive_id))
        negative = id_to_index.get(str(row.negative_id))
        if positive is None or negative is None or positive not in counts or negative not in counts:
            continue
        if categories[positive] != FLAMMABLE or categories[negative] != FLAMMABLE:
            raise ValueError("exp530 pair contains another category")
        if labels[positive] != 1 or labels[negative] != 0:
            raise ValueError("exp530 donor labels differ from runtime training labels")
        if int(fold_ids[positive]) == holdout_fold or int(fold_ids[negative]) == holdout_fold:
            raise ValueError("outer-validation row survived parent intersection")
        if not _strict_bool(member.loc[str(row.positive_id), "safe_for_selection"]):
            raise ValueError("unsafe positive donor in frozen topology")
        if not _strict_bool(member.loc[str(row.negative_id), "safe_for_selection"]):
            raise ValueError("unsafe negative donor in frozen topology")
        candidates.append((positive, negative, row))

    by_positive: dict[int, deque[tuple[int, Any]]] = defaultdict(deque)
    for positive, negative, row in candidates:
        by_positive[positive].append((negative, row))
    positive_order = sorted(
        by_positive,
        key=lambda index: hashlib.sha256(str(frame.iloc[index]["id"]).encode()).hexdigest(),
    )
    selected_edges = []
    positive_reuse: Counter[int] = Counter()
    negative_reuse: Counter[int] = Counter()
    for positive in positive_order:
        for negative, row in by_positive[positive]:
            positive_cap = min(MAX_PAIRS_PER_POSITIVE, counts[positive])
            negative_cap = min(MAX_NEGATIVE_REUSE, counts[negative])
            if positive_reuse[positive] >= positive_cap:
                break
            if negative_reuse[negative] >= negative_cap:
                continue
            selected_edges.append((positive, negative, row))
            positive_reuse[positive] += 1
            negative_reuse[negative] += 1

    batches: list[list[int]] = []
    manifest: list[PairRow] = []
    pending_pair_occurrences = Counter(
        index for positive, negative, _ in selected_edges for index in (positive, negative)
    )
    cursor = 0
    while cursor < len(selected_edges):
        positive, negative, row = selected_edges[cursor]
        cursor += 1
        if remaining[positive] < 1 or remaining[negative] < 1:
            raise RuntimeError("selected pair exceeds parent occurrence multiplicity")
        local = [(positive, negative, row)]
        pending_pair_occurrences[positive] -= 1
        pending_pair_occurrences[negative] -= 1
        nodes = [value for positive, negative, _ in local for value in (positive, negative)]
        if len(nodes) < 4:
            fillers = [
                index
                for index in training_records
                if remaining[index] - nodes.count(index) > pending_pair_occurrences[index]
                and index not in nodes
            ]
            nodes.extend(fillers[: 4 - len(nodes)])
        if len(nodes) != 4:
            break
        batch_index = len(batches)
        batches.append(nodes)
        for index in nodes:
            remaining[index] -= 1
        for positive, negative, row in local:
            manifest.append(
                PairRow(
                    batch_index=batch_index,
                    positive_slot=nodes.index(positive),
                    negative_slot=nodes.index(negative),
                    positive_index=positive,
                    positive_id=str(frame.iloc[positive]["id"]),
                    negative_index=negative,
                    negative_id=str(frame.iloc[negative]["id"]),
                    exp530_pair_index=int(row.pair_index),
                    positive_cue_mask=str(row.positive_cue_mask),
                    negative_cue_mask=str(row.negative_cue_mask),
                    selection_scope=str(row.selection_scope),
                    cosine_similarity=float(row.cosine_similarity),
                )
            )

    ordered = [index for batch in batches for index in batch]
    for index in training_records:
        if remaining[index] > 0:
            ordered.append(index)
            remaining[index] -= 1
    if any(remaining.values()):
        raise RuntimeError("failed to place every parent record occurrence")
    manifest_rows = [asdict(row) for row in manifest]
    pos_reuse = Counter(row.positive_index for row in manifest)
    neg_reuse = Counter(row.negative_index for row in manifest)
    input_multiset = canonical_sha256(sorted(counts.items()))
    output_multiset = canonical_sha256(sorted(Counter(ordered).items()))
    failures = []
    if len(manifest) < MIN_REALIZED_PAIRS:
        failures.append("insufficient_realized_pairs")
    if len(batches) < MIN_PAIR_BATCHES:
        failures.append("insufficient_pair_batches")
    if input_multiset != output_multiset:
        failures.append("parent_record_multiset_changed")
    if max(pos_reuse.values(), default=0) > MAX_PAIRS_PER_POSITIVE:
        failures.append("positive_pair_cap_exceeded")
    if max(neg_reuse.values(), default=0) > MAX_NEGATIVE_REUSE:
        failures.append("negative_reuse_cap_exceeded")
    audit = {
        "experiment_id": "560",
        "pairing_version": PAIRING_VERSION,
        "holdout_fold": int(holdout_fold),
        "selector_rows": 909,
        "selector_label_blind": True,
        "pair_labels_source": "runtime outer-train parent donors only",
        "exp530_pair_manifest_sha256": exp530_audit["folds"][str(holdout_fold)]["file_sha256"],
        "training_records": len(training_records),
        "ordered_training_records": len(ordered),
        "parent_record_multiset_sha256": input_multiset,
        "candidate_record_multiset_sha256": output_multiset,
        "ordered_records_sha256_diagnostic_only": canonical_sha256(ordered),
        "ordered_hash_is_membership_gate": False,
        "eligible_parent_intersection_pairs": len(candidates),
        "pair_batches": len(batches),
        "realized_pairs": len(manifest),
        "same_cue_mask_pairs": sum(row.selection_scope == "same_cue_mask" for row in manifest),
        "global_fallback_pairs": sum(row.selection_scope == "global_fallback" for row in manifest),
        "max_pairs_per_positive": max(pos_reuse.values(), default=0),
        "max_negative_reuse": max(neg_reuse.values(), default=0),
        "positive_pair_cap": MAX_PAIRS_PER_POSITIVE,
        "negative_reuse_cap": MAX_NEGATIVE_REUSE,
        "outer_validation_pairs": 0,
        "unsafe_pairs": 0,
        "multiplicity_unchanged": input_multiset == output_multiset,
        "steps_unchanged": True,
        "manifest_sha256": canonical_sha256(manifest_rows),
        "failures": failures,
        "decision": "GO" if not failures else "NO_GO",
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return ordered, manifest_rows, audit

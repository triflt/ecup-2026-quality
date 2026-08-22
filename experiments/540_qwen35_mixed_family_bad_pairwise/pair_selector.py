from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

BAD = "БАД"
PAIRING_VERSION = "bad_pairwise_connected_guard_v1"
MAX_NEGATIVES_PER_POSITIVE = 4
MAX_NEGATIVE_REUSE = 4
MIN_NAME_SHARED_TOKENS = 2
MIN_NAME_JACCARD = 0.45
NEAREST_FAMILIES = 8
MIN_REALIZED_PAIRS = 100
MIN_PAIR_BATCHES = 50

_TOKEN = re.compile(r"[0-9a-zа-я]+", re.IGNORECASE)
_STOP = frozenset(
    {
        "для",
        "или",
        "при",
        "это",
        "товар",
        "продукт",
        "комплекс",
        "набор",
        "пищевой",
        "добавка",
        "капсулы",
        "таблетки",
    }
)


@dataclass(frozen=True)
class TopologyEdge:
    left_family: str
    right_family: str
    shared_tokens: int
    name_jaccard: float


@dataclass(frozen=True)
class PairRow:
    batch_index: int
    positive_index: int
    positive_id: str
    negative_index: int
    negative_id: str
    positive_family: str
    negative_family: str
    relation: str
    name_jaccard: float


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def normalized_name_tokens(value: object) -> frozenset[str]:
    value = "" if pd.isna(value) else html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    return frozenset(
        token
        for token in _TOKEN.findall(value)
        if len(token) >= 4 and token not in _STOP and not token.isdigit()
    )


def build_label_blind_topology(
    frame: pd.DataFrame,
    guard: pd.DataFrame,
    fold_ids: np.ndarray,
    *,
    holdout_fold: int,
) -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, str], TopologyEdge], dict[str, Any]]:
    """Build same/nearest-family candidates without accepting a label argument."""
    if not np.array_equal(frame["id"].astype(str).to_numpy(), guard["id"].astype(str)):
        raise ValueError("connected guard id mismatch")
    safe = guard["safe_for_selection"].astype(bool).to_numpy()
    eligible = (
        (frame["category"].astype(str).to_numpy() == BAD)
        & (fold_ids.astype(np.int8) != int(holdout_fold))
        & safe
    )
    families = guard["connected_component"].astype(str).to_numpy()
    family_tokens: dict[str, set[str]] = defaultdict(set)
    for index in np.flatnonzero(eligible):
        family_tokens[families[index]].update(normalized_name_tokens(frame.iloc[index]["name"]))

    inverted: dict[str, list[str]] = defaultdict(list)
    frozen_tokens = {key: frozenset(value) for key, value in family_tokens.items()}
    for family, tokens in frozen_tokens.items():
        for token in sorted(tokens):
            inverted[token].append(family)
    nearest: dict[str, tuple[str, ...]] = {}
    edges: dict[tuple[str, str], TopologyEdge] = {}
    for family in sorted(frozen_tokens):
        tokens = frozen_tokens[family]
        candidates: set[str] = set()
        for token in tokens:
            candidates.update(inverted[token])
        candidates.discard(family)
        scored = []
        for other in candidates:
            other_tokens = frozen_tokens[other]
            shared = len(tokens & other_tokens)
            union = len(tokens | other_tokens)
            similarity = shared / union if union else 0.0
            if shared < MIN_NAME_SHARED_TOKENS or similarity < MIN_NAME_JACCARD:
                continue
            scored.append((-similarity, -shared, other))
        selected = tuple(item[2] for item in sorted(scored)[:NEAREST_FAMILIES])
        nearest[family] = selected
        for other in selected:
            key = tuple(sorted((family, other)))
            if key not in edges:
                shared = len(tokens & frozen_tokens[other])
                union = len(tokens | frozen_tokens[other])
                edges[key] = TopologyEdge(key[0], key[1], shared, shared / union)
    return (
        nearest,
        edges,
        {
            "pairing_version": PAIRING_VERSION,
            "label_blind": True,
            "eligible_safe_bad_rows": int(eligible.sum()),
            "families": len(frozen_tokens),
            "nearest_edges": len(edges),
            "min_shared_tokens": MIN_NAME_SHARED_TOKENS,
            "min_name_jaccard": MIN_NAME_JACCARD,
            "nearest_families_cap": NEAREST_FAMILIES,
            "topology_sha256": canonical_sha256(
                [
                    (edge.left_family, edge.right_family, edge.shared_tokens, edge.name_jaccard)
                    for edge in edges.values()
                ]
            ),
        },
    )


def build_pair_plan(
    frame: pd.DataFrame,
    guard: pd.DataFrame,
    fold_ids: np.ndarray,
    training_records: list[int],
    *,
    holdout_fold: int,
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    nearest, topology_edges, topology_audit = build_label_blind_topology(
        frame, guard, fold_ids, holdout_fold=holdout_fold
    )
    counts = Counter(int(index) for index in training_records)
    families = guard["connected_component"].astype(str).to_numpy()
    safe = guard["safe_for_selection"].astype(bool).to_numpy()
    labels = frame["label"].to_numpy(np.int8)
    donor_indices = [
        index
        for index in counts
        if str(frame.iloc[index]["category"]) == BAD
        and safe[index]
        and int(fold_ids[index]) != holdout_fold
    ]
    by_family_label: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in donor_indices:
        by_family_label[(families[index], int(labels[index]))].append(index)
    for values in by_family_label.values():
        values.sort(
            key=lambda index: hashlib.sha256(str(frame.iloc[index]["id"]).encode()).hexdigest()
        )

    logical: dict[tuple[int, int], tuple[str, float]] = {}
    negative_reuse: Counter[int] = Counter()
    positives = sorted(
        (index for index in donor_indices if labels[index] == 1),
        key=lambda index: hashlib.sha256(str(frame.iloc[index]["id"]).encode()).hexdigest(),
    )
    for positive in positives:
        family = families[positive]
        candidates: list[tuple[int, str, float]] = [
            (negative, "same_family", 1.0) for negative in by_family_label.get((family, 0), [])
        ]
        for other in nearest.get(family, ()):
            edge = topology_edges[tuple(sorted((family, other)))]
            candidates.extend(
                (negative, "nearest_family", edge.name_jaccard)
                for negative in by_family_label.get((other, 0), [])
            )
        candidates.sort(
            key=lambda item: (
                0 if item[1] == "same_family" else 1,
                -item[2],
                hashlib.sha256(str(frame.iloc[item[0]]["id"]).encode()).hexdigest(),
            )
        )
        for negative, relation, similarity in candidates:
            if len([key for key in logical if key[0] == positive]) >= MAX_NEGATIVES_PER_POSITIVE:
                break
            if negative_reuse[negative] >= MAX_NEGATIVE_REUSE:
                continue
            logical[(positive, negative)] = (relation, similarity)
            negative_reuse[negative] += 1

    unused = set(counts)
    pair_batches: list[list[int]] = []
    realized: list[PairRow] = []
    logical_edges = sorted(
        logical,
        key=lambda pair: hashlib.sha256(
            f"{frame.iloc[pair[0]]['id']}\0{frame.iloc[pair[1]]['id']}".encode()
        ).hexdigest(),
    )
    cursor = 0
    while cursor < len(logical_edges):
        first = logical_edges[cursor]
        cursor += 1
        if first[0] not in unused or first[1] not in unused:
            continue
        nodes = [first[0], first[1]]
        for second in logical_edges[cursor:]:
            if second[0] in unused and second[1] in unused and not set(second) & set(nodes):
                nodes.extend(second)
                break
        if len(nodes) < 4:
            fillers = [
                index
                for index in training_records
                if index in unused and index not in nodes and counts[index] == 1
            ]
            nodes.extend(fillers[: 4 - len(nodes)])
        if len(nodes) != 4:
            break
        batch_index = len(pair_batches)
        pair_batches.append(nodes)
        unused.difference_update(nodes)
        positives_in_batch = [
            index for index in nodes if (index, nodes[1]) in logical or labels[index] == 1
        ]
        negatives_in_batch = [index for index in nodes if labels[index] == 0]
        for positive in positives_in_batch:
            for negative in negatives_in_batch:
                detail = logical.get((positive, negative))
                if detail is None:
                    continue
                relation, similarity = detail
                realized.append(
                    PairRow(
                        batch_index=batch_index,
                        positive_index=positive,
                        positive_id=str(frame.iloc[positive]["id"]),
                        negative_index=negative,
                        negative_id=str(frame.iloc[negative]["id"]),
                        positive_family=families[positive],
                        negative_family=families[negative],
                        relation=relation,
                        name_jaccard=similarity,
                    )
                )

    ordered = [index for batch in pair_batches for index in batch]
    ordered.extend(index for index in training_records if index in unused)
    manifest = [asdict(row) for row in realized]
    realized_pos = Counter(row.positive_index for row in realized)
    realized_neg = Counter(row.negative_index for row in realized)
    unsafe_pairs = sum(
        not safe[row.positive_index] or not safe[row.negative_index] for row in realized
    )
    holdout_pairs = sum(
        int(fold_ids[row.positive_index]) == holdout_fold
        or int(fold_ids[row.negative_index]) == holdout_fold
        for row in realized
    )
    input_multiset = canonical_sha256(sorted(counts.items()))
    output_multiset = canonical_sha256(sorted(Counter(ordered).items()))
    failures = []
    if len(realized) < MIN_REALIZED_PAIRS:
        failures.append("insufficient_realized_pairs")
    if len(pair_batches) < MIN_PAIR_BATCHES:
        failures.append("insufficient_pair_batches")
    if unsafe_pairs:
        failures.append("unsafe_component_donor")
    if holdout_pairs:
        failures.append("outer_validation_donor")
    if input_multiset != output_multiset:
        failures.append("parent_record_multiset_changed")
    if max(realized_pos.values(), default=0) > MAX_NEGATIVES_PER_POSITIVE:
        failures.append("positive_pair_cap_exceeded")
    if max(realized_neg.values(), default=0) > MAX_NEGATIVE_REUSE:
        failures.append("negative_reuse_cap_exceeded")
    audit = {
        "experiment_id": "540",
        "holdout_fold": int(holdout_fold),
        "topology": topology_audit,
        "training_records": len(training_records),
        "ordered_training_records": len(ordered),
        "parent_record_multiset_sha256": input_multiset,
        "candidate_record_multiset_sha256": output_multiset,
        "ordered_records_sha256": canonical_sha256(ordered),
        "logical_pairs": len(logical),
        "pair_batches": len(pair_batches),
        "realized_pairs": len(realized),
        "same_family_pairs": sum(row.relation == "same_family" for row in realized),
        "nearest_family_pairs": sum(row.relation == "nearest_family" for row in realized),
        "max_pairs_per_positive": max(realized_pos.values(), default=0),
        "max_negative_reuse": max(realized_neg.values(), default=0),
        "positive_pair_cap": MAX_NEGATIVES_PER_POSITIVE,
        "negative_reuse_cap": MAX_NEGATIVE_REUSE,
        "unsafe_component_pairs": int(unsafe_pairs),
        "outer_validation_pairs": int(holdout_pairs),
        "steps_unchanged": True,
        "multiplicity_unchanged": input_multiset == output_multiset,
        "manifest_sha256": canonical_sha256(manifest),
        "failures": failures,
        "decision": "GO" if not failures else "NO_GO",
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return ordered, manifest, audit

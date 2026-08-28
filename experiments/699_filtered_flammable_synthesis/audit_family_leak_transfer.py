from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import itertools
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score

BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"
FOLDS = (0, 1, 2, 3, 4)
EXPECTED_ROW_KEYS = {
    "category",
    "description",
    "fold",
    "global_index",
    "id",
    "image_url",
    "name",
    "ocr_images",
    "semantic_component",
}
TOPOLOGY_THRESHOLDS = {
    "rare_shingle": {
        "token_width": 5,
        "document_frequency_min": 2,
        "document_frequency_max": 30,
        "intersection_min": 8,
        "containment_min": 0.70,
        "jaccard_min": 0.45,
    },
    "near_image": {
        "minimum_grayscale_std": 8.0,
        "phash_hamming_max": 4,
        "dhash_hamming_max": 6,
    },
    "high_confidence_multimodal": {
        "weak_text_intersection_min": 5,
        "weak_text_containment_min": 0.40,
        "weak_text_jaccard_min": 0.20,
        "minimum_grayscale_std": 8.0,
        "phash_hamming_max": 12,
        "dhash_hamming_max": 16,
    },
    "normalized_text": {
        "minimum_characters": 30,
        "minimum_tokens": 5,
        "digits_masked": True,
    },
}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_self_hashed(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    payload = dict(payload)
    payload["self_hash_algorithm"] = "sha256_canonical_json_without_self_sha256"
    payload["self_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return sha256_file(path), str(payload["self_sha256"])


def verify_self_hash(payload: dict[str, Any]) -> str:
    copy = dict(payload)
    declared = str(copy.pop("self_sha256", ""))
    if copy.get("self_hash_algorithm") != "sha256_canonical_json_without_self_sha256":
        raise ValueError("unsupported self-hash algorithm")
    if declared != canonical_sha256(copy):
        raise ValueError("self-hash mismatch")
    return declared


def verify_existing_self_hash(payload: dict[str, Any]) -> str:
    copy = dict(payload)
    declared = str(copy.pop("self_sha256", ""))
    copy.pop("self_hash_algorithm", None)
    if declared != canonical_sha256(copy):
        raise ValueError("existing report self-hash mismatch")
    return declared


def canonical_text(value: object, *, mask_digits: bool) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    if mask_digits:
        text = re.sub(r"\d+(?:[.,]\d+)?", " # ", text)
    text = re.sub(r"[^0-9a-zа-я#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def row_text(row: dict[str, Any], *, mask_digits: bool) -> str:
    return canonical_text(
        f"{row.get('name', '')}\n{row.get('name', '')}\n{row.get('description', '')}",
        mask_digits=mask_digits,
    )


def stable_shingle(tokens: tuple[str, ...]) -> int:
    return int.from_bytes(
        hashlib.blake2b("\x1f".join(tokens).encode("utf-8"), digest_size=8).digest(),
        "little",
    )


def image_cache_path(cache: Path, row_id: str) -> Path:
    return cache / f"{hashlib.sha256(row_id.encode()).hexdigest()}.img"


def validate_image_cache_entry(path: Path, resolved_root: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"image cache entry is not a file: {path.name}")
    resolved = path.resolve(strict=True)
    allowed = resolved_root.resolve(strict=True)
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"image cache target escapes approved root: {path.name}") from error
    if not resolved.is_file():
        raise ValueError(f"resolved image cache target is not a file: {path.name}")
    return resolved


def dct_matrix(size: int) -> np.ndarray:
    x = np.arange(size, dtype=np.float64)
    k = x[:, None]
    matrix = np.cos((math.pi / size) * (x + 0.5) * k)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


_DCT32 = dct_matrix(32)


def bits_to_int(bits: np.ndarray) -> int:
    result = 0
    for value in bits.reshape(-1):
        result = (result << 1) | int(bool(value))
    return result


def image_fingerprints(path: Path) -> dict[str, Any]:
    file_sha = sha256_file(path)
    with Image.open(path) as image:
        image.load()
        gray32 = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
        gray9x8 = np.asarray(
            image.convert("L").resize((9, 8), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
    coefficients = _DCT32 @ gray32 @ _DCT32.T
    low = coefficients[:8, :8].copy()
    median = float(np.median(low.reshape(-1)[1:]))
    phash = bits_to_int(low >= median)
    dhash = bits_to_int(gray9x8[:, 1:] >= gray9x8[:, :-1])
    return {
        "file_sha256": file_sha,
        "phash64": phash,
        "dhash64": dhash,
        "grayscale_std": float(gray32.std()),
    }


class DSU:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, dict[int, Any]] | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            child = node[1].get(distance)
            if child is None:
                node[1][distance] = (value, {})
                return
            node = child

    def query(self, value: int, maximum: int) -> list[int]:
        if self.root is None:
            return []
        results: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= maximum:
                results.append(node[0])
            low = distance - maximum
            high = distance + maximum
            stack.extend(
                child for key, child in node[1].items() if low <= key <= high
            )
        return results


def parse_fold_path(value: str) -> tuple[int, Path]:
    raw_fold, raw_path = value.split("=", 1)
    fold = int(raw_fold)
    if fold not in FOLDS:
        raise ValueError("fold path must bind folds0..4")
    return fold, Path(raw_path)


def load_label_free_rows(fold_paths: dict[int, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(fold_paths) != set(FOLDS):
        raise ValueError("exact five fold sources required")
    rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for fold in FOLDS:
        path = fold_paths[fold]
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"fold{fold} runtime is missing or symlinked")
        local = [json.loads(line) for line in path.read_text().splitlines()]
        if any(set(row) != EXPECTED_ROW_KEYS for row in local):
            raise ValueError(f"fold{fold} schema mismatch")
        if any(int(row["fold"]) != fold for row in local):
            raise ValueError(f"fold{fold} binding mismatch")
        if any("label" in row or "target" in row for row in local):
            raise ValueError("label-like field entered topology stage")
        rows.extend(local)
        sources[str(fold)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(local),
        }
    rows.sort(key=lambda row: int(row["global_index"]))
    indices = [int(row["global_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("global_index coverage mismatch")
    keys = [(str(row["id"]), int(row["fold"]), str(row["category"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("row binding key is duplicated")
    return rows, sources


def add_group_edges(
    groups: dict[str, list[int]], source: str, add_edge
) -> None:
    for members in groups.values():
        unique = sorted(set(members))
        if len(unique) > 1:
            anchor = unique[0]
            for member in unique[1:]:
                add_edge(anchor, member, source)


def build_topology(args: argparse.Namespace) -> None:
    fold_paths = dict(parse_fold_path(value) for value in args.fold_runtime)
    rows, sources = load_label_free_rows(fold_paths)
    ids = sorted({str(row["id"]) for row in rows})
    id_position = {value: index for index, value in enumerate(ids)}
    rows_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_id[str(row["id"])].append(row)

    dsu = DSU(len(ids))
    edge_sources: dict[tuple[int, int], set[str]] = defaultdict(set)

    def add_edge(left: int, right: int, source: str) -> None:
        if left == right:
            return
        pair = (left, right) if left < right else (right, left)
        edge_sources[pair].add(source)
        dsu.union(*pair)

    semantic_groups: dict[str, list[int]] = defaultdict(list)
    exact_groups: dict[str, list[int]] = defaultdict(list)
    normalized_groups: dict[str, list[int]] = defaultdict(list)
    text_by_id: dict[str, str] = {}
    masked_by_id: dict[str, str] = {}
    for row_id, local_rows in rows_by_id.items():
        texts = {row_text(row, mask_digits=False) for row in local_rows}
        masked = {row_text(row, mask_digits=True) for row in local_rows}
        if len(texts) != 1 or len(masked) != 1:
            raise ValueError("same id has inconsistent text surface")
        text = texts.pop()
        masked_text = masked.pop()
        text_by_id[row_id] = text
        masked_by_id[row_id] = masked_text
        exact_groups[text].append(id_position[row_id])
        if (
            len(masked_text) >= TOPOLOGY_THRESHOLDS["normalized_text"]["minimum_characters"]
            and len(masked_text.split())
            >= TOPOLOGY_THRESHOLDS["normalized_text"]["minimum_tokens"]
        ):
            normalized_groups[masked_text].append(id_position[row_id])
        for row in local_rows:
            semantic_groups[
                f"{row['category']}\x1f{row['semantic_component']}"
            ].append(id_position[row_id])
    add_group_edges(semantic_groups, "semantic_component", add_edge)
    add_group_edges(exact_groups, "exact_text", add_edge)
    add_group_edges(normalized_groups, "normalized_text", add_edge)

    shingle_sets: list[set[int]] = []
    frequencies: Counter[int] = Counter()
    width = int(TOPOLOGY_THRESHOLDS["rare_shingle"]["token_width"])
    for row_id in ids:
        tokens = masked_by_id[row_id].split()
        shingles = {
            stable_shingle(tuple(tokens[start : start + width]))
            for start in range(max(0, len(tokens) - width + 1))
        }
        shingle_sets.append(shingles)
        frequencies.update(shingles)
    minimum_df = int(TOPOLOGY_THRESHOLDS["rare_shingle"]["document_frequency_min"])
    maximum_df = int(TOPOLOGY_THRESHOLDS["rare_shingle"]["document_frequency_max"])
    useful = {key for key, count in frequencies.items() if minimum_df <= count <= maximum_df}
    postings: dict[int, list[int]] = defaultdict(list)
    for index, shingles in enumerate(shingle_sets):
        for key in shingles & useful:
            postings[key].append(index)
    pair_intersections: Counter[tuple[int, int]] = Counter()
    for members in postings.values():
        for left, right in itertools.combinations(sorted(set(members)), 2):
            pair_intersections[(left, right)] += 1

    weak_text_pairs: dict[tuple[int, int], tuple[int, float, float]] = {}
    rare_config = TOPOLOGY_THRESHOLDS["rare_shingle"]
    multi_config = TOPOLOGY_THRESHOLDS["high_confidence_multimodal"]
    for pair, intersection in pair_intersections.items():
        left, right = pair
        minimum_size = min(len(shingle_sets[left]), len(shingle_sets[right]))
        union_size = len(shingle_sets[left]) + len(shingle_sets[right]) - intersection
        if not minimum_size or not union_size:
            continue
        containment = intersection / minimum_size
        jaccard = intersection / union_size
        if (
            intersection >= int(multi_config["weak_text_intersection_min"])
            and containment >= float(multi_config["weak_text_containment_min"])
            and jaccard >= float(multi_config["weak_text_jaccard_min"])
        ):
            weak_text_pairs[pair] = (intersection, containment, jaccard)
        if (
            intersection >= int(rare_config["intersection_min"])
            and containment >= float(rare_config["containment_min"])
            and jaccard >= float(rare_config["jaccard_min"])
        ):
            add_edge(left, right, "rare_shingle")

    image_info: list[dict[str, Any]] = []
    image_manifest: list[dict[str, Any]] = []
    symlink_entries = 0
    for index, row_id in enumerate(ids):
        path = image_cache_path(args.image_cache, row_id)
        resolved = validate_image_cache_entry(path, args.image_resolved_root)
        symlink_entries += int(path.is_symlink())
        item = image_fingerprints(path)
        image_info.append(item)
        image_manifest.append(
            {
                "cache_name": path.name,
                "file_sha256": item["file_sha256"],
                "size": path.stat().st_size,
                "resolved_relative_path": str(
                    resolved.relative_to(args.image_resolved_root.resolve(strict=True))
                ),
            }
        )
        if (index + 1) % 500 == 0 or index + 1 == len(ids):
            print(f"image_fingerprints={index + 1}/{len(ids)}", flush=True)

    exact_image_groups: dict[str, list[int]] = defaultdict(list)
    phash_groups: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(image_info):
        if float(item["grayscale_std"]) >= float(
            TOPOLOGY_THRESHOLDS["near_image"]["minimum_grayscale_std"]
        ):
            exact_image_groups[str(item["file_sha256"])].append(index)
            phash_groups[int(item["phash64"])].append(index)
    add_group_edges(exact_image_groups, "near_image_exact_bytes", add_edge)

    tree = BKTree()
    for value in sorted(phash_groups):
        tree.add(value)
    near_config = TOPOLOGY_THRESHOLDS["near_image"]
    seen_phash_pairs: set[tuple[int, int]] = set()
    for value, members in phash_groups.items():
        for other in tree.query(value, int(near_config["phash_hamming_max"])):
            pair_hash = (value, other) if value <= other else (other, value)
            if pair_hash in seen_phash_pairs:
                continue
            seen_phash_pairs.add(pair_hash)
            for left in members:
                for right in phash_groups[other]:
                    if left >= right:
                        continue
                    if (
                        int(image_info[left]["dhash64"])
                        ^ int(image_info[right]["dhash64"])
                    ).bit_count() <= int(near_config["dhash_hamming_max"]):
                        add_edge(left, right, "near_image")

    for pair in weak_text_pairs:
        left, right = pair
        if min(
            float(image_info[left]["grayscale_std"]),
            float(image_info[right]["grayscale_std"]),
        ) < float(multi_config["minimum_grayscale_std"]):
            continue
        phash_distance = (
            int(image_info[left]["phash64"]) ^ int(image_info[right]["phash64"])
        ).bit_count()
        dhash_distance = (
            int(image_info[left]["dhash64"]) ^ int(image_info[right]["dhash64"])
        ).bit_count()
        if (
            phash_distance <= int(multi_config["phash_hamming_max"])
            and dhash_distance <= int(multi_config["dhash_hamming_max"])
        ):
            add_edge(left, right, "high_confidence_multimodal")

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(ids)):
        members_by_root[dsu.find(index)].append(index)
    edge_types_by_root: dict[int, set[str]] = defaultdict(set)
    for pair, sources_for_pair in edge_sources.items():
        root = dsu.find(pair[0])
        if dsu.find(pair[1]) != root:
            raise ValueError("edge endpoints are not in the same component")
        edge_types_by_root[root].update(sources_for_pair)
    node_folds = {
        id_position[row_id]: {int(row["fold"]) for row in local_rows}
        for row_id, local_rows in rows_by_id.items()
    }
    component_payload: dict[int, dict[str, Any]] = {}
    component_id_by_node: dict[int, str] = {}
    for root, members in members_by_root.items():
        member_ids = [ids[index] for index in members]
        component_id = hashlib.sha256("\n".join(member_ids).encode()).hexdigest()
        folds = sorted(set().union(*(node_folds[index] for index in members)))
        categories = sorted(
            {
                str(row["category"])
                for index in members
                for row in rows_by_id[ids[index]]
            }
        )
        edge_types = sorted(edge_types_by_root[root])
        row_count = sum(len(rows_by_id[ids[index]]) for index in members)
        component_payload[root] = {
            "component_id": component_id,
            "unique_ids": len(members),
            "rows": row_count,
            "folds": folds,
            "categories": categories,
            "edge_types": edge_types,
        }
        for index in members:
            component_id_by_node[index] = component_id

    component_by_id = {
        payload["component_id"]: payload for payload in component_payload.values()
    }
    registry = []
    for row in rows:
        node = id_position[str(row["id"])]
        component = component_by_id[component_id_by_node[node]]
        registry.append(
            {
                "global_index": int(row["global_index"]),
                "id": str(row["id"]),
                "fold": int(row["fold"]),
                "category": str(row["category"]),
                "component_id": component["component_id"],
                "component_rows": int(component["rows"]),
                "component_unique_ids": int(component["unique_ids"]),
                "component_fold_count": len(component["folds"]),
                "component_edge_types": list(component["edge_types"]),
                "regime": (
                    "recurrence_cross_fold"
                    if len(component["folds"]) > 1
                    else "novel_family_single_fold"
                ),
            }
        )

    edge_counts: Counter[str] = Counter()
    cross_fold_edge_counts: Counter[str] = Counter()
    for pair, source_set in edge_sources.items():
        cross_fold = bool(node_folds[pair[0]] - node_folds[pair[1]]) or bool(
            node_folds[pair[1]] - node_folds[pair[0]]
        )
        for source in source_set:
            edge_counts[source] += 1
            if cross_fold:
                cross_fold_edge_counts[source] += 1
    same_id_cross_fold = sum(1 for folds in node_folds.values() if len(folds) > 1)
    components = list(component_payload.values())
    recurrence_rows = sum(
        component["rows"] for component in components if len(component["folds"]) > 1
    )
    payload = {
        "schema": "exp699_family_leak_topology_v1",
        "experiment": 699,
        "stage": "label_free_topology_before_metrics",
        "decision": "TOPOLOGY_READY_FOR_METRICS",
        "labels_read": 0,
        "public_used": False,
        "sealed_rows": 0,
        "source_validation": sources,
        "image_cache": {
            "path": str(args.image_cache),
            "unique_ids": len(ids),
            "manifest_sha256": canonical_sha256(image_manifest),
            "resolved_root": str(args.image_resolved_root),
            "symlink_entries": symlink_entries,
        },
        "thresholds": TOPOLOGY_THRESHOLDS,
        "rows": len(rows),
        "unique_ids": len(ids),
        "edges": len(edge_sources),
        "edge_counts": dict(sorted(edge_counts.items())),
        "cross_fold_edge_counts": dict(sorted(cross_fold_edge_counts.items())),
        "same_id_cross_fold_nodes": same_id_cross_fold,
        "components": len(components),
        "cross_fold_components": sum(len(item["folds"]) > 1 for item in components),
        "recurrence_rows": recurrence_rows,
        "novel_family_rows": len(rows) - recurrence_rows,
        "largest_component_rows": max(item["rows"] for item in components),
        "component_summary": sorted(
            components,
            key=lambda item: (-int(item["rows"]), str(item["component_id"])),
        ),
        "registry": registry,
    }
    file_sha, self_sha = write_self_hashed(args.output, payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "file_sha256": file_sha,
                "self_sha256": self_sha,
                "rows": len(rows),
                "components": len(components),
                "cross_fold_components": payload["cross_fold_components"],
                "recurrence_rows": recurrence_rows,
                "edge_counts": payload["edge_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fold_category_ranks(values: np.ndarray, folds: np.ndarray, categories: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float32)
    for fold in sorted(np.unique(folds)):
        for category in sorted(np.unique(categories)):
            positions = np.flatnonzero((folds == fold) & (categories == category))
            order = np.argsort(values[positions], kind="mergesort")
            ranks = np.empty(len(positions), dtype=np.float32)
            ranks[order] = np.linspace(0.0, 1.0, len(positions), dtype=np.float32)
            result[positions] = ranks
    return result


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    return 2 * tp / max(1, 2 * tp + fp + fn)


def metric_summary(
    labels: np.ndarray,
    categories: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": int(mask.sum()), "categories": {}}
    values = []
    for category in (BAD, FLAMMABLE):
        local = mask & (categories == category)
        if not local.any():
            continue
        local_labels = labels[local]
        local_predictions = predictions[local]
        confusion = {
            "tp": int(np.sum((local_labels == 1) & (local_predictions == 1))),
            "fp": int(np.sum((local_labels == 0) & (local_predictions == 1))),
            "fn": int(np.sum((local_labels == 1) & (local_predictions == 0))),
            "tn": int(np.sum((local_labels == 0) & (local_predictions == 0))),
        }
        score = f1(local_labels, local_predictions)
        result["categories"][category] = {
            "rows": int(local.sum()),
            "f1": score,
            "confusion": confusion,
        }
        values.append(score)
    result["macro_f1"] = float(np.mean(values)) if values else None
    return result


def comparison_summary(
    labels: np.ndarray,
    categories: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    corrected = mask & (candidate == labels) & (baseline != labels)
    regressed = mask & (candidate != labels) & (baseline == labels)
    changed = mask & (candidate != baseline)
    result = {
        "changed": int(changed.sum()),
        "corrections": int(corrected.sum()),
        "regressions": int(regressed.sum()),
        "net_correct_decisions": int(corrected.sum() - regressed.sum()),
        "categories": {},
    }
    for category in (BAD, FLAMMABLE):
        local = categories == category
        result["categories"][category] = {
            "changed": int((changed & local).sum()),
            "corrections": int((corrected & local).sum()),
            "regressions": int((regressed & local).sum()),
        }
    return result


def variant_report(
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    regimes: dict[str, np.ndarray],
    baseline_before: np.ndarray,
    candidate_before: np.ndarray,
    baseline_after: np.ndarray,
    candidate_after: np.ndarray,
    baseline_q35: np.ndarray,
    candidate_q35: np.ndarray,
    scope: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {"regimes": {}}
    overall_before = comparison_summary(
        labels, categories, baseline_before, candidate_before, scope
    )
    overall_net = int(overall_before["net_correct_decisions"])
    for name, regime in regimes.items():
        mask = scope & regime
        before_baseline = metric_summary(labels, categories, baseline_before, mask)
        before_candidate = metric_summary(labels, categories, candidate_before, mask)
        after_baseline = metric_summary(labels, categories, baseline_after, mask)
        after_candidate = metric_summary(labels, categories, candidate_after, mask)
        before_compare = comparison_summary(
            labels, categories, baseline_before, candidate_before, mask
        )
        after_compare = comparison_summary(
            labels, categories, baseline_after, candidate_after, mask
        )
        flammable = mask & (categories == FLAMMABLE)
        ap = None
        if flammable.any() and len(np.unique(labels[flammable])) == 2:
            ap = {
                "baseline_q35": float(
                    average_precision_score(labels[flammable], baseline_q35[flammable])
                ),
                "candidate_q35": float(
                    average_precision_score(labels[flammable], candidate_q35[flammable])
                ),
            }
            ap["delta"] = ap["candidate_q35"] - ap["baseline_q35"]
        output["regimes"][name] = {
            "before_prior": {
                "baseline": before_baseline,
                "candidate": before_candidate,
                "macro_delta": (
                    None
                    if before_baseline["macro_f1"] is None
                    else before_candidate["macro_f1"] - before_baseline["macro_f1"]
                ),
                "comparison": before_compare,
                "flammable_q35_ap": ap,
            },
            "after_prior": {
                "baseline": after_baseline,
                "candidate": after_candidate,
                "macro_delta": (
                    None
                    if after_baseline["macro_f1"] is None
                    else after_candidate["macro_f1"] - after_baseline["macro_f1"]
                ),
                "comparison": after_compare,
            },
            "fraction_of_before_prior_net_lift": (
                None
                if overall_net == 0
                else float(before_compare["net_correct_decisions"] / overall_net)
            ),
        }
    changed_before = scope & (baseline_before != candidate_before)
    changed_after = scope & (baseline_after != candidate_after)
    output["decision_survival"] = {
        "changed_before_prior": int(changed_before.sum()),
        "survived_after_prior": int((changed_before & changed_after).sum()),
        "suppressed_by_prior": int((changed_before & ~changed_after).sum()),
        "introduced_after_prior": int((~changed_before & changed_after & scope).sum()),
    }
    output["evaluation_rows"] = int(scope.sum())
    return output


def find_scenario(report: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [row for row in report["leaderboard"] if row["scenario"] == name]
    if len(rows) != 1:
        raise ValueError(f"scenario multiplicity mismatch: {name}")
    return rows[0]


def parse_candidate(value: str) -> tuple[str, str, Path, Path]:
    name, scenario, root, report = value.split("=", 3)
    return name, scenario, Path(root), Path(report)


def parse_routed_candidate(value: str) -> tuple[str, Path, Path]:
    name, root, report = value.split("=", 2)
    return name, Path(root), Path(report)


def evaluate_topology(args: argparse.Namespace) -> None:
    topology = json.loads(args.topology.read_text())
    topology_self = verify_self_hash(topology)
    if (
        topology.get("schema") != "exp699_family_leak_topology_v1"
        or topology.get("decision") != "TOPOLOGY_READY_FOR_METRICS"
        or topology.get("labels_read") != 0
        or topology.get("public_used") is not False
    ):
        raise ValueError("topology contract mismatch")
    topology_rows = sorted(topology["registry"], key=lambda row: row["global_index"])

    screen_eval = load_module(args.screen_evaluator, "exp699_screen_eval_for_leak_audit")
    load_module(args.evaluation_module, "exp699_base_eval_for_leak_audit")
    oof = np.load(args.oof, allow_pickle=True)
    oof_ids = oof["ids"].astype(str)
    oof_labels = oof["labels"].astype(np.int8)
    oof_categories = oof["categories"].astype(str)
    oof_folds = oof["fold_ids"].astype(np.int8)
    source_positions, folds, global_indices = screen_eval.load_source_split(
        args.runtime_map,
        args.runtime_map_contract,
        oof_ids,
        oof_categories,
    )
    ids = oof_ids[source_positions]
    labels = oof_labels[source_positions]
    categories = oof_categories[source_positions]
    if len(topology_rows) != len(ids):
        raise ValueError("topology/OOF row count mismatch")
    for position, row in enumerate(topology_rows):
        expected = (
            int(global_indices[position]),
            str(ids[position]),
            int(folds[position]),
            str(categories[position]),
        )
        actual = (
            int(row["global_index"]),
            str(row["id"]),
            int(row["fold"]),
            str(row["category"]),
        )
        if actual != expected:
            raise ValueError("topology/OOF row binding mismatch")

    baseline_q3_source = np.load(args.baseline_qwen3vl, allow_pickle=True)
    baseline_q35_source = np.load(args.baseline_qwen35, allow_pickle=True)
    for source in (baseline_q3_source, baseline_q35_source):
        if (
            not np.array_equal(source["ids"].astype(str), oof_ids)
            or not np.array_equal(source["folds"].astype(np.int8), oof_folds)
            or not np.array_equal(source["categories"].astype(str), oof_categories)
            or not np.array_equal(source["labels"].astype(np.int8), oof_labels)
        ):
            raise ValueError("baseline score binding mismatch")
    robust = baseline_q3_source["base_rank"].astype(np.float32)[source_positions]
    baseline_q3 = baseline_q3_source["lora_rank"].astype(np.float32)[source_positions]
    baseline_q35 = baseline_q35_source["lora_rank"].astype(np.float32)[source_positions]

    full5 = json.loads(args.full5_report.read_text())
    full5_self = verify_existing_self_hash(full5)
    if full5.get("evaluation_folds") != list(FOLDS):
        raise ValueError("full5 report scope mismatch")
    baseline_selection = full5["baseline"]["selection"]
    baseline_before = screen_eval.apply_frozen_selection(
        categories,
        folds,
        {"robust_base": robust, "qwen3vl": baseline_q3, "qwen35": baseline_q35},
        baseline_selection,
    )

    frame = pd.DataFrame(
        {
            "id": ids,
            "label": labels,
            "category": categories,
            "fold": folds,
            "name": ["" for _ in ids],
            "description": ["" for _ in ids],
        }
    )
    metadata_rows: dict[int, dict[str, Any]] = {}
    for path in dict(parse_fold_path(value) for value in args.fold_runtime).values():
        for row in (json.loads(line) for line in path.read_text().splitlines()):
            metadata_rows[int(row["global_index"])] = row
    if set(metadata_rows) != set(range(len(ids))):
        raise ValueError("metadata coverage mismatch")
    frame["name"] = [str(metadata_rows[index]["name"]) for index in range(len(ids))]
    frame["description"] = [
        str(metadata_rows[index]["description"]) for index in range(len(ids))
    ]

    research_root = args.prior_module.parent
    sys.path.insert(0, str(research_root))
    prior = load_module(args.prior_module, "exp699_prior_for_leak_audit")
    prepared = frame.copy()
    prepared["normalized_name"] = prepared["name"].map(prior.normalize)
    texts = [
        prior.compose_text(name, description)
        for name, description in zip(prepared["name"], prepared["description"])
    ]
    prepared["text_hash"] = [prior.fingerprint(text) for text in texts]
    prepared["canonical_text_mask_digits"] = [
        prior.canonical(text, mask_digits=True) for text in texts
    ]
    neighbor_indices, neighbor_scores = prior.build_neighbor_graph(prepared)
    baseline_after, baseline_prior_audit = prior.apply_downstream_priors(
        prepared, baseline_before, neighbor_indices, neighbor_scores
    )

    component_fold_count = np.asarray(
        [int(row["component_fold_count"]) for row in topology_rows], dtype=np.int16
    )
    component_edge_types = [set(row["component_edge_types"]) for row in topology_rows]
    regimes = {
        "mixed_all": np.ones(len(ids), dtype=bool),
        "recurrence_cross_fold": component_fold_count > 1,
        "novel_family_single_fold": component_fold_count == 1,
    }
    all_edge_types = sorted(set().union(*component_edge_types))
    for edge_type in all_edge_types:
        regimes[f"edge_{edge_type}"] = np.asarray(
            [edge_type in values for values in component_edge_types], dtype=bool
        )

    variants: dict[str, Any] = {}
    source_bindings: dict[str, Any] = {}
    for raw in args.candidate:
        name, scenario, root, report_path = parse_candidate(raw)
        report = json.loads(report_path.read_text())
        scenario_row = find_scenario(report, scenario)
        artifact = report["candidate_artifacts"][scenario]
        spec = {
            "name": scenario,
            "architecture": artifact["architecture"],
            "source": artifact["source"],
            "mode": artifact["mode"],
            "cap": int(artifact["cap"]),
            "path": root,
        }
        evaluation_folds = tuple(int(value) for value in report["evaluation_folds"])
        raw_scores, audit = screen_eval.load_candidate(
            spec,
            ids,
            folds,
            categories,
            global_indices,
            evaluation_folds=evaluation_folds,
        )
        candidate_q35 = baseline_q35.copy()
        scope = np.isin(folds, evaluation_folds)
        ranked = fold_category_ranks(raw_scores[scope], folds[scope], categories[scope])
        candidate_q35[scope] = ranked
        named_scores = {
            "robust_base": robust,
            "qwen3vl": baseline_q3,
            "qwen35": candidate_q35,
        }
        candidate_before = screen_eval.apply_frozen_selection(
            categories, folds, named_scores, scenario_row["selection"]
        )
        candidate_after, candidate_prior_audit = prior.apply_downstream_priors(
            prepared, candidate_before, neighbor_indices, neighbor_scores
        )
        variants[name] = variant_report(
            labels,
            categories,
            folds,
            regimes,
            baseline_before,
            candidate_before,
            baseline_after,
            candidate_after,
            baseline_q35,
            candidate_q35,
            scope,
        )
        variants[name]["evaluation_folds"] = list(evaluation_folds)
        variants[name]["candidate_prior_audit"] = candidate_prior_audit
        source_bindings[name] = {
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
            "report_self_sha256": verify_existing_self_hash(report),
            "artifact_audit": audit,
        }

    for raw in args.routed_candidate:
        name, root, report_path = parse_routed_candidate(raw)
        report = json.loads(report_path.read_text())
        report_self = verify_existing_self_hash(report)
        if (
            report.get("schema") != "exp699_routed_q35_oof_v1"
            or report.get("route", {}).get("bad") != "frozen_solution140_qwen35"
            or report.get("route", {}).get("flammable") != "candidate_qwen35"
            or report.get("bad_component_exact") is not True
            or report.get("bad_final_predictions_exact") is not True
        ):
            raise ValueError("routed candidate report mismatch")
        candidate_contract = report["candidate"]
        evaluation_folds = tuple(int(value) for value in report["evaluation_folds"])
        spec = {
            "name": name,
            "architecture": "qwen35_4b",
            "source": candidate_contract["source"],
            "mode": candidate_contract["mode"],
            "cap": int(candidate_contract["cap"]),
            "path": root,
        }
        raw_scores, audit = screen_eval.load_candidate(
            spec,
            ids,
            folds,
            categories,
            global_indices,
            evaluation_folds=evaluation_folds,
        )
        scope = np.isin(folds, evaluation_folds)
        ranked = fold_category_ranks(raw_scores[scope], folds[scope], categories[scope])
        candidate_q35 = baseline_q35.copy()
        routed_scope = scope & (categories == FLAMMABLE)
        candidate_q35[routed_scope] = ranked[categories[scope] == FLAMMABLE]
        candidate_before = screen_eval.apply_frozen_selection(
            categories,
            folds,
            {"robust_base": robust, "qwen3vl": baseline_q3, "qwen35": candidate_q35},
            baseline_selection,
        )
        candidate_after, candidate_prior_audit = prior.apply_downstream_priors(
            prepared, candidate_before, neighbor_indices, neighbor_scores
        )
        variants[name] = variant_report(
            labels,
            categories,
            folds,
            regimes,
            baseline_before,
            candidate_before,
            baseline_after,
            candidate_after,
            baseline_q35,
            candidate_q35,
            scope,
        )
        variants[name]["evaluation_folds"] = list(evaluation_folds)
        variants[name]["candidate_prior_audit"] = candidate_prior_audit
        source_bindings[name] = {
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
            "report_self_sha256": report_self,
            "artifact_audit": audit,
            "route": report["route"],
        }

    payload = {
        "schema": "exp699_family_leak_transfer_audit_v1",
        "experiment": 699,
        "stage": "metrics_after_frozen_label_free_topology",
        "decision": "TERMINAL_FAMILY_LEAK_AUDIT",
        "topology": {
            "path": str(args.topology),
            "file_sha256": sha256_file(args.topology),
            "self_sha256": topology_self,
            "cross_fold_components": topology["cross_fold_components"],
            "recurrence_rows": topology["recurrence_rows"],
            "novel_family_rows": topology["novel_family_rows"],
        },
        "sources": {
            "oof_sha256": sha256_file(args.oof),
            "baseline_qwen3vl_sha256": sha256_file(args.baseline_qwen3vl),
            "baseline_qwen35_sha256": sha256_file(args.baseline_qwen35),
            "runtime_map_sha256": sha256_file(args.runtime_map),
            "runtime_map_contract_sha256": sha256_file(args.runtime_map_contract),
            "full5_report_sha256": sha256_file(args.full5_report),
            "full5_report_self_sha256": full5_self,
            "screen_evaluator_sha256": sha256_file(args.screen_evaluator),
            "evaluation_module_sha256": sha256_file(args.evaluation_module),
            "prior_module_sha256": sha256_file(args.prior_module),
            "candidates": source_bindings,
        },
        "public_used": False,
        "sealed_rows": 0,
        "baseline_prior_audit": baseline_prior_audit,
        "regime_definitions": {
            "mixed_all": "all frozen OOF rows",
            "recurrence_cross_fold": "connected component has rows in at least two current folds",
            "novel_family_single_fold": "connected component is confined to one current fold",
            "edge_*": "overlapping rows whose connected component contains the named edge type",
        },
        "variants": variants,
        "limitations": [
            "Edge-type regimes overlap and their lift fractions are descriptive, not additive.",
            "B/C artifacts cover only folds0/3; full-fivefold claims are made only for the original A candidate.",
            "This audit diagnoses current-fold semantic leakage and does not use Public to select thresholds or weights.",
        ],
    }
    file_sha, self_sha = write_self_hashed(args.output, payload)
    compact = {
        "decision": payload["decision"],
        "file_sha256": file_sha,
        "self_sha256": self_sha,
        "topology": payload["topology"],
        "variants": {
            name: {
                "mixed": item["regimes"]["mixed_all"],
                "recurrence": item["regimes"]["recurrence_cross_fold"],
                "novel": item["regimes"]["novel_family_single_fold"],
            }
            for name, item in variants.items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    topology = subparsers.add_parser("topology")
    topology.add_argument("--fold-runtime", action="append", required=True)
    topology.add_argument("--image-cache", type=Path, required=True)
    topology.add_argument("--image-resolved-root", type=Path, required=True)
    topology.add_argument("--output", type=Path, required=True)
    topology.set_defaults(func=build_topology)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--topology", type=Path, required=True)
    evaluate.add_argument("--fold-runtime", action="append", required=True)
    evaluate.add_argument("--screen-evaluator", type=Path, required=True)
    evaluate.add_argument("--evaluation-module", type=Path, required=True)
    evaluate.add_argument("--prior-module", type=Path, required=True)
    evaluate.add_argument("--oof", type=Path, required=True)
    evaluate.add_argument("--baseline-qwen3vl", type=Path, required=True)
    evaluate.add_argument("--baseline-qwen35", type=Path, required=True)
    evaluate.add_argument("--runtime-map", type=Path, required=True)
    evaluate.add_argument("--runtime-map-contract", type=Path, required=True)
    evaluate.add_argument("--full5-report", type=Path, required=True)
    evaluate.add_argument("--candidate", action="append", required=True)
    evaluate.add_argument("--routed-candidate", action="append", default=[])
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(func=evaluate_topology)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

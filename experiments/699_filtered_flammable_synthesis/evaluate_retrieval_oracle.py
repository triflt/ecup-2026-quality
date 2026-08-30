from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"
FOLDS = (0, 1, 2, 3, 4)
TOP_K = 30
RRF_OFFSET = 60
CHANNEL_WEIGHTS = {
    "exact_text": 2.0,
    "normalized_text": 1.5,
    "graph_component": 1.0,
    "tfidf": 1.0,
    "bm25": 1.0,
    "multimodal_embedding": 1.0,
    "image_exact": 2.0,
    "image_near": 1.0,
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
    if canonical_sha256(copy) != declared:
        raise ValueError("self-hash mismatch")
    return declared


def verify_existing_self_hash(payload: dict[str, Any]) -> str:
    copy = dict(payload)
    declared = str(copy.pop("self_sha256", ""))
    if canonical_sha256(copy) != declared:
        raise ValueError("existing report self-hash mismatch")
    return declared


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_fold_path(value: str) -> tuple[int, Path]:
    raw_fold, raw_path = value.split("=", 1)
    fold = int(raw_fold)
    if fold not in FOLDS:
        raise ValueError("fold must be in 0..4")
    return fold, Path(raw_path)


def load_runtime_rows(paths: dict[int, Path]) -> list[dict[str, Any]]:
    if set(paths) != set(FOLDS):
        raise ValueError("exact folds0..4 are required")
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        local = [json.loads(line) for line in paths[fold].read_text().splitlines()]
        if any(int(row["fold"]) != fold for row in local):
            raise ValueError(f"fold{fold} runtime binding mismatch")
        rows.extend(local)
    rows.sort(key=lambda row: int(row["global_index"]))
    if [int(row["global_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("runtime global-index coverage mismatch")
    return rows


def load_component_outputs(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    keys = [(str(row["id"]), int(row["fold"]), str(row["category"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("component-output key is duplicated")
    return rows


def align_sources(
    topology: dict[str, Any],
    runtime_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    registry = sorted(topology["registry"], key=lambda row: int(row["global_index"]))
    if len(registry) != len(runtime_rows):
        raise ValueError("topology/runtime row-count mismatch")
    component_by_key = {
        (str(row["id"]), int(row["fold"]), str(row["category"])): row
        for row in component_rows
    }
    aligned_components = []
    for registry_row, runtime_row in zip(registry, runtime_rows):
        expected = (
            str(runtime_row["id"]),
            int(runtime_row["fold"]),
            str(runtime_row["category"]),
        )
        actual = (
            str(registry_row["id"]),
            int(registry_row["fold"]),
            str(registry_row["category"]),
        )
        if actual != expected:
            raise ValueError("topology/runtime row binding mismatch")
        component = component_by_key.get(expected)
        if component is None:
            raise ValueError("component-output coverage mismatch")
        aligned_components.append(component)
    return registry, runtime_rows, aligned_components


def dense_top_k(
    query: np.ndarray,
    donors: np.ndarray,
    donor_positions: np.ndarray,
    k: int,
    *,
    chunk_size: int = 256,
) -> list[list[int]]:
    result: list[list[int]] = []
    limit = min(k, len(donor_positions))
    for start in range(0, len(query), chunk_size):
        scores = query[start : start + chunk_size] @ donors.T
        for row in scores:
            if limit == 0:
                result.append([])
                continue
            selected = np.argpartition(-row, limit - 1)[:limit]
            ordered = sorted(
                selected,
                key=lambda index: (-float(row[index]), int(donor_positions[index])),
            )
            result.append([int(donor_positions[index]) for index in ordered])
    return result


def sparse_top_k(
    query: sparse.csr_matrix,
    donors: sparse.csr_matrix,
    donor_positions: np.ndarray,
    k: int,
    *,
    chunk_size: int = 256,
) -> list[list[int]]:
    result: list[list[int]] = []
    limit = min(k, len(donor_positions))
    donor_transpose = donors.T.tocsc()
    for start in range(0, query.shape[0], chunk_size):
        scores = (query[start : start + chunk_size] @ donor_transpose).toarray()
        for row in scores:
            positive = np.flatnonzero(row > 0.0)
            local_limit = min(limit, len(positive))
            if local_limit == 0:
                result.append([])
                continue
            selected = positive[np.argpartition(-row[positive], local_limit - 1)[:local_limit]]
            ordered = sorted(
                selected,
                key=lambda index: (-float(row[index]), int(donor_positions[index])),
            )
            result.append([int(donor_positions[index]) for index in ordered])
    return result


def bm25_documents(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    rows = counts.shape[0]
    document_frequency = np.asarray((counts > 0).sum(axis=0)).reshape(-1)
    inverse_document_frequency = np.log(
        1.0 + (rows - document_frequency + 0.5) / (document_frequency + 0.5)
    ).astype(np.float32)
    lengths = np.asarray(counts.sum(axis=1)).reshape(-1).astype(np.float32)
    average_length = float(max(lengths.mean(), 1.0))
    k1 = 1.5
    b = 0.75
    weighted = counts.astype(np.float32).tocsr(copy=True)
    for row in range(weighted.shape[0]):
        start, stop = weighted.indptr[row : row + 2]
        values = weighted.data[start:stop]
        denominator = values + k1 * (1.0 - b + b * lengths[row] / average_length)
        weighted.data[start:stop] = values * (k1 + 1.0) / denominator
    return weighted.multiply(inverse_document_frequency).tocsr()


def group_candidates(
    values: list[str], folds: np.ndarray, categories: np.ndarray, k: int
) -> list[list[int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (value, category) in enumerate(zip(values, categories)):
        if value:
            groups[(str(category), value)].append(index)
    result: list[list[int]] = []
    for index, (value, category) in enumerate(zip(values, categories)):
        donors = [
            other
            for other in groups.get((str(category), value), [])
            if int(folds[other]) != int(folds[index])
        ]
        result.append(sorted(donors)[:k])
    return result


def project_embeddings(embeddings: np.ndarray, dimension: int, seed: int) -> np.ndarray:
    if dimension <= 0 or dimension > embeddings.shape[1]:
        raise ValueError("invalid projection dimension")
    rng = np.random.default_rng(seed)
    projection = rng.normal(
        0.0, 1.0 / math.sqrt(dimension), size=(embeddings.shape[1], dimension)
    ).astype(np.float32)
    projected = embeddings.astype(np.float32) @ projection
    projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
    if not np.isfinite(projected).all():
        raise ValueError("projected embedding contains non-finite values")
    return projected


def add_fold_sparse_channels(
    channel_lists: dict[str, list[list[int]]],
    texts: list[str],
    folds: np.ndarray,
    categories: np.ndarray,
    k: int,
) -> None:
    for fold in FOLDS:
        for category in (BAD, FLAMMABLE):
            donors = np.flatnonzero((folds != fold) & (categories == category))
            queries = np.flatnonzero((folds == fold) & (categories == category))
            donor_texts = [texts[index] for index in donors]
            query_texts = [texts[index] for index in queries]
            tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=200_000,
                sublinear_tf=True,
                norm="l2",
            )
            donor_tfidf = tfidf.fit_transform(donor_texts).tocsr()
            query_tfidf = tfidf.transform(query_texts).tocsr()
            local_tfidf = sparse_top_k(query_tfidf, donor_tfidf, donors, k)

            counter = CountVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=200_000, binary=False
            )
            donor_counts = counter.fit_transform(donor_texts).tocsr()
            query_counts = counter.transform(query_texts).tocsr()
            donor_bm25 = bm25_documents(donor_counts)
            query_binary = query_counts.sign().astype(np.float32).tocsr()
            local_bm25 = sparse_top_k(query_binary, donor_bm25, donors, k)
            for position, query_index in enumerate(queries):
                channel_lists["tfidf"][int(query_index)] = local_tfidf[position]
                channel_lists["bm25"][int(query_index)] = local_bm25[position]


def add_fold_embedding_channel(
    channel_lists: dict[str, list[list[int]]],
    embeddings: np.ndarray,
    folds: np.ndarray,
    categories: np.ndarray,
    k: int,
) -> None:
    for fold in FOLDS:
        for category in (BAD, FLAMMABLE):
            donors = np.flatnonzero((folds != fold) & (categories == category))
            queries = np.flatnonzero((folds == fold) & (categories == category))
            local = dense_top_k(embeddings[queries], embeddings[donors], donors, k)
            for position, query_index in enumerate(queries):
                channel_lists["multimodal_embedding"][int(query_index)] = local[position]


def add_image_channels(
    channel_lists: dict[str, list[list[int]]],
    ids: np.ndarray,
    folds: np.ndarray,
    categories: np.ndarray,
    image_cache: Path,
    image_resolved_root: Path,
    topology_module,
    k: int,
) -> dict[str, Any]:
    fingerprints = []
    for position, row_id in enumerate(ids):
        path = topology_module.image_cache_path(image_cache, str(row_id))
        topology_module.validate_image_cache_entry(path, image_resolved_root)
        fingerprints.append(topology_module.image_fingerprints(path))
        if (position + 1) % 1000 == 0 or position + 1 == len(ids):
            print(f"oracle_image_fingerprints={position + 1}/{len(ids)}", flush=True)

    exact_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    phash_groups: dict[int, list[int]] = defaultdict(list)
    tree = topology_module.BKTree()
    minimum_std = float(topology_module.TOPOLOGY_THRESHOLDS["near_image"]["minimum_grayscale_std"])
    for index, item in enumerate(fingerprints):
        if float(item["grayscale_std"]) < minimum_std:
            continue
        exact_groups[(str(categories[index]), str(item["file_sha256"]))].append(index)
        phash_groups[int(item["phash64"])].append(index)
    for value in sorted(phash_groups):
        tree.add(value)

    phash_max = int(topology_module.TOPOLOGY_THRESHOLDS["near_image"]["phash_hamming_max"])
    dhash_max = int(topology_module.TOPOLOGY_THRESHOLDS["near_image"]["dhash_hamming_max"])
    for index, item in enumerate(fingerprints):
        exact = [
            donor
            for donor in exact_groups.get(
                (str(categories[index]), str(item["file_sha256"])), []
            )
            if int(folds[donor]) != int(folds[index])
        ]
        channel_lists["image_exact"][index] = sorted(exact)[:k]
        candidates: list[tuple[int, int, int]] = []
        if float(item["grayscale_std"]) >= minimum_std:
            for value in tree.query(int(item["phash64"]), phash_max):
                for donor in phash_groups[value]:
                    if (
                        donor == index
                        or categories[donor] != categories[index]
                        or int(folds[donor]) == int(folds[index])
                    ):
                        continue
                    phash_distance = (int(item["phash64"]) ^ int(fingerprints[donor]["phash64"])).bit_count()
                    dhash_distance = (int(item["dhash64"]) ^ int(fingerprints[donor]["dhash64"])).bit_count()
                    if dhash_distance <= dhash_max:
                        candidates.append((phash_distance, dhash_distance, donor))
        channel_lists["image_near"][index] = [
            donor for _, _, donor in sorted(set(candidates))[:k]
        ]
    return {
        "rows": len(fingerprints),
        "exact_groups": len(exact_groups),
        "unique_phashes": len(phash_groups),
        "thresholds": {
            "minimum_grayscale_std": minimum_std,
            "phash_hamming_max": phash_max,
            "dhash_hamming_max": dhash_max,
        },
    }


def fuse_channels(
    channel_lists: dict[str, list[list[int]]], row_count: int, k: int
) -> list[list[int]]:
    fused: list[list[int]] = []
    for query in range(row_count):
        scores: dict[int, float] = defaultdict(float)
        counts: Counter[int] = Counter()
        best_rank: dict[int, int] = {}
        for channel, rows in channel_lists.items():
            weight = float(CHANNEL_WEIGHTS[channel])
            for rank, donor in enumerate(rows[query], 1):
                scores[int(donor)] += weight / (RRF_OFFSET + rank)
                counts[int(donor)] += 1
                best_rank[int(donor)] = min(best_rank.get(int(donor), rank), rank)
        ordered = sorted(
            scores,
            key=lambda donor: (
                -scores[donor],
                -counts[donor],
                best_rank[donor],
                donor,
            ),
        )
        fused.append(ordered[:k])
    return fused


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    return 2 * tp / max(1, 2 * tp + fp + fn)


def metric_summary(
    labels: np.ndarray, categories: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {"categories": {}}
    scores = []
    for category in (BAD, FLAMMABLE):
        mask = categories == category
        local_labels = labels[mask]
        local_predictions = predictions[mask]
        score = f1(local_labels, local_predictions)
        scores.append(score)
        result["categories"][category] = {
            "rows": int(mask.sum()),
            "f1": score,
            "tp": int(np.sum((local_labels == 1) & (local_predictions == 1))),
            "fp": int(np.sum((local_labels == 0) & (local_predictions == 1))),
            "fn": int(np.sum((local_labels == 1) & (local_predictions == 0))),
            "tn": int(np.sum((local_labels == 0) & (local_predictions == 0))),
        }
    result["macro_f1"] = float(np.mean(scores))
    return result


def cohort_masks(
    labels: np.ndarray,
    categories: np.ndarray,
    baseline_after: np.ndarray,
    registry: list[dict[str, Any]],
    semantic_components: list[str],
) -> dict[str, np.ndarray]:
    errors = baseline_after != labels
    counts = Counter(
        (str(category), str(component))
        for category, component in zip(categories, semantic_components)
    )
    family_sizes = np.asarray(
        [counts[(str(category), str(component))] for category, component in zip(categories, semantic_components)],
        dtype=np.int32,
    )
    recurrence = np.asarray(
        [str(row["regime"]) == "recurrence_cross_fold" for row in registry]
    )
    return {
        "mixed_all_errors": errors,
        "recurrence_errors": errors & recurrence,
        "novel_family_errors": errors & ~recurrence,
        "singleton_errors": errors & (family_sizes == 1),
        "rare_le2_errors": errors & (family_sizes <= 2),
        "repeated_errors": errors & (family_sizes > 2),
        "bad_fp": errors & (categories == BAD) & (labels == 0),
        "bad_fn": errors & (categories == BAD) & (labels == 1),
        "flammable_fp": errors & (categories == FLAMMABLE) & (labels == 0),
        "flammable_fn": errors & (categories == FLAMMABLE) & (labels == 1),
    }


def retrieval_summary(
    candidates: list[list[int]], labels: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    positions = np.flatnonzero(mask)
    if not len(positions):
        return {"queries": 0, "recall_at": {}, "top1_accuracy": None}
    recall_at = {}
    for k in (1, 5, 10, 30):
        hits = [
            any(int(labels[donor]) == int(labels[index]) for donor in candidates[index][:k])
            for index in positions
        ]
        recall_at[str(k)] = float(np.mean(hits))
    useful_counts = np.asarray(
        [
            sum(int(labels[donor]) == int(labels[index]) for donor in candidates[index])
            for index in positions
        ],
        dtype=np.int32,
    )
    false_counts = np.asarray(
        [len(candidates[index]) - useful for index, useful in zip(positions, useful_counts)],
        dtype=np.int32,
    )
    return {
        "queries": int(len(positions)),
        "recall_at": recall_at,
        "top1_accuracy": recall_at["1"],
        "mean_useful_donors_at_30": float(useful_counts.mean()),
        "mean_false_donors_at_30": float(false_counts.mean()),
        "queries_with_no_candidates": int(
            sum(len(candidates[index]) == 0 for index in positions)
        ),
    }


def permutation_control(
    candidates: list[list[int]],
    labels: np.ndarray,
    folds: np.ndarray,
    categories: np.ndarray,
    masks: dict[str, np.ndarray],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {name: [] for name in masks}
    for _ in range(repetitions):
        permuted = labels.copy()
        for fold in FOLDS:
            for category in (BAD, FLAMMABLE):
                positions = np.flatnonzero((folds == fold) & (categories == category))
                permuted[positions] = permuted[rng.permutation(positions)]
        for name, mask in masks.items():
            positions = np.flatnonzero(mask)
            if not len(positions):
                continue
            hits = [
                any(int(permuted[donor]) == int(labels[index]) for donor in candidates[index])
                for index in positions
            ]
            values[name].append(float(np.mean(hits)))
    return {
        name: {
            "repetitions": len(local),
            "recall_at_30_mean": float(np.mean(local)) if local else None,
            "recall_at_30_std": float(np.std(local)) if local else None,
        }
        for name, local in values.items()
    }


def apply_frozen_annotator_prior(
    component_rows: list[dict[str, Any]], predictions: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(component_rows) != len(predictions):
        raise ValueError("annotator-prior row-count mismatch")
    result = predictions.copy()
    sources: Counter[str] = Counter()
    changes = 0
    for index, row in enumerate(component_rows):
        prior = row["ensemble"]["annotator_prior"]
        source = str(prior["source"])
        value = prior["value"]
        sources[source] += 1
        if source == "none":
            if value is not None:
                raise ValueError("none prior has a value")
            continue
        if source not in {"exact", "name"} or value not in {0, 1}:
            raise ValueError("unsupported frozen annotator prior")
        changes += int(result[index] != int(value))
        result[index] = int(value)
    return result, {
        "sources": dict(sorted(sources.items())),
        "overrides": int(
            sum(
                str(row["ensemble"]["annotator_prior"]["source"]) != "none"
                for row in component_rows
            )
        ),
        "prediction_changes": changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--topology-module", type=Path, required=True)
    parser.add_argument("--component-outputs", type=Path, required=True)
    parser.add_argument("--component-audit", type=Path, required=True)
    parser.add_argument("--fold-runtime", action="append", required=True)
    parser.add_argument("--multimodal-embeddings", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--image-resolved-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--projection-dimension", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args()
    if args.top_k != TOP_K:
        raise ValueError("oracle contract requires top_k=30")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite oracle output")

    started = time.monotonic()
    topology = json.loads(args.topology.read_text())
    topology_self = verify_self_hash(topology)
    if (
        topology.get("schema") != "exp699_family_leak_topology_v1"
        or topology.get("decision") != "TOPOLOGY_READY_FOR_METRICS"
        or topology.get("labels_read") != 0
        or topology.get("public_used") is not False
    ):
        raise ValueError("topology contract mismatch")
    topology_module = load_module(args.topology_module, "exp699_topology_for_oracle")
    runtime_paths = dict(parse_fold_path(value) for value in args.fold_runtime)
    runtime_rows = load_runtime_rows(runtime_paths)
    component_rows = load_component_outputs(args.component_outputs)
    registry, runtime_rows, component_rows = align_sources(
        topology, runtime_rows, component_rows
    )
    component_audit = json.loads(args.component_audit.read_text())
    component_audit_self = verify_existing_self_hash(component_audit)
    if (
        component_audit.get("schema") != "exp699_component_output_audit_v1"
        or component_audit.get("public_used") is not False
        or type(component_audit.get("gpu_used")) is not int
        or component_audit.get("gpu_used") != 0
        or type(component_audit.get("local_downloads")) is not int
        or component_audit.get("local_downloads") != 0
        or component_audit.get("rows") != len(component_rows)
    ):
        raise ValueError("component-output audit contract mismatch")

    row_count = len(registry)
    ids = np.asarray([str(row["id"]) for row in registry], dtype=str)
    folds = np.asarray([int(row["fold"]) for row in registry], dtype=np.int8)
    categories = np.asarray([str(row["category"]) for row in registry], dtype=str)
    texts = [
        topology_module.row_text(row, mask_digits=False) for row in runtime_rows
    ]
    normalized_texts = [
        topology_module.row_text(row, mask_digits=True) for row in runtime_rows
    ]
    component_ids = [str(row["component_id"]) for row in registry]

    channel_lists = {
        name: [[] for _ in range(row_count)] for name in CHANNEL_WEIGHTS
    }
    channel_lists["exact_text"] = group_candidates(texts, folds, categories, args.top_k)
    channel_lists["normalized_text"] = group_candidates(
        normalized_texts, folds, categories, args.top_k
    )
    channel_lists["graph_component"] = group_candidates(
        component_ids, folds, categories, args.top_k
    )
    add_fold_sparse_channels(channel_lists, texts, folds, categories, args.top_k)

    embedding_source = np.load(args.multimodal_embeddings, allow_pickle=False)
    embedding_ids = embedding_source["ids"].astype(str)
    if len(set(embedding_ids)) != len(embedding_ids):
        raise ValueError("multimodal embedding ids are duplicated")
    embedding_position = {value: index for index, value in enumerate(embedding_ids)}
    if any(value not in embedding_position for value in ids):
        raise ValueError("multimodal embeddings do not cover topology ids")
    embeddings = embedding_source["embeddings"][
        [embedding_position[value] for value in ids]
    ].astype(np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    projected = project_embeddings(
        embeddings, args.projection_dimension, args.projection_seed
    )
    add_fold_embedding_channel(
        channel_lists, projected, folds, categories, args.top_k
    )
    image_audit = add_image_channels(
        channel_lists,
        ids,
        folds,
        categories,
        args.image_cache,
        args.image_resolved_root,
        topology_module,
        args.top_k,
    )
    fused = fuse_channels(channel_lists, row_count, args.top_k)

    # Labels enter only after every label-free candidate list is frozen.
    labels = np.asarray([int(row["label"]) for row in component_rows], dtype=np.int8)
    semantic_components = [str(row["semantic_component"]) for row in component_rows]
    baseline_before = np.asarray(
        [
            int(row["ensemble"]["production_fixed_baseline_before_prior"]["prediction"])
            for row in component_rows
        ],
        dtype=np.int8,
    )
    baseline_after = np.asarray(
        [int(row["ensemble"]["production_fixed_baseline_after_prior"])
         for row in component_rows],
        dtype=np.int8,
    )
    masks = cohort_masks(
        labels, categories, baseline_after, registry, semantic_components
    )
    retrieval = {
        "fused_rrf": {
            name: retrieval_summary(fused, labels, mask) for name, mask in masks.items()
        },
        "channels": {
            channel: {
                name: retrieval_summary(rows, labels, mask)
                for name, mask in masks.items()
            }
            for channel, rows in channel_lists.items()
        },
    }
    retrieval["donor_label_permutation"] = permutation_control(
        fused,
        labels,
        folds,
        categories,
        masks,
        args.permutations,
        seed=20260828,
    )

    query_only_tfidf = np.asarray(
        [int(row["component_outputs"]["tfidf_text_rank"]["nested_prediction"])
         for row in component_rows],
        dtype=np.int8,
    )
    query_only = {
        name: {
            "queries": int(mask.sum()),
            "correct_on_solution140_errors": int(
                (mask & (query_only_tfidf == labels)).sum()
            ),
            "rescue_rate": (
                float((mask & (query_only_tfidf == labels)).sum() / mask.sum())
                if mask.any()
                else None
            ),
        }
        for name, mask in masks.items()
    }

    retrievable = np.zeros(row_count, dtype=bool)
    baseline_errors = baseline_after != labels
    for index in np.flatnonzero(baseline_errors):
        retrievable[index] = any(
            int(labels[donor]) == int(labels[index]) for donor in fused[index]
        )
    oracle_before = baseline_before.copy()
    oracle_before[retrievable] = labels[retrievable]
    replay_baseline_after, baseline_prior_audit = apply_frozen_annotator_prior(
        component_rows, baseline_before
    )
    if not np.array_equal(replay_baseline_after, baseline_after):
        raise ValueError("frozen solution140 production-prior replay mismatch")
    oracle_after, oracle_prior_audit = apply_frozen_annotator_prior(
        component_rows, oracle_before
    )
    direct_final_oracle = baseline_after.copy()
    direct_final_oracle[retrievable] = labels[retrievable]
    changed_before = oracle_before != baseline_before
    changed_after = oracle_after != baseline_after
    oracle = {
        "baseline_before_prior": metric_summary(labels, categories, baseline_before),
        "oracle_before_prior": metric_summary(labels, categories, oracle_before),
        "baseline_after_prior": metric_summary(labels, categories, baseline_after),
        "oracle_after_prior": metric_summary(labels, categories, oracle_after),
        "direct_final_upper_bound": metric_summary(
            labels, categories, direct_final_oracle
        ),
        "retrievable_solution140_errors": int(retrievable.sum()),
        "solution140_errors": int(baseline_errors.sum()),
        "changed_before_prior": int(changed_before.sum()),
        "survived_after_prior": int((changed_before & changed_after).sum()),
        "suppressed_by_prior": int((changed_before & ~changed_after).sum()),
        "already_correct_before_but_wrong_after_prior": int(
            (retrievable & (baseline_before == labels) & (baseline_after != labels)).sum()
        ),
        "after_prior_corrections": int(
            ((oracle_after == labels) & (baseline_after != labels)).sum()
        ),
        "after_prior_regressions": int(
            ((oracle_after != labels) & (baseline_after == labels)).sum()
        ),
        "baseline_prior_audit": baseline_prior_audit,
        "oracle_prior_audit": oracle_prior_audit,
    }
    oracle["macro_delta_after_prior"] = (
        oracle["oracle_after_prior"]["macro_f1"]
        - oracle["baseline_after_prior"]["macro_f1"]
    )
    oracle["macro_delta_direct_final_upper_bound"] = (
        oracle["direct_final_upper_bound"]["macro_f1"]
        - oracle["baseline_after_prior"]["macro_f1"]
    )

    payload = {
        "schema": "exp699_graph_aware_hybrid_retrieval_oracle_v1",
        "experiment": 699,
        "stage": "cpu_first_oracle_no_model_training",
        "decision": "ORACLE_METRICS_READY",
        "public_used": False,
        "sealed_rows": 0,
        "candidate_generation_labels_read": 0,
        "labels_entered_after_candidate_lists_frozen": True,
        "rows": row_count,
        "top_k": args.top_k,
        "channels": list(CHANNEL_WEIGHTS),
        "channel_weights": CHANNEL_WEIGHTS,
        "rrf_offset": RRF_OFFSET,
        "embedding_projection": {
            "dimension": args.projection_dimension,
            "seed": args.projection_seed,
            "source_dimensions": int(embeddings.shape[1]),
        },
        "image_audit": image_audit,
        "retrieval": retrieval,
        "query_only_tfidf_control": query_only,
        "oracle": oracle,
        "source_bindings": {
            "topology_file_sha256": sha256_file(args.topology),
            "topology_self_sha256": topology_self,
            "topology_module_sha256": sha256_file(args.topology_module),
            "component_outputs_sha256": sha256_file(args.component_outputs),
            "component_audit_file_sha256": sha256_file(args.component_audit),
            "component_audit_self_sha256": component_audit_self,
            "frozen_production_prior_source_sha256": component_audit["provenance"]["prior"],
            "multimodal_embeddings_sha256": sha256_file(args.multimodal_embeddings),
            "fold_runtime_sha256": {
                str(fold): sha256_file(path) for fold, path in runtime_paths.items()
            },
        },
        "runtime_seconds": time.monotonic() - started,
        "limitations": [
            "This is an oracle upper bound: true query labels are used only after label-free candidate retrieval is frozen.",
            "The 2048-dimensional accepted multimodal vectors use a fixed label-free Gaussian projection for CPU retrieval.",
            "Oracle viability does not authorize a hard kNN override or Public submission.",
        ],
    }
    file_sha, self_sha = write_self_hashed(args.output, payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "file_sha256": file_sha,
                "self_sha256": self_sha,
                "runtime_seconds": payload["runtime_seconds"],
                "solution140_errors": oracle["solution140_errors"],
                "retrievable_errors": oracle["retrievable_solution140_errors"],
                "macro_delta_after_prior": oracle["macro_delta_after_prior"],
                "flammable_fn_recall_at_30": retrieval["fused_rrf"]["flammable_fn"]["recall_at"].get("30"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

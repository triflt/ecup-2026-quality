from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"
FOLDS = (0, 1, 2, 3, 4)
SCREEN_FOLDS = (0, 3)
CHANNELS = (
    "exact_text",
    "normalized_text",
    "tfidf",
    "bm25",
    "qwen_embedding",
    "image_exact",
    "image_near",
)
CHANNEL_WEIGHTS = {
    "exact_text": 2.0,
    "normalized_text": 1.5,
    "tfidf": 1.0,
    "bm25": 1.0,
    "qwen_embedding": 1.0,
    "image_exact": 2.0,
    "image_near": 1.0,
}
TOP_K = 30
RRF_OFFSET = 60
PAIR_PROBABILITY_MINIMUM = 0.8
CACHE_TOP_K = 5
MINIMUM_DONORS = 3
DONOR_CONSENSUS_MINIMUM = 0.9
CACHE_ALPHA = 0.15
QUERY_NEGATIVE_STRONG = 0.2
QUERY_POSITIVE_STRONG = 0.8
FLAMMABLE_THRESHOLD = 0.953912615776062
PERMUTATION_SEED = 42
PREREGISTER_SELF_SHA256 = "d1358a630c5a83aa4765cef764701400182d40b537e7e4706c2259920caea689"

POSITIVE_CUES = {
    "fuel": re.compile(
        r"\b(?:топлив\w*|бензин\w*|дизел\w*|керосин\w*|горюч\w*|жидкост\w*|масл\w*)\b"
    ),
    "gas": re.compile(r"\b(?:газ\w*|пропан\w*|бутан\w*|аэрозол\w*|баллон\w*)\b"),
    "included": re.compile(r"\b(?:комплект\w*|включен\w*|содержит\w*|заправлен\w*|наполнен\w*)\b"),
}
NEGATIVE_CUES = {
    "empty_equipment": re.compile(
        r"(?:без\s+(?:топлив\w*|газ\w*|жидкост\w*)|\bпуст\w*\b|\bнезаправ\w*\b|только\s+устройств\w*)"
    ),
    "component_only": re.compile(
        r"\b(?:запчаст\w*|детал\w*|компонент\w*|насадк\w*|клапан\w*|шланг\w*|переходник\w*|корпус\w*|крышк\w*)\b"
    ),
    "packaging": re.compile(r"\b(?:упаковк\w*|тар\w*|футляр\w*|контейнер\w*|чехол\w*|коробк\w*)\b"),
}


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_self_hash(payload: dict[str, Any]) -> str:
    copied = dict(payload)
    declared = str(copied.pop("self_sha256", ""))
    if copied.get("self_hash_algorithm") != "sha256_canonical_json_without_self_sha256":
        raise ValueError("unsupported self-hash algorithm")
    if canonical_sha256(copied) != declared:
        raise ValueError("self-hash mismatch")
    return declared


def verify_legacy_self_hash(payload: dict[str, Any]) -> str:
    """Verify the accepted legacy compact-contract format without an algorithm field."""
    copied = dict(payload)
    declared = str(copied.pop("self_sha256", ""))
    if "self_hash_algorithm" in copied:
        raise ValueError("legacy contract unexpectedly declares self-hash algorithm")
    if canonical_sha256(copied) != declared:
        raise ValueError("legacy self-hash mismatch")
    return declared


def write_self_hashed(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    payload = dict(payload)
    payload["self_hash_algorithm"] = "sha256_canonical_json_without_self_sha256"
    payload["self_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return sha256_file(path), str(payload["self_sha256"])


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


def verify_source_manifest(
    manifest_path: Path,
    preregister_path: Path,
    *,
    topology_path: Path,
    topology_module_path: Path,
    component_outputs_path: Path,
    component_audit_path: Path,
    embeddings_path: Path,
    v1_report_path: Path,
    runtime_paths: dict[int, Path],
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    manifest = json.loads(manifest_path.read_text())
    manifest_self = verify_self_hash(manifest)
    preregister = json.loads(preregister_path.read_text())
    preregister_self = verify_self_hash(preregister)
    if (
        manifest.get("schema") != "exp699_retrieval_source_manifest_v1"
        or manifest.get("decision") != "ACCEPT_INPUTS_FOR_DIAGNOSTIC_AND_PACKAGE_PARITY_BUILD"
        or manifest.get("public_used") is not False
    ):
        raise ValueError("source-manifest scope mismatch")
    if (
        preregister_self != PREREGISTER_SELF_SHA256
        or preregister.get("schema") != "exp699_retrieval_pair_verifier_preregister_v1"
        or preregister.get("screen", {}).get("folds") != [0, 3]
        or preregister.get("screen", {}).get("public_used") is not False
        or preregister.get("screen", {}).get("sealed_used") is not False
        or preregister.get("source_bindings", {}).get("retrieval_source_manifest_self_sha256")
        != manifest_self
    ):
        raise ValueError("frozen preregistration mismatch")
    expected = preregister["source_bindings"]
    if sha256_file(component_outputs_path) != expected["component_outputs_sha256"]:
        raise ValueError("component-output preregistration mismatch")
    if sha256_file(topology_path) != expected["topology_sha256"]:
        raise ValueError("topology preregistration mismatch")
    if sha256_file(embeddings_path) != expected["qwen_train_embeddings_sha256"]:
        raise ValueError("Qwen embedding preregistration mismatch")
    if sha256_file(v1_report_path) != expected["soft_cache_v1_evaluation_sha256"]:
        raise ValueError("soft-cache-v1 preregistration mismatch")
    if sha256_file(topology_module_path) != manifest["topology"]["module_sha256"]:
        raise ValueError("topology-module source mismatch")
    if sha256_file(component_audit_path) != manifest["component_outputs"]["audit_file_sha256"]:
        raise ValueError("component-audit source mismatch")
    if sha256_file(component_outputs_path) != manifest["component_outputs"]["file_sha256"]:
        raise ValueError("component-output source mismatch")
    if sha256_file(topology_path) != manifest["topology"]["file_sha256"]:
        raise ValueError("topology source mismatch")
    expected_runtime = manifest.get("fold_runtime_sha256")
    if not isinstance(expected_runtime, dict) or set(expected_runtime) != {
        str(fold) for fold in FOLDS
    }:
        raise ValueError("source-contract fold inventory mismatch")
    if set(runtime_paths) != set(FOLDS):
        raise ValueError("exact folds0..4 are required")
    for fold, path in runtime_paths.items():
        if sha256_file(path) != expected_runtime[str(fold)]:
            raise ValueError(f"source-manifest fold{fold} mismatch")
    return manifest, manifest_self, preregister, preregister_self


def load_runtime_rows(paths: dict[int, Path]) -> list[dict[str, Any]]:
    if set(paths) != set(FOLDS):
        raise ValueError("exact folds0..4 are required")
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        with paths[fold].open() as stream:
            local = [json.loads(line) for line in stream]
        if any(int(row["fold"]) != fold for row in local):
            raise ValueError(f"fold{fold} runtime binding mismatch")
        rows.extend(local)
    rows.sort(key=lambda row: int(row["global_index"]))
    if [int(row["global_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("runtime global-index coverage mismatch")
    return rows


def load_component_rows(path: Path) -> list[dict[str, Any]]:
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
    if len(registry) != len(runtime_rows) or len(component_rows) != len(runtime_rows):
        raise ValueError("source row-count mismatch")
    component_by_key = {
        (str(row["id"]), int(row["fold"]), str(row["category"])): row for row in component_rows
    }
    aligned: list[dict[str, Any]] = []
    for expected_index, (registry_row, runtime_row) in enumerate(zip(registry, runtime_rows)):
        if int(registry_row["global_index"]) != expected_index:
            raise ValueError("topology global-index coverage mismatch")
        registry_key = (
            str(registry_row["id"]),
            int(registry_row["fold"]),
            str(registry_row["category"]),
        )
        runtime_key = (
            str(runtime_row["id"]),
            int(runtime_row["fold"]),
            str(runtime_row["category"]),
        )
        if registry_key != runtime_key:
            raise ValueError("topology/runtime binding mismatch")
        row = component_by_key.get(registry_key)
        if row is None or int(row.get("global_index", expected_index)) != expected_index:
            raise ValueError("component-output binding mismatch")
        aligned.append(row)
    if len(aligned) != len(component_by_key):
        raise ValueError("component-output exact coverage mismatch")
    return registry, runtime_rows, aligned


def cue_vector(text: str) -> np.ndarray:
    values = [bool(pattern.search(text)) for pattern in POSITIVE_CUES.values()]
    values.extend(bool(pattern.search(text)) for pattern in NEGATIVE_CUES.values())
    return np.asarray(values, dtype=np.float64)


def cue_mismatches(query: np.ndarray, donor: np.ndarray) -> np.ndarray:
    query_positive = query[:3]
    query_negative = query[3:]
    donor_positive = donor[:3]
    donor_negative = donor[3:]
    forward = np.outer(query_positive, donor_negative).reshape(-1)
    reverse = np.outer(donor_positive, query_negative).reshape(-1)
    return np.concatenate([forward, reverse]).astype(np.float64)


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


def eligible_donors(
    donor_pool: np.ndarray,
    query_index: int,
    folds: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    selected = donor_pool[
        (folds[donor_pool] != folds[query_index])
        & (components[donor_pool] != components[query_index])
        & (donor_pool != query_index)
    ]
    return selected.astype(np.int64, copy=False)


def _ordered_top(
    donors: np.ndarray,
    scores: np.ndarray,
    limit: int,
    *,
    positive_only: bool,
) -> list[tuple[int, float]]:
    if positive_only:
        keep = np.flatnonzero(scores > 0.0)
        donors = donors[keep]
        scores = scores[keep]
    if not len(donors):
        return []
    if not np.isfinite(scores).all():
        raise ValueError("retrieval score is non-finite")
    ordered = sorted(
        range(len(donors)), key=lambda local: (-float(scores[local]), int(donors[local]))
    )[:limit]
    return [(int(donors[local]), float(scores[local])) for local in ordered]


def _add_channel(
    channels: dict[str, dict[int, list[dict[str, Any]]]],
    channel: str,
    query: int,
    rows: Iterable[tuple[int, float, dict[str, Any]]],
) -> None:
    channels[channel][query] = [
        {"donor": int(donor), "score": float(score), **extra} for donor, score, extra in rows
    ]


def build_candidates(
    *,
    outer_fold: int,
    texts: list[str],
    normalized_texts: list[str],
    folds: np.ndarray,
    categories: np.ndarray,
    components: np.ndarray,
    embeddings: np.ndarray,
    fingerprints: list[dict[str, Any]],
    topology_module,
) -> tuple[dict[int, list[int]], dict[int, dict[int, dict[str, Any]]]]:
    flammable = categories == FLAMMABLE
    donor_pool = np.flatnonzero(flammable & (folds != outer_fold))
    query_pool = np.flatnonzero(flammable)
    if not len(donor_pool) or not len(query_pool):
        raise ValueError("flammable query/donor pool is empty")
    channels: dict[str, dict[int, list[dict[str, Any]]]] = {name: {} for name in CHANNELS}

    exact_groups: dict[str, list[int]] = defaultdict(list)
    normalized_groups: dict[str, list[int]] = defaultdict(list)
    image_exact_groups: dict[str, list[int]] = defaultdict(list)
    phash_groups: dict[int, list[int]] = defaultdict(list)
    for donor in donor_pool:
        exact_groups[texts[donor]].append(int(donor))
        normalized_groups[normalized_texts[donor]].append(int(donor))
        fingerprint = fingerprints[donor]
        if float(fingerprint["grayscale_std"]) >= float(
            topology_module.TOPOLOGY_THRESHOLDS["near_image"]["minimum_grayscale_std"]
        ):
            image_exact_groups[str(fingerprint["file_sha256"])].append(int(donor))
            phash_groups[int(fingerprint["phash64"])].append(int(donor))

    tree = topology_module.BKTree()
    for value in sorted(phash_groups):
        tree.add(value)
    phash_max = int(topology_module.TOPOLOGY_THRESHOLDS["near_image"]["phash_hamming_max"])
    dhash_max = int(topology_module.TOPOLOGY_THRESHOLDS["near_image"]["dhash_hamming_max"])
    minimum_std = float(topology_module.TOPOLOGY_THRESHOLDS["near_image"]["minimum_grayscale_std"])

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=200_000,
        sublinear_tf=True,
        norm="l2",
    )
    tfidf.fit([texts[index] for index in donor_pool])
    all_tfidf = tfidf.transform(texts).tocsr()
    counter = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, binary=False)
    counter.fit([texts[index] for index in donor_pool])
    all_counts = counter.transform(texts).tocsr()
    donor_bm25_all = bm25_documents(all_counts[donor_pool])
    donor_bm25_position = {int(index): local for local, index in enumerate(donor_pool)}

    for query in query_pool:
        donors = eligible_donors(donor_pool, int(query), folds, components)
        donor_set = {int(value) for value in donors}
        exact = [donor for donor in exact_groups.get(texts[query], []) if donor in donor_set][
            :TOP_K
        ]
        normalized = [
            donor
            for donor in normalized_groups.get(normalized_texts[query], [])
            if donor in donor_set
        ][:TOP_K]
        _add_channel(
            channels,
            "exact_text",
            int(query),
            ((donor, 1.0, {}) for donor in exact),
        )
        _add_channel(
            channels,
            "normalized_text",
            int(query),
            ((donor, 1.0, {}) for donor in normalized),
        )

        tfidf_scores = (all_tfidf[query] @ all_tfidf[donors].T).toarray().reshape(-1)
        _add_channel(
            channels,
            "tfidf",
            int(query),
            (
                (donor, score, {})
                for donor, score in _ordered_top(donors, tfidf_scores, TOP_K, positive_only=True)
            ),
        )
        bm25_local = np.asarray(
            [donor_bm25_position[int(donor)] for donor in donors], dtype=np.int64
        )
        bm25_scores = (
            (all_counts[query].sign().astype(np.float32) @ donor_bm25_all[bm25_local].T)
            .toarray()
            .reshape(-1)
        )
        _add_channel(
            channels,
            "bm25",
            int(query),
            (
                (donor, score, {})
                for donor, score in _ordered_top(donors, bm25_scores, TOP_K, positive_only=True)
            ),
        )
        qwen_scores = embeddings[query] @ embeddings[donors].T
        _add_channel(
            channels,
            "qwen_embedding",
            int(query),
            (
                (donor, score, {})
                for donor, score in _ordered_top(donors, qwen_scores, TOP_K, positive_only=False)
            ),
        )

        query_fingerprint = fingerprints[query]
        exact_image = [
            donor
            for donor in image_exact_groups.get(str(query_fingerprint["file_sha256"]), [])
            if donor in donor_set
        ][:TOP_K]
        _add_channel(
            channels,
            "image_exact",
            int(query),
            ((donor, 1.0, {"phash_distance": 0, "dhash_distance": 0}) for donor in exact_image),
        )
        near: list[tuple[int, float, dict[str, Any]]] = []
        if float(query_fingerprint["grayscale_std"]) >= minimum_std:
            for value in tree.query(int(query_fingerprint["phash64"]), phash_max):
                for donor in phash_groups[value]:
                    if donor not in donor_set:
                        continue
                    phash_distance = (
                        int(query_fingerprint["phash64"]) ^ int(fingerprints[donor]["phash64"])
                    ).bit_count()
                    dhash_distance = (
                        int(query_fingerprint["dhash64"]) ^ int(fingerprints[donor]["dhash64"])
                    ).bit_count()
                    if dhash_distance <= dhash_max:
                        score = 1.0 - 0.5 * (
                            phash_distance / max(1, phash_max) + dhash_distance / max(1, dhash_max)
                        )
                        near.append(
                            (
                                donor,
                                score,
                                {
                                    "phash_distance": phash_distance,
                                    "dhash_distance": dhash_distance,
                                },
                            )
                        )
        near.sort(
            key=lambda item: (
                -item[1],
                item[2]["phash_distance"],
                item[2]["dhash_distance"],
                item[0],
            )
        )
        _add_channel(channels, "image_near", int(query), near[:TOP_K])

    fused: dict[int, list[int]] = {}
    evidence: dict[int, dict[int, dict[str, Any]]] = {}
    for query in query_pool:
        rrf: dict[int, float] = defaultdict(float)
        details: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"channels": {}, "phash_distance": None, "dhash_distance": None}
        )
        for channel in CHANNELS:
            for rank, row in enumerate(channels[channel][int(query)], 1):
                donor = int(row["donor"])
                rrf[donor] += CHANNEL_WEIGHTS[channel] / (RRF_OFFSET + rank)
                details[donor]["channels"][channel] = {
                    "rank": rank,
                    "score": float(row["score"]),
                }
                if channel in {"image_exact", "image_near"}:
                    for key in ("phash_distance", "dhash_distance"):
                        value = row.get(key)
                        old = details[donor].get(key)
                        if value is not None and (old is None or int(value) < int(old)):
                            details[donor][key] = int(value)
        ordered = sorted(
            rrf,
            key=lambda donor: (
                -rrf[donor],
                -len(details[donor]["channels"]),
                donor,
            ),
        )[:TOP_K]
        fused[int(query)] = ordered
        evidence[int(query)] = {}
        for donor in ordered:
            evidence[int(query)][donor] = {
                **details[donor],
                "rrf_score": float(rrf[donor]),
                "channel_count": len(details[donor]["channels"]),
            }
            if folds[donor] == folds[query] or components[donor] == components[query]:
                raise ValueError("connected-safe donor exclusion failed")
    return fused, evidence


def component_statistics(
    components: np.ndarray,
    labels: np.ndarray,
    donor_pool: np.ndarray,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in donor_pool:
        grouped[str(components[index])].append(int(labels[index]))
    result: dict[str, dict[str, float]] = {}
    for component, values in grouped.items():
        positive_rate = float(np.mean(values))
        agreement = max(positive_rate, 1.0 - positive_rate)
        result[component] = {
            "count": float(len(values)),
            "agreement": agreement,
            "reliability": 2.0 * abs(positive_rate - 0.5),
        }
    return result


def query_feature(
    index: int,
    baseline_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    cues: np.ndarray,
    texts: list[str],
    fingerprints: list[dict[str, Any]],
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(
                [
                    float(baseline_scores[index]),
                    float(baseline_predictions[index]),
                    math.log1p(len(texts[index])),
                    math.log1p(len(texts[index].split())),
                    float(fingerprints[index]["grayscale_std"] >= 8.0),
                ],
                dtype=np.float64,
            ),
            cues[index].astype(np.float64),
        ]
    )


def pair_feature(
    query: int,
    donor: int,
    *,
    evidence: dict[str, Any],
    donor_labels: np.ndarray,
    component_stats: dict[str, dict[str, float]],
    components: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    cues: np.ndarray,
    texts: list[str],
    fingerprints: list[dict[str, Any]],
) -> np.ndarray:
    query_side = query_feature(
        query, baseline_scores, baseline_predictions, cues, texts, fingerprints
    )
    channel_features: list[float] = []
    for channel in CHANNELS:
        row = evidence["channels"].get(channel)
        channel_features.extend(
            [
                0.0 if row is None else 1.0 / float(row["rank"]),
                0.0 if row is None else float(row["score"]),
            ]
        )
    donor_component = str(components[donor])
    stats = component_stats[donor_component]
    pair_side = np.asarray(
        [
            float(donor_labels[donor]),
            1.0,  # Both rows are in the frozen flammable category route.
            float(np.mean(cues[query] == cues[donor])),
            float(evidence["rrf_score"]),
            float(evidence["channel_count"]),
            0.0
            if evidence.get("phash_distance") is None
            else float(evidence["phash_distance"]) / 64.0,
            0.0
            if evidence.get("dhash_distance") is None
            else float(evidence["dhash_distance"]) / 64.0,
            math.log1p(stats["count"]),
            stats["agreement"],
            stats["reliability"],
            math.log1p(len(texts[donor])),
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        [
            query_side,
            cues[donor].astype(np.float64),
            cue_mismatches(cues[query], cues[donor]),
            np.asarray(channel_features, dtype=np.float64),
            pair_side,
        ]
    )


def make_logistic(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def fit_query_model(
    train_queries: np.ndarray,
    *,
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    cues: np.ndarray,
    texts: list[str],
    fingerprints: list[dict[str, Any]],
    seed: int,
) -> Pipeline:
    matrix = np.vstack(
        [
            query_feature(
                int(index),
                baseline_scores,
                baseline_predictions,
                cues,
                texts,
                fingerprints,
            )
            for index in train_queries
        ]
    )
    target = labels[train_queries]
    if {int(value) for value in target} != {0, 1}:
        raise ValueError("query-only training requires both labels")
    model = make_logistic(seed)
    model.fit(matrix, target)
    return model


def fit_pair_model(
    train_queries: np.ndarray,
    *,
    fused: dict[int, list[int]],
    evidence: dict[int, dict[int, dict[str, Any]]],
    query_labels: np.ndarray,
    donor_labels: np.ndarray,
    component_stats: dict[str, dict[str, float]],
    components: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    cues: np.ndarray,
    texts: list[str],
    fingerprints: list[dict[str, Any]],
    seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    features: list[np.ndarray] = []
    targets: list[int] = []
    weights: list[float] = []
    used_queries = 0
    for query in train_queries:
        donors = fused[int(query)]
        if not donors:
            continue
        used_queries += 1
        query_weight = 1.0 / len(donors)
        for donor in donors:
            features.append(
                pair_feature(
                    int(query),
                    donor,
                    evidence=evidence[int(query)][donor],
                    donor_labels=donor_labels,
                    component_stats=component_stats,
                    components=components,
                    baseline_scores=baseline_scores,
                    baseline_predictions=baseline_predictions,
                    cues=cues,
                    texts=texts,
                    fingerprints=fingerprints,
                )
            )
            targets.append(int(donor_labels[donor] == query_labels[query]))
            weights.append(query_weight)
    if not features or set(targets) != {0, 1}:
        raise ValueError("pair training requires non-empty positive and negative pairs")
    model = make_logistic(seed)
    model.fit(np.vstack(features), np.asarray(targets), model__sample_weight=np.asarray(weights))
    return model, {
        "queries": used_queries,
        "pairs": len(features),
        "positive_pairs": int(sum(targets)),
        "negative_pairs": int(len(targets) - sum(targets)),
        "per_query_sample_weight": True,
        "class_weight": "balanced",
    }


def query_probabilities(
    model: Pipeline,
    queries: np.ndarray,
    *,
    baseline_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    cues: np.ndarray,
    texts: list[str],
    fingerprints: list[dict[str, Any]],
) -> dict[int, float]:
    matrix = np.vstack(
        [
            query_feature(
                int(index),
                baseline_scores,
                baseline_predictions,
                cues,
                texts,
                fingerprints,
            )
            for index in queries
        ]
    )
    probabilities = model.predict_proba(matrix)[:, 1]
    return {int(index): float(value) for index, value in zip(queries, probabilities)}


def apply_query_control(
    baseline: np.ndarray, queries: np.ndarray, probabilities: dict[int, float]
) -> np.ndarray:
    candidate = baseline.copy()
    for query in queries:
        probability = probabilities[int(query)]
        if probability <= QUERY_NEGATIVE_STRONG:
            candidate[query] = 0
        elif probability >= QUERY_POSITIVE_STRONG:
            candidate[query] = 1
    return candidate


def apply_pair_cache(
    *,
    queries: np.ndarray,
    pair_model: Pipeline,
    fused: dict[int, list[int]],
    evidence: dict[int, dict[int, dict[str, Any]]],
    donor_labels: np.ndarray,
    component_stats: dict[str, dict[str, float]],
    components: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    query_probabilities_by_index: dict[int, float],
    cues: np.ndarray,
    texts: list[str],
    fingerprints: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    scores = baseline_scores.copy()
    predictions = baseline_predictions.copy()
    audit: list[dict[str, Any]] = []
    for query in queries:
        donors = fused[int(query)]
        if not donors:
            continue
        matrix = np.vstack(
            [
                pair_feature(
                    int(query),
                    donor,
                    evidence=evidence[int(query)][donor],
                    donor_labels=donor_labels,
                    component_stats=component_stats,
                    components=components,
                    baseline_scores=baseline_scores,
                    baseline_predictions=baseline_predictions,
                    cues=cues,
                    texts=texts,
                    fingerprints=fingerprints,
                )
                for donor in donors
            ]
        )
        compatibility = pair_model.predict_proba(matrix)[:, 1]
        compatible = [
            (donor, float(probability), rank)
            for rank, (donor, probability) in enumerate(zip(donors, compatibility), 1)
            if probability >= PAIR_PROBABILITY_MINIMUM
        ]
        compatible.sort(key=lambda item: (-item[1], item[2], item[0]))
        selected = compatible[:CACHE_TOP_K]
        if len(selected) < MINIMUM_DONORS:
            continue
        selected_donors = [row[0] for row in selected]
        donor_values = donor_labels[selected_donors]
        selected_weights = np.asarray([row[1] for row in selected], dtype=np.float64)
        positive_rate = float(np.average(donor_values, weights=selected_weights))
        consensus = max(positive_rate, 1.0 - positive_rate)
        if consensus < DONOR_CONSENSUS_MINIMUM:
            continue

        exact_donors = [
            donor
            for donor in selected_donors
            if "exact_text" in evidence[int(query)][donor]["channels"]
            and "image_exact" in evidence[int(query)][donor]["channels"]
        ]
        hard_override = (
            len(exact_donors) >= 2
            and len({int(donor_labels[donor]) for donor in exact_donors}) == 1
            and all(
                component_stats[str(components[donor])]["agreement"] == 1.0
                for donor in exact_donors
            )
        )
        old_prediction = int(baseline_predictions[query])
        old_score = float(baseline_scores[query])
        query_probability = query_probabilities_by_index[int(query)]
        boundary_authorized = True
        if hard_override:
            new_prediction = int(donor_labels[exact_donors[0]])
            new_score = float(new_prediction)
        else:
            new_score = (1.0 - CACHE_ALPHA) * old_score + CACHE_ALPHA * positive_rate
            new_prediction = int(new_score >= FLAMMABLE_THRESHOLD)
            if new_prediction != old_prediction:
                boundary_authorized = (
                    old_prediction == 1
                    and new_prediction == 0
                    and query_probability <= QUERY_NEGATIVE_STRONG
                ) or (
                    old_prediction == 0
                    and new_prediction == 1
                    and query_probability >= QUERY_POSITIVE_STRONG
                )
        if boundary_authorized:
            scores[query] = new_score
            predictions[query] = new_prediction
        audit.append(
            {
                "global_index": int(query),
                "selected_donors": [int(value) for value in selected_donors],
                "compatibility": [float(row[1]) for row in selected],
                "donor_positive_rate": positive_rate,
                "donor_consensus": consensus,
                "query_probability": query_probability,
                "hard_override": hard_override,
                "boundary_authorized": boundary_authorized,
                "baseline_prediction": old_prediction,
                "candidate_prediction": int(predictions[query]),
                "baseline_score": old_score,
                "candidate_score": float(scores[query]),
            }
        )
    return scores, predictions, audit


def deterministic_donor_permutation(
    labels: np.ndarray,
    folds: np.ndarray,
    categories: np.ndarray,
    donor_pool: np.ndarray,
    seed: int,
) -> np.ndarray:
    permuted = labels.copy()
    rng = np.random.default_rng(seed)
    for fold in FOLDS:
        for category in (BAD, FLAMMABLE):
            positions = donor_pool[
                (folds[donor_pool] == fold) & (categories[donor_pool] == category)
            ]
            if len(positions):
                permuted[positions] = labels[rng.permutation(positions)]
    return permuted


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    return 2.0 * tp / max(1, 2 * tp + fp + fn)


def metric_summary(
    labels: np.ndarray,
    categories: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": int(mask.sum()), "categories": {}}
    scores: list[float] = []
    for category in (BAD, FLAMMABLE):
        local = mask & (categories == category)
        local_labels = labels[local]
        local_predictions = predictions[local]
        score = f1(local_labels, local_predictions)
        scores.append(score)
        result["categories"][category] = {
            "rows": int(local.sum()),
            "f1": score,
            "tp": int(np.sum((local_labels == 1) & (local_predictions == 1))),
            "fp": int(np.sum((local_labels == 0) & (local_predictions == 1))),
            "fn": int(np.sum((local_labels == 1) & (local_predictions == 0))),
            "tn": int(np.sum((local_labels == 0) & (local_predictions == 0))),
        }
    result["macro_f1"] = float(np.mean(scores))
    return result


def comparison(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, int]:
    changed = mask & (baseline != candidate)
    corrections = changed & (candidate == labels)
    regressions = changed & (candidate != labels)
    return {
        "changed": int(changed.sum()),
        "corrections": int(corrections.sum()),
        "regressions": int(regressions.sum()),
        "net": int(corrections.sum() - regressions.sum()),
    }


def survival(
    baseline_before: np.ndarray,
    candidate_before: np.ndarray,
    baseline_after: np.ndarray,
    candidate_after: np.ndarray,
    mask: np.ndarray,
) -> dict[str, int]:
    changed_before = mask & (baseline_before != candidate_before)
    changed_after = mask & (baseline_after != candidate_after)
    return {
        "changed_before_prior": int(changed_before.sum()),
        "survived_after_prior": int((changed_before & changed_after).sum()),
        "suppressed_by_prior": int((changed_before & ~changed_after).sum()),
        "new_after_prior": int((~changed_before & changed_after & mask).sum()),
    }


def load_v1_harmful_indices(path: Path, expected: int) -> tuple[dict[int, dict[str, Any]], str]:
    report = json.loads(path.read_text())
    self_sha = verify_self_hash(report)
    if report.get("schema") != "exp699_soft_cache_exploratory_oof_v1":
        raise ValueError("v1 report schema mismatch")
    harmful: dict[int, dict[str, Any]] = {}
    for row in report.get("changed_decisions", []):
        if int(row["baseline_after"]) != int(row["candidate_after"]):
            harmful[int(row["global_index"])] = row
    if len(harmful) != expected:
        raise ValueError(f"v1 harmful cohort must contain exactly {expected} rows")
    return harmful, self_sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--topology-module", type=Path, required=True)
    parser.add_argument("--component-outputs", type=Path, required=True)
    parser.add_argument("--component-audit", type=Path, required=True)
    parser.add_argument("--fold-runtime", action="append", required=True)
    parser.add_argument("--multimodal-embeddings", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--image-resolved-root", type=Path, required=True)
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--expected-v1-harmful", type=int, default=28)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite pair-verifier output")

    oracle_path = Path(__file__).resolve().parent / "evaluate_retrieval_oracle.py"
    runtime_paths = dict(parse_fold_path(value) for value in args.fold_runtime)
    source_manifest, source_manifest_self, preregister, preregister_self = verify_source_manifest(
        args.source_manifest,
        args.preregister,
        topology_path=args.topology,
        topology_module_path=args.topology_module,
        component_outputs_path=args.component_outputs,
        component_audit_path=args.component_audit,
        embeddings_path=args.multimodal_embeddings,
        v1_report_path=args.v1_report,
        runtime_paths=runtime_paths,
    )

    topology = json.loads(args.topology.read_text())
    topology_self = verify_self_hash(topology)
    if (
        topology.get("schema") != "exp699_family_leak_topology_v1"
        or topology.get("decision") != "TOPOLOGY_READY_FOR_METRICS"
        or topology.get("labels_read") != 0
        or topology.get("public_used") is not False
        or topology.get("sealed_rows") != 0
    ):
        raise ValueError("topology contract mismatch")
    topology_module = load_module(args.topology_module, "exp699_pair_topology")
    oracle = load_module(oracle_path, "exp699_pair_oracle")
    runtime_rows = load_runtime_rows(runtime_paths)
    component_rows = load_component_rows(args.component_outputs)
    registry, runtime_rows, component_rows = align_sources(topology, runtime_rows, component_rows)
    component_audit = json.loads(args.component_audit.read_text())
    component_audit_self = verify_legacy_self_hash(component_audit)
    if (
        component_audit.get("schema") != "exp699_component_output_audit_v1"
        or component_audit.get("public_used") is not False
        or component_audit.get("rows") != len(component_rows)
    ):
        raise ValueError("component-output audit mismatch")

    row_count = len(registry)
    ids = np.asarray([str(row["id"]) for row in registry], dtype=str)
    folds = np.asarray([int(row["fold"]) for row in registry], dtype=np.int8)
    categories = np.asarray([str(row["category"]) for row in registry], dtype=str)
    components = np.asarray([str(row["component_id"]) for row in registry], dtype=str)
    if {int(value) for value in folds} != set(FOLDS):
        raise ValueError("fold coverage mismatch")
    if {str(value) for value in categories} != {BAD, FLAMMABLE}:
        raise ValueError("category coverage mismatch")
    texts = [topology_module.row_text(row, mask_digits=False) for row in runtime_rows]
    normalized_texts = [topology_module.row_text(row, mask_digits=True) for row in runtime_rows]
    cues = np.vstack([cue_vector(text) for text in texts])

    embedding_source = np.load(args.multimodal_embeddings, allow_pickle=False)
    embedding_ids = embedding_source["ids"].astype(str)
    if len(set(embedding_ids)) != len(embedding_ids):
        raise ValueError("Qwen embedding ids are duplicated")
    embedding_position = {value: index for index, value in enumerate(embedding_ids)}
    if any(value not in embedding_position for value in ids):
        raise ValueError("Qwen embeddings do not cover topology ids")
    embeddings = embedding_source["embeddings"][
        [embedding_position[value] for value in ids]
    ].astype(np.float32)
    if embeddings.shape != (row_count, 2048):
        raise ValueError("Qwen embedding shape mismatch")
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    if not np.isfinite(embeddings).all():
        raise ValueError("Qwen embeddings are non-finite")

    fingerprints: list[dict[str, Any]] = []
    for row_id in ids:
        path = topology_module.image_cache_path(args.image_cache, str(row_id))
        topology_module.validate_image_cache_entry(path, args.image_resolved_root)
        fingerprints.append(topology_module.image_fingerprints(path))

    labels = np.asarray([int(row["label"]) for row in component_rows], dtype=np.int8)
    if {int(value) for value in labels} != {0, 1}:
        raise ValueError("label domain mismatch")
    baseline_scores = np.asarray(
        [
            float(row["ensemble"]["production_fixed_baseline_before_prior"]["score"])
            for row in component_rows
        ],
        dtype=np.float64,
    )
    baseline_before = np.asarray(
        [
            int(row["ensemble"]["production_fixed_baseline_before_prior"]["prediction"])
            for row in component_rows
        ],
        dtype=np.int8,
    )
    baseline_after = np.asarray(
        [int(row["ensemble"]["production_fixed_baseline_after_prior"]) for row in component_rows],
        dtype=np.int8,
    )
    if not np.isfinite(baseline_scores).all():
        raise ValueError("baseline score is non-finite")
    flammable = categories == FLAMMABLE
    if not np.array_equal(
        baseline_before[flammable],
        (baseline_scores[flammable] >= FLAMMABLE_THRESHOLD).astype(np.int8),
    ):
        raise ValueError("exact packaged flammable threshold replay mismatch")
    replay_prior, baseline_prior_audit = oracle.apply_frozen_annotator_prior(
        component_rows, baseline_before
    )
    if not np.array_equal(replay_prior, baseline_after):
        raise ValueError("exact packaged prior replay mismatch")
    harmful_indices, v1_self = load_v1_harmful_indices(args.v1_report, args.expected_v1_harmful)

    candidate_before = baseline_before.copy()
    candidate_after = baseline_after.copy()
    query_before = baseline_before.copy()
    query_after = baseline_after.copy()
    permutation_before = baseline_before.copy()
    permutation_after = baseline_after.copy()
    fold_payload: dict[str, Any] = {}
    exact_changed: list[dict[str, Any]] = []
    all_pair_audit: dict[int, dict[str, Any]] = {}

    for outer_fold in SCREEN_FOLDS:
        fused, evidence = build_candidates(
            outer_fold=outer_fold,
            texts=texts,
            normalized_texts=normalized_texts,
            folds=folds,
            categories=categories,
            components=components,
            embeddings=embeddings,
            fingerprints=fingerprints,
            topology_module=topology_module,
        )
        outer_train = np.flatnonzero((folds != outer_fold) & flammable)
        eval_queries = np.flatnonzero((folds == outer_fold) & flammable)
        component_stats = component_statistics(components, labels, outer_train)
        query_model = fit_query_model(
            outer_train,
            labels=labels,
            baseline_scores=baseline_scores,
            baseline_predictions=baseline_before,
            cues=cues,
            texts=texts,
            fingerprints=fingerprints,
            seed=PERMUTATION_SEED + outer_fold,
        )
        query_probs = query_probabilities(
            query_model,
            eval_queries,
            baseline_scores=baseline_scores,
            baseline_predictions=baseline_before,
            cues=cues,
            texts=texts,
            fingerprints=fingerprints,
        )
        query_fold_before = apply_query_control(baseline_before, eval_queries, query_probs)
        query_fold_after, _ = oracle.apply_frozen_annotator_prior(component_rows, query_fold_before)
        query_before[eval_queries] = query_fold_before[eval_queries]
        query_after[eval_queries] = query_fold_after[eval_queries]

        pair_model, pair_train_audit = fit_pair_model(
            outer_train,
            fused=fused,
            evidence=evidence,
            query_labels=labels,
            donor_labels=labels,
            component_stats=component_stats,
            components=components,
            baseline_scores=baseline_scores,
            baseline_predictions=baseline_before,
            cues=cues,
            texts=texts,
            fingerprints=fingerprints,
            seed=PERMUTATION_SEED + 100 + outer_fold,
        )
        _, pair_fold_before, pair_audit = apply_pair_cache(
            queries=eval_queries,
            pair_model=pair_model,
            fused=fused,
            evidence=evidence,
            donor_labels=labels,
            component_stats=component_stats,
            components=components,
            baseline_scores=baseline_scores,
            baseline_predictions=baseline_before,
            query_probabilities_by_index=query_probs,
            cues=cues,
            texts=texts,
            fingerprints=fingerprints,
        )
        pair_fold_after, _ = oracle.apply_frozen_annotator_prior(component_rows, pair_fold_before)
        candidate_before[eval_queries] = pair_fold_before[eval_queries]
        candidate_after[eval_queries] = pair_fold_after[eval_queries]
        all_pair_audit.update({int(row["global_index"]): row for row in pair_audit})

        permuted_labels = deterministic_donor_permutation(
            labels,
            folds,
            categories,
            outer_train,
            PERMUTATION_SEED + 1000 + outer_fold,
        )
        permuted_stats = component_statistics(components, permuted_labels, outer_train)
        permutation_model, permutation_train_audit = fit_pair_model(
            outer_train,
            fused=fused,
            evidence=evidence,
            query_labels=labels,
            donor_labels=permuted_labels,
            component_stats=permuted_stats,
            components=components,
            baseline_scores=baseline_scores,
            baseline_predictions=baseline_before,
            cues=cues,
            texts=texts,
            fingerprints=fingerprints,
            seed=PERMUTATION_SEED + 2000 + outer_fold,
        )
        _, permutation_fold_before, permutation_audit = apply_pair_cache(
            queries=eval_queries,
            pair_model=permutation_model,
            fused=fused,
            evidence=evidence,
            donor_labels=permuted_labels,
            component_stats=permuted_stats,
            components=components,
            baseline_scores=baseline_scores,
            baseline_predictions=baseline_before,
            query_probabilities_by_index=query_probs,
            cues=cues,
            texts=texts,
            fingerprints=fingerprints,
        )
        permutation_fold_after, _ = oracle.apply_frozen_annotator_prior(
            component_rows, permutation_fold_before
        )
        permutation_before[eval_queries] = permutation_fold_before[eval_queries]
        permutation_after[eval_queries] = permutation_fold_after[eval_queries]

        fold_mask = folds == outer_fold
        fold_payload[str(outer_fold)] = {
            "baseline": metric_summary(labels, categories, baseline_after, fold_mask),
            "candidate": metric_summary(labels, categories, pair_fold_after, fold_mask),
            "query_only": metric_summary(labels, categories, query_fold_after, fold_mask),
            "permutation": metric_summary(labels, categories, permutation_fold_after, fold_mask),
            "candidate_comparison": comparison(labels, baseline_after, pair_fold_after, fold_mask),
            "query_only_comparison": comparison(
                labels, baseline_after, query_fold_after, fold_mask
            ),
            "permutation_comparison": comparison(
                labels, baseline_after, permutation_fold_after, fold_mask
            ),
            "candidate_survival": survival(
                baseline_before,
                pair_fold_before,
                baseline_after,
                pair_fold_after,
                fold_mask,
            ),
            "pair_training": pair_train_audit,
            "permutation_training": permutation_train_audit,
            "candidate_gate_rows": len(pair_audit),
            "permutation_gate_rows": len(permutation_audit),
        }

    screen_mask = np.isin(folds, SCREEN_FOLDS)
    if not np.array_equal(candidate_before[~flammable], baseline_before[~flammable]):
        raise ValueError("BAD route changed before prior")
    if not np.array_equal(candidate_after[~flammable], baseline_after[~flammable]):
        raise ValueError("BAD route changed after prior")
    if not np.array_equal(query_after[~flammable], baseline_after[~flammable]):
        raise ValueError("query-only BAD route changed")
    if not np.array_equal(permutation_after[~flammable], baseline_after[~flammable]):
        raise ValueError("permutation BAD route changed")

    family_counts = Counter(
        (str(category), str(component)) for category, component in zip(categories, components)
    )
    family_sizes = np.asarray(
        [
            family_counts[(str(category), str(component))]
            for category, component in zip(categories, components)
        ],
        dtype=np.int32,
    )
    cohort_masks = {
        "all": screen_mask,
        "singleton": screen_mask & (family_sizes == 1),
        "rare_le2": screen_mask & (family_sizes <= 2),
        "repeated": screen_mask & (family_sizes > 2),
        "flammable": screen_mask & flammable,
        "flammable_fp": screen_mask & flammable & (labels == 0) & (baseline_after == 1),
        "flammable_fn": screen_mask & flammable & (labels == 1) & (baseline_after == 0),
    }
    cohorts = {
        name: {
            "candidate": comparison(labels, baseline_after, candidate_after, mask),
            "query_only": comparison(labels, baseline_after, query_after, mask),
            "permutation": comparison(labels, baseline_after, permutation_after, mask),
            "survival": survival(
                baseline_before,
                candidate_before,
                baseline_after,
                candidate_after,
                mask,
            ),
        }
        for name, mask in cohort_masks.items()
    }
    pooled = {
        "baseline": metric_summary(labels, categories, baseline_after, screen_mask),
        "candidate": metric_summary(labels, categories, candidate_after, screen_mask),
        "query_only": metric_summary(labels, categories, query_after, screen_mask),
        "permutation": metric_summary(labels, categories, permutation_after, screen_mask),
        "candidate_comparison": comparison(labels, baseline_after, candidate_after, screen_mask),
        "query_only_comparison": comparison(labels, baseline_after, query_after, screen_mask),
        "permutation_comparison": comparison(
            labels, baseline_after, permutation_after, screen_mask
        ),
    }
    for index in np.flatnonzero(screen_mask & (candidate_after != baseline_after)):
        exact_changed.append(
            {
                "global_index": int(index),
                "id": str(ids[index]),
                "fold": int(folds[index]),
                "category": str(categories[index]),
                "label": int(labels[index]),
                "baseline_before": int(baseline_before[index]),
                "candidate_before": int(candidate_before[index]),
                "baseline_after": int(baseline_after[index]),
                "candidate_after": int(candidate_after[index]),
                "corrected": bool(
                    baseline_after[index] != labels[index]
                    and candidate_after[index] == labels[index]
                ),
                "regressed": bool(
                    baseline_after[index] == labels[index]
                    and candidate_after[index] != labels[index]
                ),
                "pair_audit": all_pair_audit.get(int(index)),
            }
        )

    harmful_screen = sorted(
        index for index in harmful_indices if 0 <= index < row_count and screen_mask[index]
    )
    harmful_payload = {
        "frozen_total": len(harmful_indices),
        "screen_rows": len(harmful_screen),
        "avoided": int(sum(candidate_after[index] == labels[index] for index in harmful_screen)),
        "repeated": int(sum(candidate_after[index] != labels[index] for index in harmful_screen)),
        "v1_decision_repeated": int(
            sum(
                candidate_after[index] == int(harmful_indices[index]["candidate_after"])
                for index in harmful_screen
            )
        ),
        "baseline_decision_restored": int(
            sum(
                candidate_after[index] == int(harmful_indices[index]["baseline_after"])
                for index in harmful_screen
            )
        ),
        "indices": harmful_screen,
    }

    candidate_macro = pooled["candidate"]["macro_f1"]
    baseline_macro = pooled["baseline"]["macro_f1"]
    query_macro = pooled["query_only"]["macro_f1"]
    permutation_macro = pooled["permutation"]["macro_f1"]
    baseline_fn = pooled["baseline"]["categories"][FLAMMABLE]["fn"]
    candidate_fn = pooled["candidate"]["categories"][FLAMMABLE]["fn"]
    gates = {
        "fold0_net_nonnegative": fold_payload["0"]["candidate_comparison"]["net"] >= 0,
        "fold3_net_nonnegative": fold_payload["3"]["candidate_comparison"]["net"] >= 0,
        "aggregate_flammable_fn_nonincrease": candidate_fn <= baseline_fn,
        "candidate_beats_query_only_macro": candidate_macro > query_macro,
        "permutation_has_no_positive_net": pooled["permutation_comparison"]["net"] <= 0,
        "permutation_has_no_positive_macro_delta": permutation_macro <= baseline_macro,
        "useful_changes_survive": pooled["candidate_comparison"]["corrections"] > 0,
    }
    decision = (
        "ACCEPT_CONNECTED_SAFE_PAIR_SCREEN"
        if all(gates.values())
        else "REJECT_CONNECTED_SAFE_PAIR_SCREEN"
    )
    payload = {
        "schema": "exp699_connected_safe_retrieval_pair_screen_v1",
        "experiment": 699,
        "stage": "cpu_oof_folds0_3_no_public",
        "decision": decision,
        "public_used": False,
        "sealed_rows": 0,
        "ods_submit_authorized": False,
        "package_authorized": False,
        "screen_folds": [0, 3],
        "recipe": {
            "same_fold_donors_excluded": True,
            "same_connected_component_donors_excluded": True,
            "top_k": TOP_K,
            "pair_probability_minimum": PAIR_PROBABILITY_MINIMUM,
            "cache_top_k": CACHE_TOP_K,
            "minimum_donors": MINIMUM_DONORS,
            "donor_consensus_minimum": DONOR_CONSENSUS_MINIMUM,
            "cache_alpha": CACHE_ALPHA,
            "threshold": FLAMMABLE_THRESHOLD,
            "query_negative_strong": QUERY_NEGATIVE_STRONG,
            "query_positive_strong": QUERY_POSITIVE_STRONG,
            "hard_override": "exact_text_and_exact_image_at_least2_unanimous_component_agreement1",
            "prior": "exact_frozen_solution140_after_pair_cache",
            "bad_route": "byte_identical_solution140",
        },
        "gates": gates,
        "folds": fold_payload,
        "pooled": pooled,
        "cohorts": cohorts,
        "v1_harmful_28": harmful_payload,
        "exact_changed_decisions": exact_changed,
        "baseline_prior_audit": baseline_prior_audit,
        "source_bindings": {
            "source_manifest_file_sha256": sha256_file(args.source_manifest),
            "source_manifest_self_sha256": source_manifest_self,
            "preregister_file_sha256": sha256_file(args.preregister),
            "preregister_self_sha256": preregister_self,
            "baseline_semantics": source_manifest["baseline_semantics"],
            "solution140_run_sha256": preregister["source_bindings"]["solution140_run_sha256"],
            "topology_self_sha256": topology_self,
            "component_audit_self_sha256": component_audit_self,
            "v1_report_self_sha256": v1_self,
            "file_sha256": {
                "topology": sha256_file(args.topology),
                "topology_module": sha256_file(args.topology_module),
                "component_outputs": sha256_file(args.component_outputs),
                "component_audit": sha256_file(args.component_audit),
                "multimodal_embeddings": sha256_file(args.multimodal_embeddings),
                "v1_report": sha256_file(args.v1_report),
                "oracle_module": sha256_file(oracle_path),
                "pair_verifier": sha256_file(Path(__file__).resolve()),
            },
            "fold_runtime_sha256": {
                str(fold): sha256_file(path) for fold, path in runtime_paths.items()
            },
        },
    }
    file_sha, self_sha = write_self_hashed(args.output, payload)
    print(
        json.dumps(
            {
                "decision": decision,
                "file_sha256": file_sha,
                "self_sha256": self_sha,
                "macro_delta": candidate_macro - baseline_macro,
                "query_only_macro_delta": query_macro - baseline_macro,
                "permutation_macro_delta": permutation_macro - baseline_macro,
                "corrections": pooled["candidate_comparison"]["corrections"],
                "regressions": pooled["candidate_comparison"]["regressions"],
                "flammable_fn": [baseline_fn, candidate_fn],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

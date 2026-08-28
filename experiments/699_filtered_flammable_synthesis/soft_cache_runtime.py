from __future__ import annotations

import hashlib
import html
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from scipy import sparse


SCHEMA = "exp699_solution140_flammable_soft_cache_v1"
FLAMMABLE = "Легковоспламеняющиеся"
PREREGISTER_SELF_SHA256 = (
    "dbc0d666d4e9e1ce93f46034c393fcc67fe082732c6e8bd5ccf786cfb5914ff9"
)
SOURCE_MANIFEST_SELF_SHA256 = (
    "b06f3e3d98637239b227ae946ca690c2c65dc72addfd71fe8fb8801bd24767a0"
)
CHANNEL_WEIGHTS = {
    "exact_text": 2.0,
    "normalized_text": 1.5,
    "tfidf": 1.0,
    "bm25": 1.0,
    "image_exact": 2.0,
    "image_near": 1.0,
}
CANDIDATE_TOP_K = 30
CACHE_TOP_K = 5
RRF_OFFSET = 60
MINIMUM_DONORS = 3
MINIMUM_AGREEMENT = 0.8
MINIMUM_CHANNELS = 2
CACHE_ALPHA = 0.25
FLAMMABLE_THRESHOLD = 0.953912615776062
MINIMUM_GRAYSCALE_STD = 8.0
PHASH_HAMMING_MAX = 4
DHASH_HAMMING_MAX = 6
EXPECTED_DONORS = 4716


def canonical_text(value: object, *, mask_digits: bool) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    if mask_digits:
        text = re.sub(r"\d+(?:[.,]\d+)?", " # ", text)
    text = re.sub(r"[^0-9a-zа-я#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def row_text(name: object, description: object, *, mask_digits: bool) -> str:
    return canonical_text(
        f"{name or ''}\n{name or ''}\n{description or ''}",
        mask_digits=mask_digits,
    )


def text_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dct_matrix(size: int) -> np.ndarray:
    x = np.arange(size, dtype=np.float64)
    k = x[:, None]
    matrix = np.cos((math.pi / size) * (x + 0.5) * k)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


_DCT32 = _dct_matrix(32)


def _bits_to_int(bits: np.ndarray) -> int:
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
    return {
        "file_sha256": file_sha,
        "phash64": _bits_to_int(low >= median),
        "dhash64": _bits_to_int(gray9x8[:, 1:] >= gray9x8[:, :-1]),
        "grayscale_std": float(gray32.std()),
    }


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
        return sorted(results)


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
        denominator = values + k1 * (
            1.0 - b + b * lengths[row] / average_length
        )
        weighted.data[start:stop] = values * (k1 + 1.0) / denominator
    return weighted.multiply(inverse_document_frequency).tocsr()


def sparse_top_k(
    query: sparse.csr_matrix,
    donors: sparse.csr_matrix,
    donor_order: np.ndarray,
    k: int,
) -> list[list[int]]:
    result: list[list[int]] = []
    donor_transpose = donors.T.tocsc()
    limit = min(k, donors.shape[0])
    for start in range(0, query.shape[0], 256):
        scores = (query[start : start + 256] @ donor_transpose).toarray()
        for row in scores:
            positive = np.flatnonzero(row > 0.0)
            local_limit = min(limit, len(positive))
            if local_limit == 0:
                result.append([])
                continue
            selected = positive[
                np.argpartition(-row[positive], local_limit - 1)[:local_limit]
            ]
            ordered = sorted(
                selected,
                key=lambda index: (-float(row[index]), int(donor_order[index])),
            )
            result.append([int(index) for index in ordered])
    return result


def fuse_with_evidence(
    channel_lists: dict[str, list[list[int]]],
    channel_weights: dict[str, float],
    row_count: int,
    top_k: int,
    rrf_offset: int,
    donor_order: np.ndarray | None = None,
) -> tuple[list[list[int]], list[dict[int, dict[str, Any]]]]:
    fused: list[list[int]] = []
    evidence: list[dict[int, dict[str, Any]]] = []
    for query in range(row_count):
        scores: dict[int, float] = defaultdict(float)
        channels: dict[int, set[str]] = defaultdict(set)
        best_rank: dict[int, int] = {}
        for channel, rows in channel_lists.items():
            weight = float(channel_weights[channel])
            for rank, donor in enumerate(rows[query], 1):
                donor = int(donor)
                scores[donor] += weight / (rrf_offset + rank)
                channels[donor].add(channel)
                best_rank[donor] = min(best_rank.get(donor, rank), rank)
        ordered = sorted(
            scores,
            key=lambda donor: (
                -scores[donor],
                -len(channels[donor]),
                best_rank[donor],
                donor if donor_order is None else int(donor_order[donor]),
            ),
        )[:top_k]
        fused.append(ordered)
        evidence.append(
            {
                donor: {
                    "rrf_score": float(scores[donor]),
                    "channels": sorted(channels[donor]),
                }
                for donor in ordered
            }
        )
    return fused, evidence


def apply_fused_cache(
    donor_labels: np.ndarray,
    categories: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    fused: list[list[int]],
    evidence: list[dict[int, dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    candidate_scores = np.asarray(baseline_scores, dtype=np.float64).copy()
    candidate_predictions = np.asarray(baseline_predictions, dtype=np.int8).copy()
    audit: list[dict[str, Any]] = []
    for query in np.flatnonzero(np.asarray(categories) == FLAMMABLE):
        donors = fused[int(query)][:CACHE_TOP_K]
        if len(donors) < MINIMUM_DONORS:
            continue
        weights = np.asarray(
            [float(evidence[int(query)][donor]["rrf_score"]) for donor in donors],
            dtype=np.float64,
        )
        local_labels = np.asarray(donor_labels, dtype=np.float64)[donors]
        positive_rate = float(np.average(local_labels, weights=weights))
        agreement = max(positive_rate, 1.0 - positive_rate)
        evidence_channels = sorted(
            {
                channel
                for donor in donors
                for channel in evidence[int(query)][donor]["channels"]
            }
        )
        gate = agreement >= MINIMUM_AGREEMENT and (
            len(evidence_channels) >= MINIMUM_CHANNELS
            or "exact_text" in evidence_channels
        )
        if not gate:
            continue
        old_score = float(candidate_scores[query])
        new_score = (1.0 - CACHE_ALPHA) * old_score + CACHE_ALPHA * positive_rate
        candidate_scores[query] = new_score
        candidate_predictions[query] = int(new_score >= FLAMMABLE_THRESHOLD)
        audit.append(
            {
                "row_index": int(query),
                "donors": [int(value) for value in donors],
                "donor_count": len(donors),
                "donor_positive_rate": positive_rate,
                "donor_agreement": agreement,
                "evidence_channels": evidence_channels,
                "baseline_score": old_score,
                "candidate_score": new_score,
                "baseline_prediction": int(baseline_predictions[query]),
                "candidate_prediction": int(candidate_predictions[query]),
            }
        )
    return candidate_scores, candidate_predictions, audit


def dump_cache(cache: dict[str, Any], path: Path) -> None:
    joblib.dump(cache, path, compress=("zlib", 3), protocol=4)


def load_cache(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("soft-cache artifact must be a regular file")
    cache = joblib.load(path)
    expected = {
        "schema": SCHEMA,
        "preregister_self_sha256": PREREGISTER_SELF_SHA256,
        "source_manifest_self_sha256": SOURCE_MANIFEST_SELF_SHA256,
        "donor_count": EXPECTED_DONORS,
        "category": FLAMMABLE,
        "channel_weights": CHANNEL_WEIGHTS,
        "candidate_top_k": CANDIDATE_TOP_K,
        "cache_top_k": CACHE_TOP_K,
        "rrf_offset": RRF_OFFSET,
        "minimum_unique_donors": MINIMUM_DONORS,
        "minimum_donor_label_agreement": MINIMUM_AGREEMENT,
        "minimum_evidence_channels": MINIMUM_CHANNELS,
        "cache_alpha": CACHE_ALPHA,
        "threshold": FLAMMABLE_THRESHOLD,
    }
    mismatch = {
        key: {"expected": value, "actual": cache.get(key)}
        for key, value in expected.items()
        if cache.get(key) != value
    }
    labels = np.asarray(cache.get("donor_labels"))
    order = np.asarray(cache.get("donor_order"))
    if (
        mismatch
        or labels.shape != (EXPECTED_DONORS,)
        or order.shape != (EXPECTED_DONORS,)
        or not np.isin(labels, [0, 1]).all()
        or len(np.unique(order)) != EXPECTED_DONORS
        or not np.all(order[:-1] < order[1:])
    ):
        raise ValueError(f"soft-cache contract mismatch: {mismatch}")
    return cache


def _exact_channel(keys: list[str], groups: dict[str, list[int]]) -> list[list[int]]:
    return [list(groups.get(text_key(value), ()))[:CANDIDATE_TOP_K] for value in keys]


def _image_channels(
    paths: list[Path | None], cache: dict[str, Any]
) -> tuple[list[list[int]], list[list[int]]]:
    exact_groups = cache["image_exact_groups"]
    phash_groups = cache["image_phash_groups"]
    tree = BKTree()
    for value in sorted(int(item) for item in phash_groups):
        tree.add(value)
    exact_result: list[list[int]] = []
    near_result: list[list[int]] = []
    donor_dhash = np.asarray(cache["image_dhash"], dtype=np.uint64)
    for path in paths:
        if path is None:
            exact_result.append([])
            near_result.append([])
            continue
        fingerprint = image_fingerprints(path)
        exact_result.append(
            list(exact_groups.get(fingerprint["file_sha256"], ()))[:CANDIDATE_TOP_K]
        )
        candidates: list[tuple[int, int, int]] = []
        if float(fingerprint["grayscale_std"]) >= MINIMUM_GRAYSCALE_STD:
            for value in tree.query(int(fingerprint["phash64"]), PHASH_HAMMING_MAX):
                for donor in phash_groups[str(value)]:
                    phash_distance = (int(fingerprint["phash64"]) ^ value).bit_count()
                    dhash_distance = (
                        int(fingerprint["dhash64"]) ^ int(donor_dhash[donor])
                    ).bit_count()
                    if dhash_distance <= DHASH_HAMMING_MAX:
                        candidates.append((phash_distance, dhash_distance, int(donor)))
        near_result.append(
            [donor for _, _, donor in sorted(set(candidates))[:CANDIDATE_TOP_K]]
        )
    return exact_result, near_result


def apply_runtime_cache(
    frame: Any,
    baseline_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cache = load_cache(cache_path)
    categories = frame["category"].astype(str).to_numpy()
    flammable_positions = np.flatnonzero(categories == FLAMMABLE)
    if len(flammable_positions) == 0:
        return (
            np.asarray(baseline_scores, dtype=np.float64).copy(),
            np.asarray(baseline_predictions, dtype=np.int8).copy(),
            [],
        )
    texts = [
        row_text(frame.iloc[index]["name"], frame.iloc[index]["description"], mask_digits=False)
        for index in flammable_positions
    ]
    normalized = [
        row_text(frame.iloc[index]["name"], frame.iloc[index]["description"], mask_digits=True)
        for index in flammable_positions
    ]
    tfidf_query = cache["tfidf_vectorizer"].transform(texts).tocsr()
    bm25_query = cache["bm25_vectorizer"].transform(texts).sign().astype(np.float32).tocsr()
    donor_order = np.asarray(cache["donor_order"], dtype=np.int64)
    paths: list[Path | None] = []
    for index in flammable_positions:
        values = frame.iloc[index]["image_paths"]
        paths.append(None if not values else Path(values[0]))
    image_exact, image_near = _image_channels(paths, cache)
    channels = {
        "exact_text": _exact_channel(texts, cache["exact_text_groups"]),
        "normalized_text": _exact_channel(normalized, cache["normalized_text_groups"]),
        "tfidf": sparse_top_k(
            tfidf_query, cache["tfidf_matrix"], donor_order, CANDIDATE_TOP_K
        ),
        "bm25": sparse_top_k(
            bm25_query, cache["bm25_matrix"], donor_order, CANDIDATE_TOP_K
        ),
        "image_exact": image_exact,
        "image_near": image_near,
    }
    local_fused, local_evidence = fuse_with_evidence(
        channels,
        CHANNEL_WEIGHTS,
        len(flammable_positions),
        CANDIDATE_TOP_K,
        RRF_OFFSET,
        donor_order,
    )
    fused = [[] for _ in range(len(frame))]
    evidence = [{} for _ in range(len(frame))]
    for local, global_index in enumerate(flammable_positions):
        fused[int(global_index)] = local_fused[local]
        evidence[int(global_index)] = local_evidence[local]
    return apply_fused_cache(
        np.asarray(cache["donor_labels"], dtype=np.int8),
        categories,
        baseline_scores,
        baseline_predictions,
        fused,
        evidence,
    )

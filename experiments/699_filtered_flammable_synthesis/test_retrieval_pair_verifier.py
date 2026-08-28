from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "exp699_retrieval_pair_verifier",
    ROOT / "evaluate_retrieval_pair_verifier.py",
)
assert SPEC is not None and SPEC.loader is not None
pair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pair)


class FixedModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        positive = np.full(len(matrix), self.probability, dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])


def minimal_evidence(*, exact: bool = False) -> dict:
    channels = {
        "qwen_embedding": {"rank": 1, "score": 0.9},
        "tfidf": {"rank": 1, "score": 0.8},
    }
    if exact:
        channels["exact_text"] = {"rank": 1, "score": 1.0}
        channels["image_exact"] = {"rank": 1, "score": 1.0}
    return {
        "channels": channels,
        "rrf_score": 0.1,
        "channel_count": len(channels),
        "phash_distance": 0 if exact else None,
        "dhash_distance": 0 if exact else None,
    }


def shared_arrays() -> dict:
    return {
        "components": np.asarray(["q", "a", "b", "c"]),
        "cues": np.zeros((4, 6), dtype=np.float64),
        "texts": ["query", "donor a", "donor b", "donor c"],
        "fingerprints": [{"grayscale_std": 10.0}] * 4,
    }


def test_eligible_donors_excludes_same_fold_and_component() -> None:
    donor_pool = np.asarray([1, 2, 3, 4], dtype=np.int64)
    folds = np.asarray([0, 0, 1, 2, 3], dtype=np.int8)
    components = np.asarray(["q", "x", "q", "y", "z"])
    assert pair.eligible_donors(donor_pool, 0, folds, components).tolist() == [3, 4]


def test_cue_mismatch_is_explicit_and_directional() -> None:
    query = pair.cue_vector("Баллон с газом, топливо включено в комплект")
    donor = pair.cue_vector("Пустой корпус без газа, только упаковка")
    mismatch = pair.cue_mismatches(query, donor)
    assert query[:3].tolist() == [1.0, 1.0, 1.0]
    assert donor[3:].sum() >= 2
    assert mismatch.shape == (18,)
    assert mismatch.sum() >= 6


def test_normal_boundary_cross_requires_strong_query_control() -> None:
    common = shared_arrays()
    donor_labels = np.asarray([0, 1, 1, 1], dtype=np.int8)
    component_stats = {
        name: {"count": 1.0, "agreement": 1.0, "reliability": 1.0} for name in common["components"]
    }
    kwargs = {
        "queries": np.asarray([0], dtype=np.int64),
        "pair_model": FixedModel(0.9),
        "fused": {0: [1, 2, 3]},
        "evidence": {0: {index: minimal_evidence() for index in (1, 2, 3)}},
        "donor_labels": donor_labels,
        "component_stats": component_stats,
        "components": common["components"],
        "baseline_scores": np.asarray([0.95, 0.0, 0.0, 0.0]),
        "baseline_predictions": np.asarray([0, 0, 0, 0], dtype=np.int8),
        "cues": common["cues"],
        "texts": common["texts"],
        "fingerprints": common["fingerprints"],
    }
    _, accepted, audit = pair.apply_pair_cache(**kwargs, query_probabilities_by_index={0: 0.9})
    _, rejected, rejected_audit = pair.apply_pair_cache(
        **kwargs, query_probabilities_by_index={0: 0.7}
    )
    assert accepted[0] == 1
    assert audit[0]["boundary_authorized"] is True
    assert rejected[0] == 0
    assert rejected_audit[0]["boundary_authorized"] is False


def test_exact_duplicate_override_requires_two_text_and_image_matches() -> None:
    common = shared_arrays()
    donor_labels = np.asarray([1, 0, 0, 0], dtype=np.int8)
    component_stats = {
        name: {"count": 2.0, "agreement": 1.0, "reliability": 1.0} for name in common["components"]
    }
    _, predictions, audit = pair.apply_pair_cache(
        queries=np.asarray([0], dtype=np.int64),
        pair_model=FixedModel(0.95),
        fused={0: [1, 2, 3]},
        evidence={
            0: {
                1: minimal_evidence(exact=True),
                2: minimal_evidence(exact=True),
                3: minimal_evidence(),
            }
        },
        donor_labels=donor_labels,
        component_stats=component_stats,
        components=common["components"],
        baseline_scores=np.asarray([0.97, 0.0, 0.0, 0.0]),
        baseline_predictions=np.asarray([1, 0, 0, 0], dtype=np.int8),
        query_probabilities_by_index={0: 0.95},
        cues=common["cues"],
        texts=common["texts"],
        fingerprints=common["fingerprints"],
    )
    assert predictions[0] == 0
    assert audit[0]["hard_override"] is True


def test_donor_permutation_is_deterministic_and_stratified() -> None:
    labels = np.asarray([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int8)
    folds = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int8)
    categories = np.asarray([pair.FLAMMABLE] * 8)
    donors = np.arange(8, dtype=np.int64)
    first = pair.deterministic_donor_permutation(labels, folds, categories, donors, 42)
    second = pair.deterministic_donor_permutation(labels, folds, categories, donors, 42)
    assert np.array_equal(first, second)
    for fold in range(4):
        mask = folds == fold
        assert int(first[mask].sum()) == int(labels[mask].sum())


def test_final_comparison_counts_exact_changes() -> None:
    labels = np.asarray([0, 1, 1, 0], dtype=np.int8)
    baseline = np.asarray([1, 1, 0, 0], dtype=np.int8)
    candidate = np.asarray([0, 0, 1, 0], dtype=np.int8)
    result = pair.comparison(labels, baseline, candidate, np.ones(4, dtype=bool))
    assert result == {"changed": 3, "corrections": 2, "regressions": 1, "net": 1}


def test_legacy_self_hash_is_verified_without_weakening_current_contracts() -> None:
    payload = {"schema": "legacy", "rows": 2}
    payload["self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert pair.verify_legacy_self_hash(payload) == payload["self_sha256"]
    tampered = dict(payload, rows=3)
    try:
        pair.verify_legacy_self_hash(tampered)
    except ValueError as error:
        assert str(error) == "legacy self-hash mismatch"
    else:
        raise AssertionError("tampered legacy contract was accepted")


def test_frozen_constants_match_preregister() -> None:
    assert pair.SCREEN_FOLDS == (0, 3)
    assert pair.TOP_K == 30
    assert pair.PAIR_PROBABILITY_MINIMUM == 0.8
    assert pair.CACHE_TOP_K == 5
    assert pair.MINIMUM_DONORS == 3
    assert pair.DONOR_CONSENSUS_MINIMUM == 0.9
    assert pair.CACHE_ALPHA == 0.15
    assert pair.PERMUTATION_SEED == 42
    assert pair.FLAMMABLE_THRESHOLD == 0.953912615776062

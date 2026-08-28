from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "exp699_soft_cache", ROOT / "evaluate_soft_cache_exploratory.py"
)
assert spec is not None and spec.loader is not None
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


def test_fuse_with_evidence_preserves_channel_provenance() -> None:
    channels = {
        "exact_text": [[1, 2], [0], [0]],
        "tfidf": [[2, 1], [0], [0]],
    }
    fused, evidence = cache.fuse_with_evidence(
        channels, {"exact_text": 2.0, "tfidf": 1.0}, 3, 2, 60
    )
    assert fused[0] == [1, 2]
    assert evidence[0][1]["channels"] == ["exact_text", "tfidf"]


def test_soft_cache_is_flammable_only_and_uses_frozen_threshold() -> None:
    labels = np.asarray([1, 1, 1, 1], dtype=np.int8)
    categories = np.asarray(["Легковоспламеняющиеся", "БАД", "БАД", "БАД"])
    scores = np.asarray([0.94, 0.2, 0.3, 0.4], dtype=np.float64)
    predictions = np.asarray([0, 0, 0, 0], dtype=np.int8)
    fused = [[1, 2, 3], [], [], []]
    evidence = [
        {
            1: {"rrf_score": 1.0, "channels": ["tfidf"]},
            2: {"rrf_score": 1.0, "channels": ["bm25"]},
            3: {"rrf_score": 1.0, "channels": ["tfidf", "bm25"]},
        },
        {},
        {},
        {},
    ]
    candidate_scores, candidate_predictions, audit = cache.apply_soft_cache(
        labels, categories, scores, predictions, fused, evidence
    )
    assert len(audit) == 1
    assert candidate_scores[0] > scores[0]
    assert candidate_predictions[0] == int(
        candidate_scores[0] >= cache.FLAMMABLE_THRESHOLD
    )
    assert np.array_equal(candidate_predictions[1:], predictions[1:])


def test_soft_cache_rejects_low_agreement() -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    categories = np.asarray(["Легковоспламеняющиеся", "БАД", "БАД", "БАД"])
    scores = np.asarray([0.96, 0.2, 0.3, 0.4])
    predictions = np.asarray([1, 0, 0, 0], dtype=np.int8)
    fused = [[1, 2, 3], [], [], []]
    evidence = [
        {
            1: {"rrf_score": 1.0, "channels": ["tfidf"]},
            2: {"rrf_score": 1.0, "channels": ["bm25"]},
            3: {"rrf_score": 1.0, "channels": ["tfidf", "bm25"]},
        },
        {},
        {},
        {},
    ]
    _, candidate_predictions, audit = cache.apply_soft_cache(
        labels, categories, scores, predictions, fused, evidence
    )
    assert audit == []
    assert np.array_equal(candidate_predictions, predictions)

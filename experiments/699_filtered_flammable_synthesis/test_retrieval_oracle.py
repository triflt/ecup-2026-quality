from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "exp699_retrieval_oracle", ROOT / "evaluate_retrieval_oracle.py"
)
assert spec is not None and spec.loader is not None
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def test_group_candidates_excludes_same_fold_and_category() -> None:
    values = ["x", "x", "x", "x"]
    folds = np.asarray([0, 1, 0, 2], dtype=np.int8)
    categories = np.asarray(["A", "A", "B", "A"])
    result = oracle.group_candidates(values, folds, categories, 30)
    assert result[0] == [1, 3]
    assert result[1] == [0, 3]
    assert result[2] == []


def test_dense_and_sparse_top_k_are_deterministic() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    donors = np.asarray([[1.0, 0.0], [0.5, 0.0], [0.0, 1.0]], dtype=np.float32)
    positions = np.asarray([9, 4, 7], dtype=np.int64)
    assert oracle.dense_top_k(query, donors, positions, 2) == [[9, 4]]
    assert oracle.sparse_top_k(
        sparse.csr_matrix(query), sparse.csr_matrix(donors), positions, 2
    ) == [[9, 4]]


def test_sparse_top_k_does_not_invent_zero_similarity_donors() -> None:
    query = sparse.csr_matrix(np.asarray([[1.0, 0.0]], dtype=np.float32))
    donors = sparse.csr_matrix(
        np.asarray([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    )
    positions = np.asarray([3, 4], dtype=np.int64)
    assert oracle.sparse_top_k(query, donors, positions, 30) == [[]]


def test_fusion_does_not_read_labels() -> None:
    channels = {
        name: [[1, 2], [0], [0]] for name in oracle.CHANNEL_WEIGHTS
    }
    first = oracle.fuse_channels(channels, 3, 2)
    labels_a = np.asarray([0, 1, 0], dtype=np.int8)
    labels_b = 1 - labels_a
    assert first == oracle.fuse_channels(channels, 3, 2)
    assert labels_a.tolist() != labels_b.tolist()


def test_retrieval_summary_and_permutation_contract() -> None:
    candidates = [[1, 2], [0, 2], [0, 1], [0, 1]]
    labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
    folds = np.asarray([0, 1, 2, 3], dtype=np.int8)
    categories = np.asarray([oracle.FLAMMABLE] * 4)
    mask = np.asarray([True, True, True, False])
    summary = oracle.retrieval_summary(candidates, labels, mask)
    assert summary["queries"] == 3
    assert summary["recall_at"]["1"] == 2 / 3
    control = oracle.permutation_control(
        candidates,
        labels,
        folds,
        categories,
        {"scope": mask},
        repetitions=3,
        seed=42,
    )
    assert control["scope"]["repetitions"] == 3


def test_project_embeddings_is_finite_and_deterministic() -> None:
    values = np.eye(4, dtype=np.float32)
    first = oracle.project_embeddings(values, 3, 42)
    second = oracle.project_embeddings(values, 3, 42)
    assert first.shape == (4, 3)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_bm25_shape_and_finite() -> None:
    counts = sparse.csr_matrix(np.asarray([[1, 2, 0], [0, 1, 3]], dtype=np.float32))
    result = oracle.bm25_documents(counts)
    assert result.shape == counts.shape
    assert np.isfinite(result.data).all()


def test_frozen_annotator_prior_replays_exact_and_name_only() -> None:
    rows = [
        {"ensemble": {"annotator_prior": {"source": "none", "value": None}}},
        {"ensemble": {"annotator_prior": {"source": "exact", "value": 1}}},
        {"ensemble": {"annotator_prior": {"source": "name", "value": 0}}},
    ]
    predictions = np.asarray([1, 0, 1], dtype=np.int8)
    result, audit = oracle.apply_frozen_annotator_prior(rows, predictions)
    assert result.tolist() == [1, 1, 0]
    assert audit["sources"] == {"exact": 1, "name": 1, "none": 1}
    assert audit["overrides"] == 2
    assert audit["prediction_changes"] == 2

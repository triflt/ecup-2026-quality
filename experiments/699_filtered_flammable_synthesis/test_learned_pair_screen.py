from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load("exp699_learned_preflight", ROOT / "prepare_learned_pair_preflight.py")
trainer = load("exp699_learned_trainer", ROOT / "train_learned_pair_screen.py")


def test_hard_pair_selection_is_exact_balanced_and_rank_preserving() -> None:
    labels = np.asarray([1, 1, 0, 1, 0, 0, 1, 0, 1, 0], dtype=np.int8)
    selected = preflight.select_balanced_hard_pairs(0, list(range(1, 10)), labels, maximum=4)
    assert selected == [
        (1, 1),
        (3, 1),
        (6, 1),
        (8, 1),
        (2, 0),
        (4, 0),
        (5, 0),
        (7, 0),
    ]
    assert sum(target for _, target in selected) == len(selected) // 2


def test_hard_pair_selection_abstains_without_both_targets() -> None:
    labels = np.asarray([1, 1, 1, 1], dtype=np.int8)
    assert preflight.select_balanced_hard_pairs(0, [1, 2, 3], labels) == []


def test_projection_is_biasless_2048_to_256_and_l2_normalized() -> None:
    projection = trainer.Projection()
    assert projection.linear.bias is None
    assert projection.linear.in_features == 2048
    assert projection.linear.out_features == 256
    output = projection(torch.randn(3, 2048))
    assert torch.allclose(torch.linalg.vector_norm(output, dim=1), torch.ones(3), atol=1e-5)


def test_controls_have_equal_architecture_and_fixed_projection_is_frozen() -> None:
    models = trainer.cloned_models(60, torch.device("cpu"))
    parameter_counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    assert len(set(parameter_counts.values())) == 1
    assert all(
        not parameter.requires_grad
        for parameter in models["fixed_embedding"].projection.parameters()
    )
    reference = models["learned_pair"].projection.linear.weight
    assert torch.equal(reference, models["fixed_embedding"].projection.linear.weight)


def test_query_only_uses_same_width_with_zero_donor_inputs() -> None:
    model = trainer.LearnedPairModel(60)
    query = torch.randn(2, 2048)
    scalar = torch.randn(2, 60)
    values = model.query_input(query, scalar)
    assert values.shape == (2, 4 * 256 + 60)
    assert torch.count_nonzero(values[:, 256 : 4 * 256]) == 0
    assert torch.count_nonzero(values[:, 4 * 256 + trainer.QUERY_SCALAR_WIDTH :]) == 0


def test_permutation_preserves_each_query_balance_and_changes_donor_label() -> None:
    queries = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    targets = np.asarray([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.int8)
    scalar = np.zeros((8, 60), dtype=np.float32)
    query_labels = np.asarray([1, 0], dtype=np.int8)
    permuted, permuted_scalar = trainer.permuted_targets_and_scalar(
        queries, targets, scalar, query_labels
    )
    for query in (0, 1):
        mask = queries == query
        assert int(permuted[mask].sum()) == int(targets[mask].sum())
        expected_donor = permuted[mask] if query_labels[query] == 1 else 1 - permuted[mask]
        assert np.array_equal(
            permuted_scalar[mask, trainer.DONOR_LABEL_SCALAR_INDEX], expected_donor
        )


def test_frozen_residual_formula() -> None:
    assert trainer.residual_from_vote(0.5, 0.8) == 0.0
    assert np.isclose(trainer.residual_from_vote(1.0, 0.8), 0.08)
    assert np.isclose(trainer.residual_from_vote(0.0, 0.8), -0.08)
    assert np.isclose(trainer.residual_from_vote(0.75, 0.6), 0.03)


def test_aggregation_obeys_probability_count_agreement_and_mean_gates() -> None:
    heldout = np.asarray([0, 1], dtype=np.int64)
    donors = np.asarray([2, 3, 4, 2, 3], dtype=np.int64)
    offsets = np.asarray([0, 3, 5], dtype=np.int64)
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.55, 0.51], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1, 1], dtype=np.int8)
    scores = np.asarray([0.94, 0.97, 0.0, 0.0, 0.0], dtype=np.float32)
    candidate_scores, _, head, contributions = trainer.aggregate_pairs(
        heldout_queries=heldout,
        eval_donors=donors,
        offsets=offsets,
        probabilities=probabilities,
        donor_labels=labels,
        baseline_scores=scores,
    )
    assert candidate_scores[0] > scores[0]
    assert candidate_scores[1] == scores[1]
    assert head[0] == 1
    assert 0 in contributions
    assert 1 not in contributions


def test_prior_is_exact_override() -> None:
    predictions = np.asarray([0, 1, 0, 1], dtype=np.int8)
    source = np.asarray([0, 1, 2, 0], dtype=np.int8)
    value = np.asarray([-1, 0, 1, -1], dtype=np.int8)
    assert trainer.apply_prior(predictions, source, value).tolist() == [0, 0, 1, 1]


def test_frozen_training_and_aggregation_constants() -> None:
    assert trainer.SEED == 42
    assert trainer.PROJECTION_DIMENSION == 256
    assert trainer.BATCH_SIZE == 512
    assert trainer.EPOCHS == 10
    assert trainer.LEARNING_RATE == 3e-4
    assert trainer.WEIGHT_DECAY == 1e-4
    assert trainer.GRADIENT_CLIP == 1.0
    assert trainer.COMPATIBILITY_MINIMUM == 0.5
    assert trainer.NEIGHBORS_MAX == 5
    assert trainer.NEIGHBORS_MIN == 2
    assert trainer.AGREEMENT_MINIMUM == 0.75
    assert trainer.MEAN_COMPATIBILITY_MINIMUM == 0.6
    assert trainer.RESIDUAL_SCALE == 0.10
    assert trainer.FLAMMABLE_THRESHOLD == 0.953912615776062

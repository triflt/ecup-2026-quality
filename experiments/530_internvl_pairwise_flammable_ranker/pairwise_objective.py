from __future__ import annotations

import random
from collections.abc import Iterator

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # CPU-only validation environments test the scheduler.
    torch = None
    F = None


def pairwise_logistic_loss(
    positive_scores: torch.Tensor, negative_scores: torch.Tensor
) -> torch.Tensor:
    if F is None:
        raise RuntimeError("pairwise loss requires the training environment with torch")
    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive and negative score tensors must have equal shape")
    if positive_scores.numel() == 0:
        raise ValueError("pairwise loss requires at least one pair")
    return F.softplus(-(positive_scores - negative_scores)).mean()


def cyclic_index_batches(
    *, length: int, batch_size: int, batches: int, seed: int
) -> Iterator[list[int]]:
    if length < batch_size or batch_size < 1 or batches < 1:
        raise ValueError("invalid cyclic batch dimensions")
    cycle = 0
    emitted = 0
    while emitted < batches:
        order = list(range(length))
        random.Random(seed + cycle).shuffle(order)
        for start in range(0, length - batch_size + 1, batch_size):
            yield order[start : start + batch_size]
            emitted += 1
            if emitted == batches:
                return
        cycle += 1

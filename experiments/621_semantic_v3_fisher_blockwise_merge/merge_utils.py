"""Pure NumPy utilities for a deterministic, label-free adapter merge.

This module deliberately does not know about datasets, validation labels, or
runtime infrastructure.  Fisher values are accepted as inputs and must be
computed by a caller on a train-only partition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

ArrayMap = Mapping[str, np.ndarray]


@dataclass(frozen=True)
class AdapterMetadata:
    """Identity fields that must match before two LoRA adapters are merged."""

    base_model_id: str
    base_model_revision: str
    target_modules: tuple[str, ...]
    rank: int
    lora_alpha: float
    use_rslora: bool = False
    config_identity_sha256: str = ""


def _finite_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_compatible_metadata(original: AdapterMetadata, specialist: AdapterMetadata) -> None:
    """Fail closed when adapters were not produced for the same base model."""

    fields = (
        "base_model_id",
        "base_model_revision",
        "target_modules",
        "rank",
        "lora_alpha",
        "use_rslora",
        "config_identity_sha256",
    )
    for field in fields:
        if getattr(original, field) != getattr(specialist, field):
            raise ValueError(f"adapter metadata mismatch: {field}")
    if original.rank <= 0 or original.lora_alpha <= 0:
        raise ValueError("rank and lora_alpha must be positive")


def effective_lora_delta(
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    *,
    lora_alpha: float,
    rank: int,
    use_rslora: bool = False,
) -> np.ndarray:
    """Return the effective update ``scale * B @ A`` used by LoRA."""

    if rank <= 0 or lora_alpha <= 0:
        raise ValueError("rank and lora_alpha must be positive")
    a = _finite_matrix(lora_a, name="lora_a")
    b = _finite_matrix(lora_b, name="lora_b")
    if a.shape[0] != rank or b.shape[1] != rank:
        raise ValueError("LoRA matrices do not match the declared rank")
    scale = float(lora_alpha) / (np.sqrt(rank) if use_rslora else rank)
    return scale * (b @ a)


def compute_blockwise_alpha(
    fisher_original: Mapping[str, float],
    fisher_specialist: Mapping[str, float],
    *,
    max_alpha: float = 0.5,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Compute a fixed conservative specialist weight per target module.

    Fisher values must already have been calculated on train-only data.  No
    validation scores or predictions are consulted here.
    """

    if not 0.0 < max_alpha <= 1.0:
        raise ValueError("max_alpha must be in (0, 1]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if set(fisher_original) != set(fisher_specialist):
        raise ValueError("Fisher maps must contain the same target modules")
    result: dict[str, float] = {}
    for module in sorted(fisher_original):
        broad = float(fisher_original[module])
        specialist = float(fisher_specialist[module])
        if not np.isfinite(broad) or not np.isfinite(specialist) or broad < 0 or specialist < 0:
            raise ValueError(f"invalid Fisher value for {module}")
        result[module] = min(max_alpha, specialist / (broad + specialist + epsilon))
    return result


def merge_effective_deltas(
    original: ArrayMap,
    specialist: ArrayMap,
    alpha: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Interpolate effective deltas without averaging LoRA A/B factors."""

    if set(original) != set(specialist) or set(original) != set(alpha):
        raise ValueError("delta and alpha maps must contain identical modules")
    merged: dict[str, np.ndarray] = {}
    for module in sorted(original):
        broad = _finite_matrix(original[module], name=f"original[{module}]")
        expert = _finite_matrix(specialist[module], name=f"specialist[{module}]")
        if broad.shape != expert.shape:
            raise ValueError(f"delta shape mismatch for {module}")
        weight = float(alpha[module])
        if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError(f"alpha for {module} must be in [0, 1]")
        merged[module] = (1.0 - weight) * broad + weight * expert
    return merged


def deterministic_svd_compress(delta: np.ndarray, rank: int) -> tuple[np.ndarray, float]:
    """Compress a delta to rank with deterministic sign convention.

    Returns ``(compressed_delta, relative_frobenius_error)``.  Sign flips in an
    SVD do not change the product, but fixing them makes serialized factors and
    manifests reproducible across runs.
    """

    matrix = _finite_matrix(delta, name="delta")
    if rank <= 0:
        raise ValueError("rank must be positive")
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    kept = min(rank, singular.size)
    u = u[:, :kept].copy()
    vh = vh[:kept, :].copy()
    singular = singular[:kept]
    for index in range(kept):
        pivot = int(np.argmax(np.abs(u[:, index])))
        if u[pivot, index] < 0:
            u[:, index] *= -1.0
            vh[index, :] *= -1.0
    compressed = (u * singular) @ vh
    denominator = max(float(np.linalg.norm(matrix, ord="fro")), np.finfo(float).eps)
    error = float(np.linalg.norm(matrix - compressed, ord="fro") / denominator)
    return compressed, error


def factorize_delta(
    delta: np.ndarray,
    *,
    rank: int,
    lora_alpha: float,
    use_rslora: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(A, B, reconstruction_error)`` for a compressed delta."""

    if lora_alpha <= 0:
        raise ValueError("lora_alpha must be positive")
    matrix = _finite_matrix(delta, name="delta")
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    kept = min(rank, singular.size)
    u = u[:, :kept].copy()
    vh = vh[:kept, :].copy()
    singular = singular[:kept]
    for index in range(kept):
        pivot = int(np.argmax(np.abs(u[:, index])))
        if u[pivot, index] < 0:
            u[:, index] *= -1.0
            vh[index, :] *= -1.0
    scale = float(lora_alpha) / (np.sqrt(rank) if use_rslora else rank)
    b = u * singular
    a = vh / scale
    reconstructed = scale * (b @ a)
    denominator = max(float(np.linalg.norm(matrix, ord="fro")), np.finfo(float).eps)
    error = float(np.linalg.norm(matrix - reconstructed, ord="fro") / denominator)
    return a, b, error

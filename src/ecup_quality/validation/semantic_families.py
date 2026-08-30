from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ecup_quality.data.text import normalize_text

JOINT_STRATA = (
    "БАД|0",
    "БАД|1",
    "Легковоспламеняющиеся|0",
    "Легковоспламеняющиеся|1",
)

_TOKEN = re.compile(
    r"[a-zа-яё]+(?:-?\d+)+[a-zа-яё]*|"
    r"\d+(?:[.,]\d+)?(?:[a-zа-яё]+)?|"
    r"[a-zа-яё]+",
    re.IGNORECASE,
)
_QUANTITY = re.compile(
    r"^(?P<number>\d+(?:[.,]\d+)?)(?P<unit>мкг|мг|кг|г|мл|л|мм|см|м|шт|вт|w|v)?$",
    re.IGNORECASE,
)


def masked_quantity_name(value: object) -> str:
    """Mask standalone quantities while preserving alphanumeric model tokens.

    ``B12``, ``D3``, ``CoQ10`` and ``K-206`` remain distinct. Standalone
    quantities such as ``450 мл`` or ``10шт`` become a versioned ``#`` marker.
    The function returns an empty key unless the title is specific enough to be
    considered for a corroborated family edge.
    """

    raw = html.unescape(str(value or ""))
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = unicodedata.normalize("NFKC", raw).lower().replace("ё", "е")
    tokens = _TOKEN.findall(raw)
    output: list[str] = []
    masked = False
    for token in tokens:
        token = token.lower().replace("ё", "е")
        if token[0].isalpha() and any(char.isdigit() for char in token):
            output.append(token)
            continue
        match = _QUANTITY.fullmatch(token)
        if match:
            output.append("#")
            unit = match.group("unit")
            if unit:
                output.append(unit.lower())
            masked = True
        else:
            output.append(token)
    alpha_tokens = [token for token in output if token != "#" and token.isalpha()]
    alpha_chars = sum(len(token) for token in alpha_tokens)
    if not masked or len(alpha_tokens) < 3 or alpha_chars < 10:
        return ""
    return " ".join(output)


def char_shingles(value: object, *, width: int = 3) -> frozenset[str]:
    normalized = normalize_text(value)
    padded = f"  {normalized}  "
    if not normalized:
        return frozenset()
    return frozenset(padded[index : index + width] for index in range(len(padded) - width + 1))


def description_similarity(left: object, right: object) -> float:
    left_shingles = char_shingles(left)
    right_shingles = char_shingles(right)
    if not left_shingles or not right_shingles:
        return 0.0
    return len(left_shingles & right_shingles) / len(left_shingles | right_shingles)


def stable_rank(*parts: object) -> int:
    payload = "\0".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


@dataclass(frozen=True)
class BalancedAssignment:
    row_folds: np.ndarray
    component_folds: dict[str, int]
    fold_counts: np.ndarray
    objective: tuple[float, ...]
    trials: int


def _assignment_objective(counts: np.ndarray, totals: np.ndarray) -> tuple[float, ...]:
    targets = totals / counts.shape[0]
    deviations = np.abs(counts - targets[None, :])
    normalized = deviations / np.maximum(targets[None, :], 1.0)
    rare_index = 4  # total rows + JOINT_STRATA; flammable positive is last.
    return (
        float(deviations[:, rare_index].max()),
        float(deviations[:, 0].max()),
        float(normalized[:, 1:].max()),
        float(np.square(normalized).sum()),
    )


def assign_balanced_component_folds(
    frame: pd.DataFrame,
    *,
    component_column: str,
    n_splits: int,
    seed: int,
    trials: int = 128,
) -> BalancedAssignment:
    """Deterministically balance whole components over four category/label strata.

    Candidate scores never enter the assignment. Multiple fixed hash orderings
    are evaluated against a frozen objective rather than choosing a convenient
    random seed by hand.
    """

    required = {component_column, "category", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if n_splits < 2 or trials < 1:
        raise ValueError("n_splits must be >=2 and trials must be positive")
    strata = frame.category.astype(str) + "|" + frame.label.astype(str)
    unexpected = sorted(set(strata) - set(JOINT_STRATA))
    if unexpected:
        raise ValueError(f"unexpected joint strata: {unexpected}")

    components = frame[component_column].astype(str)
    unique_components = sorted(components.unique())
    component_index = {component: index for index, component in enumerate(unique_components)}
    vectors = np.zeros((len(unique_components), 1 + len(JOINT_STRATA)), dtype=np.int32)
    for component, stratum in zip(components, strata):
        index = component_index[component]
        vectors[index, 0] += 1
        vectors[index, 1 + JOINT_STRATA.index(stratum)] += 1
    totals = vectors.sum(axis=0).astype(np.float64)
    if np.any(totals[1:] < n_splits):
        raise ValueError(f"too few rows for stratification: {totals.tolist()}")

    target = totals / n_splits
    rarity = np.max(vectors[:, 1:] / np.maximum(target[None, 1:], 1.0), axis=1)
    best: tuple[tuple[float, ...], np.ndarray, np.ndarray] | None = None
    for trial in range(trials):
        order = sorted(
            range(len(unique_components)),
            key=lambda index: (
                -float(rarity[index]),
                -int(vectors[index, 0]),
                stable_rank(seed, trial, unique_components[index]),
                unique_components[index],
            ),
        )
        fold_counts = np.zeros((n_splits, vectors.shape[1]), dtype=np.int32)
        assignments = np.full(len(unique_components), -1, dtype=np.int16)
        for position, index in enumerate(order):
            candidate_folds = range(n_splits)
            if position < n_splits:
                # Seed each fold once. The hash-order trial controls which
                # high-impact component enters first; fold identity stays fixed.
                candidate_folds = (position,)
            choices: list[tuple[tuple[float, ...], int]] = []
            for fold in candidate_folds:
                proposed = fold_counts.copy()
                proposed[fold] += vectors[index]
                fractions = proposed / np.maximum(totals[None, :], 1.0)
                partial_score = (
                    float(np.std(fractions[:, 4])),
                    float(np.std(fractions[:, 1:], axis=0).mean()),
                    float(np.std(fractions[:, 0])),
                    float(proposed[fold, 0] / max(target[0], 1.0)),
                    float(stable_rank(seed, trial, unique_components[index], fold)),
                )
                choices.append((partial_score, fold))
            _, chosen = min(choices)
            assignments[index] = chosen
            fold_counts[chosen] += vectors[index]
        objective = _assignment_objective(fold_counts, totals)
        candidate = (objective, assignments.copy(), fold_counts.copy())
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    objective, assignments, fold_counts = best
    component_folds = {
        component: int(assignments[index])
        for component, index in component_index.items()
    }
    row_folds = components.map(component_folds).to_numpy(dtype=np.int16)
    component_fold_counts = pd.DataFrame({"component": components, "fold": row_folds}).groupby(
        "component"
    ).fold.nunique()
    if int(component_fold_counts.max()) != 1:
        raise RuntimeError("component assignment invariant failed")
    return BalancedAssignment(
        row_folds=row_folds,
        component_folds=component_folds,
        fold_counts=fold_counts,
        objective=objective,
        trials=trials,
    )

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "540_qwen35_mixed_family_bad_pairwise"
sys.path.insert(0, str(EXPERIMENT))

import pair_selector as selector


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[int]]:
    frame = pd.DataFrame(
        {
            "id": ["p0", "n0", "p1", "n1", "unsafe", "holdout"],
            "name": [
                "Омега витамин премиум",
                "Омега витамин премиум",
                "Магний цитрат здоровье",
                "Магний цитрат здоровье плюс",
                "Магний цитрат здоровье",
                "Омега витамин премиум",
            ],
            "description": [""] * 6,
            "category": [selector.BAD] * 6,
            "label": [1, 0, 1, 0, 0, 0],
        }
    )
    guard = pd.DataFrame(
        {
            "id": frame.id,
            "connected_component": ["a", "a", "b", "c", "u", "h"],
            "safe_for_selection": [True, True, True, True, False, True],
        }
    )
    folds = np.asarray([1, 1, 1, 1, 1, 0], dtype=np.int8)
    return frame, guard, folds, [0, 1, 2, 3, 4]


def test_exp540_family_topology_is_label_blind_under_relabeling() -> None:
    frame, guard, folds, _ = _fixture()
    nearest_a, edges_a, audit_a = selector.build_label_blind_topology(
        frame, guard, folds, holdout_fold=0
    )
    relabeled = frame.copy()
    relabeled["label"] = 1 - relabeled["label"]
    nearest_b, edges_b, audit_b = selector.build_label_blind_topology(
        relabeled, guard, folds, holdout_fold=0
    )

    assert nearest_a == nearest_b
    assert edges_a == edges_b
    assert audit_a == audit_b
    assert audit_a["label_blind"] is True


def test_exp540_pair_plan_uses_only_safe_outer_train_donors() -> None:
    frame, guard, folds, records = _fixture()
    ordered, manifest, audit = selector.build_pair_plan(
        frame, guard, folds, records, holdout_fold=0
    )

    assert Counter(ordered) == Counter(records)
    assert audit["multiplicity_unchanged"] is True
    assert audit["steps_unchanged"] is True
    assert audit["unsafe_component_pairs"] == 0
    assert audit["outer_validation_pairs"] == 0
    assert all(row["positive_id"] != "unsafe" for row in manifest)
    assert all(row["negative_id"] not in {"unsafe", "holdout"} for row in manifest)


def test_exp540_same_family_precedes_label_blind_nearest_family() -> None:
    frame, guard, folds, records = _fixture()
    _, manifest, _ = selector.build_pair_plan(frame, guard, folds, records, holdout_fold=0)

    relations = {(row["positive_id"], row["negative_id"]): row["relation"] for row in manifest}
    assert relations[("p0", "n0")] == "same_family"
    assert relations[("p1", "n1")] == "nearest_family"


def test_exp540_pair_caps_are_enforced_in_realized_manifest() -> None:
    frame, guard, folds, records = _fixture()
    _, manifest, audit = selector.build_pair_plan(frame, guard, folds, records, holdout_fold=0)
    positive_counts = Counter(row["positive_id"] for row in manifest)
    negative_counts = Counter(row["negative_id"] for row in manifest)

    assert max(positive_counts.values(), default=0) <= selector.MAX_NEGATIVES_PER_POSITIVE
    assert max(negative_counts.values(), default=0) <= selector.MAX_NEGATIVE_REUSE
    assert audit["max_pairs_per_positive"] <= 4
    assert audit["max_negative_reuse"] <= 4


def test_exp540_insufficient_pairs_fail_closed() -> None:
    frame, guard, folds, records = _fixture()
    _, _, audit = selector.build_pair_plan(frame, guard, folds, records, holdout_fold=0)

    assert audit["decision"] == "NO_GO"
    assert "insufficient_realized_pairs" in audit["failures"]

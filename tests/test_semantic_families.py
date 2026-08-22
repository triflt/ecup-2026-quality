from __future__ import annotations

import numpy as np
import pandas as pd

from ecup_quality.validation.semantic_families import (
    JOINT_STRATA,
    assign_balanced_component_folds,
    masked_quantity_name,
)


def test_quantity_mask_preserves_alphanumeric_product_tokens() -> None:
    examples = {
        "Витамин B12 метилкобаламин 450 мкг": "b12",
        "Витамин D3 холекальциферол 2000 МЕ": "d3",
        "Коэнзим CoQ10 комплекс 100 мг": "coq10",
        "Подшипник K-206 усиленный 10шт": "k-206",
    }
    for title, token in examples.items():
        masked = masked_quantity_name(title)
        assert token in masked
        assert "#" in masked


def test_quantity_mask_masks_standalone_and_attached_quantities() -> None:
    assert masked_quantity_name("Газовый баллон туристический 450 мл") == (
        "газовый баллон туристический # мл"
    )
    assert masked_quantity_name("Газовый баллон туристический 10шт") == (
        "газовый баллон туристический # шт"
    )


def _balanced_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    component_id = 0
    # The rare stratum deliberately contains uneven component sizes. A row-wise
    # splitter would hide the constraint; the optimizer must keep each family whole.
    for stratum, component_sizes in {
        "БАД|0": [3] * 49 + [1] * 49,
        "БАД|1": [4] * 42 + [2] * 28,
        "Легковоспламеняющиеся|0": [5] * 35 + [2] * 35,
        "Легковоспламеняющиеся|1": [3] * 28 + [2] * 42 + [1] * 28,
    }.items():
        category, label_text = stratum.split("|")
        for size in component_sizes:
            component = f"component-{component_id:04d}"
            component_id += 1
            rows.extend(
                {
                    "id": f"row-{len(rows):05d}",
                    "component": component,
                    "category": category,
                    "label": int(label_text),
                }
                for _ in range(size)
            )
    return pd.DataFrame(rows)


def test_balanced_assignment_is_deterministic_and_permutation_invariant() -> None:
    frame = _balanced_frame()
    first = assign_balanced_component_folds(
        frame,
        component_column="component",
        n_splits=7,
        seed=20260822,
        trials=64,
    )
    second = assign_balanced_component_folds(
        frame.sample(frac=1.0, random_state=73).reset_index(drop=True),
        component_column="component",
        n_splits=7,
        seed=20260822,
        trials=64,
    )
    assert first.component_folds == second.component_folds

    audit = frame.assign(fold=first.row_folds)
    assert audit.groupby("component").fold.nunique().max() == 1
    observed = (
        audit.assign(stratum=audit.category.astype(str) + "|" + audit.label.astype(str))
        .groupby(["fold", "stratum"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=JOINT_STRATA)
    )
    rare_target = observed["Легковоспламеняющиеся|1"].sum() / 7
    assert np.abs(observed["Легковоспламеняющиеся|1"] - rare_target).max() <= 2


def test_assignment_rejects_unknown_joint_stratum() -> None:
    frame = _balanced_frame()
    frame.loc[0, "category"] = "unknown"
    try:
        assign_balanced_component_folds(
            frame,
            component_column="component",
            n_splits=7,
            seed=20260822,
        )
    except ValueError as error:
        assert "unexpected joint strata" in str(error)
    else:
        raise AssertionError("unknown stratum must fail closed")

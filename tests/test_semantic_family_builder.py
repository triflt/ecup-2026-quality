from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation/build_semantic_family_v1.py"
SPEC = importlib.util.spec_from_file_location("semantic_family_builder", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def test_digit_masked_name_is_conservative() -> None:
    assert builder.digit_masked_name("Газовый баллон 450 мл") == "газовый баллон # мл"
    assert builder.digit_masked_name("Баллон 450 мл") == ""
    assert builder.digit_masked_name("Протеин сывороточный шоколад") == ""


def test_cross_category_tokens_form_one_component() -> None:
    dsu = builder.DisjointSet(4)
    duplicate = builder.union_rows_by_tokens(
        dsu,
        [["image-a"], ["image-b"], ["image-a", "image-c"], ["image-d"]],
    )
    assert dsu.find(0) == dsu.find(2)
    assert dsu.find(0) != dsu.find(1)
    assert duplicate.tolist() == [True, False, True, False]


def test_component_holdout_and_dev_folds_do_not_split_groups() -> None:
    rows = []
    groups = []
    for category in ("БАД", "Легковоспламеняющиеся"):
        for label in (0, 1):
            for group_index in range(14):
                group = f"{category}-{label}-{group_index}"
                for repeat in range(2 if group_index % 3 == 0 else 1):
                    rows.append({
                        "id": f"{category}-{label}-{group_index}-{repeat}",
                        "category": category,
                        "label": label,
                    })
                    groups.append(group)
    frame = pd.DataFrame(rows)
    holdout, dev_fold = builder.assign_splits(
        frame=frame,
        groups=np.asarray(groups),
        holdout_splits=7,
        holdout_seed=20260822,
        dev_splits=5,
        dev_seed=314159,
        chosen_holdout_fold=3,
    )
    audit = pd.DataFrame({"group": groups, "holdout": holdout, "fold": dev_fold})
    assert audit.groupby("group").holdout.nunique().max() == 1
    assert audit.loc[~holdout].groupby("group").fold.nunique().max() == 1
    assert (dev_fold[holdout] == -1).all()
    assert (dev_fold[~holdout] >= 0).all()

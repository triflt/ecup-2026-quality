from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("exp694_train", HERE / "train_fold.py")
assert spec and spec.loader
TRAIN = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = TRAIN
spec.loader.exec_module(TRAIN)


def rows():
    return [
        {"id": "b", "global_index": 4, "category": "БАД", "label": 0},
        {
            "id": "n1",
            "global_index": 1,
            "category": TRAIN.FLAMMABLE,
            "label": 0,
            "teacher_score": 3.0,
            "teacher_outer_fold": 0,
        },
        {
            "id": "n2",
            "global_index": 2,
            "category": TRAIN.FLAMMABLE,
            "label": 0,
            "teacher_score": 1.0,
            "teacher_outer_fold": 0,
        },
        {
            "id": "p",
            "global_index": 3,
            "category": TRAIN.FLAMMABLE,
            "label": 1,
            "teacher_score": -2.0,
            "teacher_outer_fold": 0,
        },
    ]


def test_curriculum_changes_only_order_and_survives_parent_shuffle():
    source = rows()
    arranged = TRAIN.arrange_for_frozen_shuffle(source)
    indices = list(range(len(arranged)))
    random.Random(42).shuffle(indices)
    observed = [arranged[index] for index in indices]
    assert observed == sorted(source, key=TRAIN.curriculum_key)
    assert sorted(row["id"] for row in arranged) == sorted(row["id"] for row in source)


def test_teacher_signal_is_flammable_only_and_not_a_loss():
    assert callable(TRAIN.load_fold)
    assert "teacher" not in TRAIN.control.primary_loss.__name__

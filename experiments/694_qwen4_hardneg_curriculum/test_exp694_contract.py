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


def test_paired_curriculum_order_survives_parent_shuffle():
    source = rows()
    arranged = TRAIN.arrange_for_frozen_shuffle(source)
    indices = list(range(len(arranged)))
    random.Random(42).shuffle(indices)
    observed = [arranged[index] for index in indices]
    assert observed == sorted(source, key=TRAIN.curriculum_key)
    assert sorted(row["id"] for row in arranged) == sorted(row["id"] for row in source)


def test_teacher_signal_is_flammable_only_and_weights_hard_gold_bce():
    assert callable(TRAIN.load_fold)
    values = {row["id"]: TRAIN.frozen_hard_example_weight(type("R", (), row)()) for row in rows()}
    assert values["b"] == 1.0
    assert values["n1"] == TRAIN.MAX_HARD_EXAMPLE_WEIGHT
    assert 1.0 < values["n2"] < TRAIN.MAX_HARD_EXAMPLE_WEIGHT
    assert values["p"] > 1.0

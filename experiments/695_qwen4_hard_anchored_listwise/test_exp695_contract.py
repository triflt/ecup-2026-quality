from __future__ import annotations

import importlib.util
import inspect
import random
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("exp695_train", HERE / "train_fold.py")
assert spec and spec.loader
TRAIN = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = TRAIN
spec.loader.exec_module(TRAIN)


def row(label: int, score: float, category: str = TRAIN.FLAMMABLE):
    return SimpleNamespace(label=label, teacher_score=score, category=category)


def test_pairs_never_cross_hard_label_boundary_or_bad():
    rows = [row(1, 3.0), row(1, 1.0), row(0, 2.0), row(0, -1.0), row(1, 9.0, "БАД")]
    pairs = TRAIN.within_stratum_pairs(rows)
    assert {(left, right) for left, right, _ in pairs} == {(0, 1), (2, 3)}
    assert all(rows[left].label == rows[right].label for left, right, _ in pairs)


def test_literal_frozen_rank_formula_and_hard_is_not_downweighted():
    source = inspect.getsource(TRAIN.listwise_loss)
    candidate = inspect.getsource(TRAIN.candidate_loss)
    assert "torch.minimum" not in source
    assert "hard.detach" not in source
    assert "return hard + RANK_COEFFICIENT * listwise_loss" in candidate
    assert TRAIN.RANK_COEFFICIENT == 0.50


def test_both_arms_receive_identical_same_stratum_pair_batches():
    rows = [
        {"global_index": 0, "category": TRAIN.FLAMMABLE, "label": 0},
        {"global_index": 1, "category": TRAIN.FLAMMABLE, "label": 0},
        {"global_index": 2, "category": TRAIN.FLAMMABLE, "label": 1},
        {"global_index": 3, "category": TRAIN.FLAMMABLE, "label": 1},
        {"global_index": 4, "category": "БАД", "label": 0},
    ]
    arranged = TRAIN.arrange_matched_batches(rows)
    indices = list(range(len(rows)))
    random.Random(42).shuffle(indices)
    observed = [arranged[index] for index in indices]
    assert observed[:2][0]["label"] == observed[:2][1]["label"] == 0
    assert observed[2:4][0]["label"] == observed[2:4][1]["label"] == 1
    assert sorted(row["global_index"] for row in arranged) == list(range(5))

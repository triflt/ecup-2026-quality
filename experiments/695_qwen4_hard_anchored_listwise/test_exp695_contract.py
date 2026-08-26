from __future__ import annotations

import importlib.util
import inspect
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
    assert "return hard + RANK_COEFFICIENT * rank" in candidate
    assert TRAIN.RANK_COEFFICIENT == 0.50


def test_pair_selection_does_not_reorder_parent_rows():
    rows = [
        {"global_index": 0, "category": TRAIN.FLAMMABLE, "label": 0},
        {"global_index": 1, "category": TRAIN.FLAMMABLE, "label": 0},
        {"global_index": 2, "category": TRAIN.FLAMMABLE, "label": 1},
        {"global_index": 3, "category": TRAIN.FLAMMABLE, "label": 1},
        {"global_index": 4, "category": "БАД", "label": 0},
    ]
    observed = TRAIN.preserve_frozen_base_order(rows)
    assert observed == rows
    assert [row["global_index"] for row in observed] == list(range(5))
    pairs = TRAIN.within_stratum_pairs(
        [SimpleNamespace(**item, teacher_score=float(i)) for i, item in enumerate(observed)]
    )
    assert all(observed[left]["label"] == observed[right]["label"] for left, right, _ in pairs)


def test_train_contract_records_measured_cuda_peak(monkeypatch):
    calls: list[str] = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        reset_peak_memory_stats=lambda: calls.append("reset"),
        max_memory_allocated=lambda: 345678,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    TRAIN.reset_cuda_peak_memory()
    peak = TRAIN.measured_cuda_peak_memory_bytes()
    assert calls == ["reset"]
    assert isinstance(peak, int) and peak == 345678
    assert '"peak_gpu_memory_bytes": peak_gpu_memory_bytes' in inspect.getsource(TRAIN.run)

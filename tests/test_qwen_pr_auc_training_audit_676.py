from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/676_qwen_pr_auc_training_audit"


def load_module():
    spec = importlib.util.spec_from_file_location("experiment_676_run", EXPERIMENT / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUN = load_module()


def test_average_precision_groups_tied_scores() -> None:
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.8, 0.1]
    assert RUN.average_precision(labels, scores) == pytest.approx(5 / 6)


def test_average_precision_rejects_invalid_input() -> None:
    for labels, scores in (([], []), ([0, 0], [0.2, 0.1]), ([1], [math.inf])):
        try:
            RUN.average_precision(labels, scores)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid AP input was accepted")


def test_committed_metrics_use_tie_aware_contract() -> None:
    metrics = json.loads((EXPERIMENT / "results/metrics.json").read_text(encoding="utf-8"))
    assert metrics["metric_definition"]["equivalent"] == (
        "sklearn.metrics.average_precision_score"
    )
    assert metrics["metric_definition"]["ties"] == "all equal scores form one threshold group"
    folds = metrics["flammable_average_precision"]["per_fold"]
    assert [row["fold"] for row in folds] == [0, 3]
    assert all(row["average_precision"]["large"] > row["average_precision"]["small"] for row in folds)
    assert metrics["public_used"] is False
    assert metrics["sealed_rows"] == 0
    assert metrics["training_recipe"]["final_checkpoint_only"] is True
    assert metrics["decision"].endswith("WAIT_FOR_TERMINAL_659")

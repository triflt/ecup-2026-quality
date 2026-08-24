from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/659_qwen36_27b_flammable_equal_logit_blend"
PARENT = ROOT / "experiments/654_qwen36_27b_class_only_lora/evaluate_screen.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_659", EXPERIMENT / "evaluate_screen.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVALUATE = load_module()


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_equal_blend_changes_only_flammable(tmp_path: Path) -> None:
    registry_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    large_rows: list[dict[str, object]] = []
    specifications = [
        (0, "БАД", 1, 1.0, -2.0),
        (0, "БАД", 0, -1.0, 2.0),
        (0, "Легковоспламеняющиеся", 1, -0.2, 1.0),
        (0, "Легковоспламеняющиеся", 0, -1.0, -1.0),
        (3, "БАД", 1, 1.0, -2.0),
        (3, "БАД", 0, -1.0, 2.0),
        (3, "Легковоспламеняющиеся", 1, -0.2, 1.0),
        (3, "Легковоспламеняющиеся", 0, -1.0, -1.0),
    ]
    for index, (fold, category, label, baseline_score, large_score) in enumerate(specifications):
        identifier = f"{index:02d}"
        registry_rows.append(
            {
                "id": identifier,
                "split": "development",
                "development_fold": fold,
                "category": category,
                "label": label,
            }
        )
        common = {
            "global_index": index,
            "id": identifier,
            "fold": fold,
            "category": category,
        }
        baseline_rows.append(
            {
                **common,
                "score": baseline_score,
                "prediction": int(baseline_score >= 0.0),
            }
        )
        large_rows.append({**common, "score": large_score, "prediction": int(large_score >= 0.0)})

    registry = tmp_path / "folds.csv"
    baseline = tmp_path / "baseline.jsonl"
    large = tmp_path / "large.jsonl"
    output = tmp_path / "audit.json"
    pd.DataFrame(registry_rows).to_csv(registry, index=False)
    write_jsonl(baseline, baseline_rows)
    write_jsonl(large, large_rows)

    result = EVALUATE.evaluate(
        registry_path=registry,
        baseline_paths=[baseline],
        large_paths=[large],
        output_path=output,
        parent_evaluator_path=PARENT,
    )

    assert result["passed"] is True
    assert result["categories"]["БАД"]["delta"] == 0.0
    assert result["categories"]["Легковоспламеняющиеся"]["delta"] > 0.0
    assert result["corrected"] == 2
    assert result["regressed"] == 0
    assert result["weights"] == {"641": 0.5, "654": 0.5}


def test_full_evaluator_requires_exact_screen_gate() -> None:
    spec = importlib.util.spec_from_file_location("exp659_full", EXPERIMENT / "evaluate_full.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gate = module.validate_screen_gate(EXPERIMENT / "results/screen_acceptance_audit.json")
    assert gate["decision"] == "OPEN_REMAINING_FOLDS"
    assert module.FOLDS == (0, 1, 2, 3, 4)
    assert module.BASELINE_WEIGHT == module.LARGE_WEIGHT == 0.5
    assert module.THRESHOLD == 0.0
    assert module.screen_fold_deltas(gate) == {
        "0": gate["folds"]["0"]["delta"],
        "3": gate["folds"]["3"]["delta"],
    }


def test_average_precision_handles_tied_scores_at_one_threshold() -> None:
    spec = importlib.util.spec_from_file_location("exp659_full_ap", EXPERIMENT / "evaluate_full.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
    scores = np.asarray([0.9, 0.8, 0.8, 0.1], dtype=np.float64)
    # First positive contributes precision 1.0 at recall 0.5; the tied second
    # threshold contributes precision 2/3 for the remaining recall 0.5.
    assert abs(module.average_precision(labels, scores) - (0.5 + 1.0 / 3.0)) < 1e-12

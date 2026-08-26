from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"exp693_{name}", HERE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = load("train_fold")
EVAL = load("evaluate")


def teacher_row(**overrides):
    row = {
        "id": "a",
        "global_index": 1,
        "category": TRAIN.FLAMMABLE,
        "label": 1,
        "teacher_outer_fold": 2,
        "teacher_verdict": 1,
        "sold_object": "fuel_consumable",
        "substance": "gas",
        "relation": "primary_sold_object",
        "evidence_ids": ["e1"],
    }
    row.update(overrides)
    return row


def test_teacher_is_flammable_outer_train_only():
    assert (
        TRAIN.validate_teacher_contract(
            {
                "teacher_contract": {
                    "schema": "qwen27_all_outer_safe_v1",
                    "teacher_recipe": "qwen27-all",
                    "outer_fold": 2,
                    "train_scope": "outer_train_only",
                    "outer_validation_labels_read": 0,
                    "target_category": TRAIN.FLAMMABLE,
                    "artifact_sha256": "a" * 64,
                }
            },
            2,
        )
        == "a" * 64
    )
    TRAIN.validate_teacher_rows([teacher_row()], [{"id": "b", "global_index": 2}], 2)
    try:
        TRAIN.validate_teacher_rows(
            [teacher_row(category="БАД")], [{"id": "b", "global_index": 2}], 2
        )
    except ValueError as error:
        assert "outside flammable" in str(error)
    else:
        raise AssertionError("BAD teacher target was accepted")


def test_hard_anchor_is_full_and_aux_is_additive():
    assert TRAIN.combine_losses(3.0, 2.0, mode="hard_bce_control") == 3.0
    assert TRAIN.combine_losses(3.0, 2.0, mode="causal_candidate") == 3.2
    target = TRAIN.structured_target(SimpleNamespace(**teacher_row()))
    assert '"sold_object":"fuel_consumable"' in target


def test_evaluator_reports_required_metrics_and_bad_exact():
    rows = [
        {
            "label": 1,
            "category": "БАД",
            "baseline": 1,
            "control": 1,
            "candidate": 1,
            "control_score": 1.0,
            "candidate_score": 1.0,
            "component_size": 1,
        },
        {
            "label": 1,
            "category": TRAIN.FLAMMABLE,
            "baseline": 0,
            "control": 0,
            "candidate": 1,
            "control_score": 0.1,
            "candidate_score": 0.9,
            "component_size": 1,
        },
        {
            "label": 0,
            "category": TRAIN.FLAMMABLE,
            "baseline": 0,
            "control": 0,
            "candidate": 0,
            "control_score": 0.1,
            "candidate_score": 0.2,
            "component_size": 2,
        },
    ]
    report = EVAL.evaluate_rows(rows)
    assert report["bad_exact"] is True
    assert report["corrected"] == 1
    assert report["singleton_delta"] == 1
    assert "tie_aware_ap" in report and "macro" in report

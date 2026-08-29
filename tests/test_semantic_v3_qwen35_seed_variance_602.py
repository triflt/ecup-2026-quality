from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/602_semantic_v3_qwen35_seed_variance"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load("exp602_seed_protocol_test", EXP / "seed_protocol.py")
trainer = _load("exp602_trainer_test", EXP / "train_seed.py")
evaluator = _load("exp602_evaluator_test", EXP / "evaluate_seed_variance.py")


def test_predeclared_grid_is_exactly_three_new_seeds_by_five_folds() -> None:
    assert protocol.REFERENCE_SEED == 42
    assert protocol.NEW_SEEDS == (31415, 271828, 161803)
    assert protocol.seed_job_manifest() == [
        {"seed": seed, "fold": fold, "sealed_rows": 0}
        for seed in protocol.NEW_SEEDS
        for fold in protocol.FOLDS
    ]
    assert len(protocol.seed_job_manifest()) == 15


@pytest.mark.parametrize("seed", protocol.SEEDS)
@pytest.mark.parametrize("fold", protocol.FOLDS)
def test_seed_is_the_only_permitted_recipe_change(seed: int, fold: int) -> None:
    environment = trainer.configure_environment(seed=seed, fold=fold, environment={})
    assert environment["SEED"] == str(seed)
    assert environment["HOLDOUT_FOLD"] == str(fold)
    assert environment["FULL_TRAIN"] == "0"
    assert environment["TRAINING_MODE"] == "hard"
    assert environment["MODEL_CLASS"] == "multimodal"
    assert environment["USE_CHAT_BATCH"] == "1"
    assert environment["DESCRIPTION_LIMIT"] == "1800"
    with pytest.raises(ValueError, match="non-parent switch"):
        trainer.configure_environment(
            seed=seed, fold=fold, environment={"SOFT_TARGETS": "/tmp/not-allowed.csv"}
        )


def test_shared_exp600_interface_is_present_and_strict() -> None:
    shared_protocol, shared_trainer = protocol.load_exp600_dependencies()
    assert shared_protocol.EXPERIMENT_ID == "600"
    assert shared_protocol.PROTOCOL_VERSION == "semantic_family_v3"
    assert shared_protocol.DEVELOPMENT_ROWS == 11_118
    assert shared_protocol.SEALED_ROWS == 1_853
    assert set(shared_trainer.PARENT_SHA256) == {"original", "specialist"}


def test_missing_shared_runtime_inputs_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="shared experiment-600 runtime inputs"):
        protocol.runtime_input_paths(tmp_path)


def test_reference_seed_accepts_exp600_original_contract(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "id": ["a"],
            "category": ["БАД"],
            "label": [1],
            "fold": [0],
            "lora_score": [0.5],
        }
    )
    prediction_path = tmp_path / "lora_holdout_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    contract = {
        "experiment_id": "600",
        "component": "original",
        "outer_fold": 0,
        "decision": "GO",
        "sealed_rows_in_predictions": 0,
        "predictions_sha256": protocol.sha256_file(prediction_path),
    }
    (tmp_path / "output_contract.runtime.json").write_text(json.dumps(contract))

    actual = evaluator._read_completed_predictions(tmp_path, seed=42, fold=0)

    assert actual["id"].tolist() == ["a"]


def test_fixed_probability_mean_and_leave_one_fold_out_calibration() -> None:
    labels = np.asarray([0, 1] * 5, dtype=np.int8)
    categories = np.asarray(["БАД"] * 10)
    folds = np.repeat(np.arange(5, dtype=np.int8), 2)
    scores = np.asarray([0.1, 0.9] * 5, dtype=np.float64)
    predictions, calibration = protocol.strictly_nested_predictions(
        probabilities=scores, labels=labels, categories=categories, folds=folds
    )
    assert predictions.tolist() == labels.tolist()
    assert len(calibration["БАД"]) == 5
    assert all(row["calibration_rows"] == 8 for row in calibration["БАД"])
    assert np.allclose(
        np.mean(np.stack([scores, scores, scores, scores]), axis=0), scores
    )


def test_acceptance_gate_includes_all_predeclared_rules() -> None:
    labels = np.asarray([0, 1] * 10, dtype=np.int8)
    categories = np.asarray(["БАД"] * 10 + ["Легковоспламеняющиеся"] * 10)
    folds = np.repeat(np.arange(5, dtype=np.int8), 4)
    components = np.asarray([f"c{index}" for index in range(20)])
    baseline = np.zeros(20, dtype=np.int8)
    candidate = labels.copy()
    report = protocol.audit_candidate(
        name="synthetic", labels=labels, categories=categories, folds=folds,
        components=components, baseline=baseline, candidate=candidate,
        calibration={},
    )
    assert set(report["acceptance"]) == {
        "macro_delta_at_least_0_003",
        "wins_at_least_4_of_5",
        "no_category_drop_over_0_005",
        "bootstrap_positive",
        "corrected_to_regressed_at_least_1_5",
    }
    assert report["accepted"] is True


def test_card_and_metrics_record_completed_no_go_grid() -> None:
    card = tomllib.loads((EXP / "experiment.toml").read_text(encoding="utf-8"))
    metrics = json.loads((EXP / "results/metrics.json").read_text(encoding="utf-8"))
    assert card["execution"]["new_seed_jobs"] == 15
    assert card["ensemble"]["weights"] == [0.25, 0.25, 0.25, 0.25]
    assert card["ensemble"]["weight_tuning"] is False
    assert metrics["launched"] is True
    assert metrics["status"] == "complete_no_go"
    assert metrics["completed_jobs"] == 15
    assert metrics["working_jobs"] == 0
    assert metrics["fully_nested_meta_validation"] is False
    assert metrics["decision"] == "NO_GO"
    assert metrics["sealed_holdout_used"] is False


def test_public_files_contain_no_private_infrastructure_references() -> None:
    public_paths = [
        EXP / "README.md",
        EXP / "experiment.toml",
        EXP / "artifacts/README.md",
        EXP / "results/metrics.json",
    ]
    forbidden = ("s3" + "-msk", "tin" + "koff", "ml" + " core", "secrettoken", "task_id", "job_id")
    for path in public_paths:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(token in text for token in forbidden), path

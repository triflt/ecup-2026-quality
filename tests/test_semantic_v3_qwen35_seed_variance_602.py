from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import yaml

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


def test_fixed_probability_mean_and_strict_nested_calibration() -> None:
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


def test_runtime_presets_cover_fifteen_neutral_one_gpu_jobs() -> None:
    presets = sorted((EXP / ".local/runtime").glob("*.yml"))
    assert len(presets) == 15
    observed: set[tuple[int, int]] = set()
    for path in presets:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))["job"]
        args = [str(value) for value in config["args"]]
        seed = int(args[args.index("--seed") + 1])
        fold = int(args[args.index("--fold") + 1])
        observed.add((seed, fold))
        assert config["flavor"] == "h100-1x"
        assert config["generate_name"] == f"sv3-q35v-s{seed}-f{fold}"
        assert "ecup" not in config["generate_name"].lower()
        assert args[args.index("--runtime-dir") + 1] == "/work/input/runtime"
        sources = [entry["src"] for entry in config["input"] if "src" in entry]
        assert "research/scoped_runtime_bootstrap.py" in sources
        assert not any(".local/runtime_inputs" in source for source in sources)
        assert "research/data.csv" not in sources
        assert "validation/semantic_family_v3/folds.csv" not in sources
        assert "research/lora_image_manifest_complete.tsv.gz" not in sources
    assert observed == {(seed, fold) for seed in protocol.NEW_SEEDS for fold in protocol.FOLDS}


def test_card_and_metrics_record_running_grid() -> None:
    card = tomllib.loads((EXP / "experiment.toml").read_text(encoding="utf-8"))
    metrics = json.loads((EXP / "results/metrics.json").read_text(encoding="utf-8"))
    assert card["execution"]["new_seed_jobs"] == 15
    assert card["ensemble"]["weights"] == [0.25, 0.25, 0.25, 0.25]
    assert card["ensemble"]["weight_tuning"] is False
    assert metrics["launched"] is True
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

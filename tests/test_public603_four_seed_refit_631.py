from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/631_public603_four_seed_refit"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_full_seed_refit_is_exact_and_bounded() -> None:
    trainer = load_module("exp631_trainer_test", EXPERIMENT / "train_full_seed.py")
    assert trainer.ALLOWED_SEEDS == (161803, 271828)
    assert len(trainer.PARENT_SHA256) == 64
    config = tomllib.loads((EXPERIMENT / "experiment.toml").read_text(encoding="utf-8"))
    assert config["dependency"]["development_experiment"] == "603"
    assert config["training"]["new_full_data_seeds"] == [271828, 161803]
    assert config["training"]["gpu_per_job"] == 1
    assert config["ensemble"]["weights"] == [0.25, 0.25, 0.25, 0.25]
    assert config["ensemble"]["weight_tuning"] is False
    assert config["data"]["public_tuning"] is False


def test_submission_runtime_keeps_original_route_and_four_seeds() -> None:
    source = (EXPERIMENT / "submission/run.py").read_text(encoding="utf-8")
    for seed in ("seed31415", "seed271828", "seed161803"):
        assert seed in source
    assert "probability_sum / len(adapter_names)" in source
    assert '"weight_base": 0.50' in source
    assert '"weight_qwen3vl": 0.25' in source
    assert '"weight_qwen35": 0.25' in source
    assert '"weight_base": 0.15' in source
    assert '"weight_qwen3vl": 0.10' in source
    assert '"weight_qwen35": 0.75' in source
    assert '"threshold": 0.27193570137023926' in source
    assert '"threshold": 0.953912615776062' in source


def test_builder_accepts_only_verified_manifest_and_runtime() -> None:
    builder = load_module("exp631_builder_test", EXPERIMENT / "build_submission.py")
    metrics = json.loads((EXPERIMENT / "results/metrics.json").read_text(encoding="utf-8"))
    assert metrics["submission_ready"] is True
    assert metrics["schema_valid"] is True
    assert metrics["projected_public_minutes"] <= 20.0
    assert metrics["projected_private_minutes"] <= 40.0
    builder.verify_production_manifest()

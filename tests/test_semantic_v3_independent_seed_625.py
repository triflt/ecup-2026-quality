from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/625_semantic_v3_independent_seed"


def load_runner():
    spec = importlib.util.spec_from_file_location("exp625_runner_test", EXP / "run_fold.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def signed_manifest(runner, spec, **updates):
    value = {
        "schema_version": "exp624_route_recipe_v1",
        "experiment_id": "624",
        "decision": "ACCEPT",
        "winner_component_experiment": "623",
        "validation": "semantic_family_v3",
        "source_seed": 42,
        "threshold_contract_sha256": spec["frozen_full_threshold_contract_sha256"],
        "uses_sealed_holdout": False,
        "reference_route_weights_unchanged": True,
        "source_recipe_sha256": spec["source_recipe_sha256"],
    }
    value.update(updates)
    value["manifest_sha256"] = runner.canonical_sha256(value)
    return value


def test_frozen_spec_changes_only_seed_and_binds_all_five_folds():
    spec = json.loads((EXP / "frozen_spec.json").read_text())
    assert spec["source_seed"] == 42
    assert spec["independent_seed"] == 31415
    assert spec["only_changed_factor"] == "training_seed"
    assert spec["outer_folds"] == [0, 1, 2, 3, 4]
    assert spec["one_gpu_per_fold"] is True
    assert spec["sealed_rows_allowed"] == 0
    assert len(spec["frozen_full_threshold_contract_sha256"]) == 64
    assert all(
        value == "experiment_623_exact"
        for key, value in spec["immutable_recipe"].items()
        if key != "thresholds"
    )


def test_source_recipe_checksums_are_current():
    spec = json.loads((EXP / "frozen_spec.json").read_text())
    source = ROOT / "experiments/623_semantic_v3_multitask_span_head"
    observed = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in spec["source_recipe_sha256"]
    }
    assert observed == spec["source_recipe_sha256"]


def test_route_manifest_is_fail_closed(tmp_path: Path):
    runner = load_runner()
    spec = runner.load_spec()
    valid = signed_manifest(runner, spec)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(valid))
    assert runner.verify_route_manifest(path, spec)["winner_component_experiment"] == "623"

    invalid = signed_manifest(runner, spec, winner_component_experiment="621")
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="route manifest mismatch"):
        runner.verify_route_manifest(path, spec)


def test_route_manifest_rejects_tampering(tmp_path: Path):
    runner = load_runner()
    spec = runner.load_spec()
    manifest = signed_manifest(runner, spec)
    manifest["reference_route_weights_unchanged"] = False
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        runner.verify_route_manifest(path, spec)


def test_terminal_skip_claims_no_acceptance():
    metrics = json.loads((EXP / "results/metrics.json").read_text())
    assert metrics["validation_complete"] is False
    assert metrics["metrics"] == {}
    assert metrics["status"] == "skipped_by_gate"
    assert metrics["decision"] == "SKIPPED_NO_ACCEPTED_624_RECIPE"


def test_wrapper_enforces_single_gpu_and_overrides_only_seed():
    source = (EXP / "run_fold.py").read_text()
    assert "torch.cuda.device_count() != 1" in source
    assert 'module.SEED = int(spec["independent_seed"])' in source
    for name in (
        "architecture",
        "loss",
        "data_and_selector",
        "thresholds",
        "renderer",
        "training_schedule",
    ):
        assert name in json.loads((EXP / "frozen_spec.json").read_text())["immutable_recipe"]

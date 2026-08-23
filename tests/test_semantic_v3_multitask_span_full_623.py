from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/623_semantic_v3_multitask_span_head"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


full = _module("exp623_full_test", EXP / "evaluate_full.py")
fixtures = _module(
    "exp623_screen_fixture_for_full_test",
    ROOT / "tests/test_semantic_v3_multitask_span_screen_623.py",
)


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, bad_span: bool = False):
    registry_path, registry = fixtures._registry(tmp_path, monkeypatch)
    monkeypatch.setattr(full.parent_screen, "EXPECTED_FOLDS_SHA256", full.screen.sha256_file(registry_path))
    monkeypatch.setattr(full.parent_screen, "EXPECTED_BASELINE_PREDICTION_SHA256", None)
    baselines = fixtures._baselines(tmp_path, registry)
    threshold_path = tmp_path / "full-thresholds.json"
    full.freeze_full_thresholds(
        registry_path=registry_path,
        baseline_dirs=baselines,
        output_path=threshold_path,
    )
    monkeypatch.setattr(fixtures.screen, "SCREEN_FOLDS", full.FOLDS)
    runtimes = fixtures._runtimes(tmp_path, registry)
    artifacts = fixtures._artifacts(
        tmp_path,
        registry,
        runtimes,
        bad_exact_span=bad_span,
    )
    return registry_path, baselines, threshold_path, runtimes, artifacts


def test_freeze_full_thresholds_is_donor_only_self_hashed_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path, registry = fixtures._registry(tmp_path, monkeypatch)
    monkeypatch.setattr(full.parent_screen, "EXPECTED_FOLDS_SHA256", full.screen.sha256_file(registry_path))
    monkeypatch.setattr(full.parent_screen, "EXPECTED_BASELINE_PREDICTION_SHA256", None)
    baselines = fixtures._baselines(tmp_path, registry)
    output = tmp_path / "full-thresholds.json"
    result = full.freeze_full_thresholds(
        registry_path=registry_path,
        baseline_dirs=baselines,
        output_path=output,
    )
    assert result["candidate_inputs_read"] == 0
    assert result["sealed_rows"] == 0
    assert result["screen_folds"] == list(full.FOLDS)
    assert set(result["thresholds"]) == {str(fold) for fold in full.FOLDS}
    payload = dict(result)
    digest = payload.pop("threshold_contract_sha256")
    assert digest == full.screen.canonical_sha256(payload)
    with pytest.raises(FileExistsError, match="overwrite"):
        full.freeze_full_thresholds(
            registry_path=registry_path,
            baseline_dirs=baselines,
            output_path=output,
        )


def test_full_five_fold_acceptance_reports_all_frozen_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, baselines, thresholds, runtimes, artifacts = _setup(tmp_path, monkeypatch)
    result = full.evaluate_full(
        registry_path=registry,
        baseline_dirs=baselines,
        runtime_dirs=runtimes,
        artifact_dirs=artifacts,
        threshold_contract_path=thresholds,
        output_path=tmp_path / "full.json",
    )
    assert result["passed"] is True
    assert result["decision"] == "GO_INTEGRATE_624"
    assert result["folds_evaluated"] == list(full.FOLDS)
    assert result["winning_folds"] == 5
    assert result["macro_delta"] >= 0.003
    assert all(result["gates"].values())
    assert result["component_bootstrap"]["iterations"] == 10_000
    assert result["component_bootstrap"]["seed"] == 623_042
    assert result["component_bootstrap"]["probability_delta_positive"] >= 0.9
    assert result["grounding"]["overall"]["grounded_coverage"] == 1.0
    assert result["grounding"]["human_quality_evaluated"] is False
    assert result["grounding"]["human_quality"] is None
    assert result["sealed_rows_loaded"] == 0
    assert result["thresholds_tuned_after_candidate"] is False


def test_full_rejects_missing_fold_before_candidate_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, baselines, thresholds, runtimes, artifacts = _setup(tmp_path, monkeypatch)
    del artifacts[4]
    with pytest.raises(ValueError, match="folds 0..4"):
        full.evaluate_full(
            registry_path=registry,
            baseline_dirs=baselines,
            runtime_dirs=runtimes,
            artifact_dirs=artifacts,
            threshold_contract_path=thresholds,
            output_path=tmp_path / "full.json",
        )


def test_full_rejects_threshold_and_grounding_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, baselines, thresholds, runtimes, artifacts = _setup(tmp_path, monkeypatch)
    contract = json.loads(thresholds.read_text(encoding="utf-8"))
    contract["thresholds"]["4"][full.CATEGORIES[0]]["threshold"] = -999.0
    thresholds.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold_contract_sha256 mismatch"):
        full.evaluate_full(
            registry_path=registry,
            baseline_dirs=baselines,
            runtime_dirs=runtimes,
            artifact_dirs=artifacts,
            threshold_contract_path=thresholds,
            output_path=tmp_path / "full.json",
        )

    grounding_dir = tmp_path / "grounding"
    grounding_dir.mkdir()
    registry, baselines, thresholds, runtimes, artifacts = _setup(
        grounding_dir,
        monkeypatch,
        bad_span=True,
    )
    with pytest.raises(ValueError, match="exact substring"):
        full.evaluate_full(
            registry_path=registry,
            baseline_dirs=baselines,
            runtime_dirs=runtimes,
            artifact_dirs=artifacts,
            threshold_contract_path=thresholds,
            output_path=tmp_path / "grounding-full.json",
        )


def test_bootstrap_rejects_post_hoc_seed_or_iteration_changes() -> None:
    import numpy as np

    kwargs = {
        "labels": np.asarray([0, 1], dtype=np.int8),
        "categories": np.asarray(full.CATEGORIES),
        "components": np.asarray(["a", "b"]),
        "baseline": np.asarray([0, 1], dtype=np.int8),
        "candidate": np.asarray([0, 1], dtype=np.int8),
    }
    with pytest.raises(ValueError, match="frozen"):
        full.grouped_component_bootstrap(**kwargs, iterations=999)
    with pytest.raises(ValueError, match="frozen"):
        full.grouped_component_bootstrap(**kwargs, seed=1)

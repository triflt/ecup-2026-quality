from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/624_semantic_v3_route_integration"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _module("exp624_route_test", EXP / "evaluate.py")
seed625 = _module(
    "exp625_seed_manifest_test",
    ROOT / "experiments/625_semantic_v3_independent_seed/run_fold.py",
)


def _synthetic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, improve: bool = True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    labels: list[int] = []
    categories: list[str] = []
    folds: list[int] = []
    components: list[str] = []
    for fold in evaluator.FOLDS:
        for category in evaluator.CATEGORIES:
            for label in (0, 1):
                ids.append(f"{fold}-{category}-{label}")
                labels.append(label)
                categories.append(category)
                folds.append(fold)
                components.append(f"component-{fold}-{category}-{label}")
    labels_array = np.asarray(labels, dtype=np.int8)
    categories_array = np.asarray(categories)
    folds_array = np.asarray(folds, dtype=np.int8)
    visual = {
        "ids": np.asarray(ids),
        "labels": labels_array,
        "categories": categories_array,
        "folds": folds_array,
        "semantic_components": np.asarray(components),
        "robust_base_rank": np.zeros(len(ids), dtype=np.float64),
        "qwen3vl_rank": np.zeros(len(ids), dtype=np.float64),
    }
    registry = pd.DataFrame(
        {
            "id": ids,
            "category": categories,
            "label": labels,
            "semantic_component": components,
            "split": "development",
            "development_fold": folds,
        }
    )
    original_logits = np.where(labels_array == 1, -3.0, 3.0)
    candidate_logits = np.where(labels_array == 1, 3.0, -3.0)
    if not improve:
        candidate_logits = original_logits.copy()
    upstream = {
        "gates": {"all": True},
        "candidate_provenance": {str(fold): {"verified": True} for fold in evaluator.FOLDS},
    }
    monkeypatch.setattr(evaluator, "_verify_recipe", lambda spec: None)
    monkeypatch.setattr(evaluator, "_verify_accepted_623", lambda **kwargs: upstream)
    monkeypatch.setattr(evaluator, "EXPECTED_BASELINE_MACRO_F1", 2.0 / 3.0)
    monkeypatch.setattr(evaluator.route603, "load_visual_bundle", lambda **kwargs: visual)
    monkeypatch.setattr(evaluator.route603, "load_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr(
        evaluator,
        "_assemble_original_logits",
        lambda **kwargs: (original_logits, {str(fold): {} for fold in evaluator.FOLDS}),
    )
    monkeypatch.setattr(
        evaluator,
        "_assemble_candidate_logits",
        lambda **kwargs: (candidate_logits, {str(fold): {} for fold in evaluator.FOLDS}),
    )
    accepted = tmp_path / "accepted-623.json"
    threshold = tmp_path / "threshold.json"
    visual_path = tmp_path / "visual.npz"
    visual_contract = tmp_path / "visual.json"
    registry_path = tmp_path / "folds.csv"
    for path in (accepted, threshold, visual_path, visual_contract, registry_path):
        path.write_text("{}", encoding="utf-8")
    fold_dirs = {fold: tmp_path / f"fold-{fold}" for fold in evaluator.FOLDS}
    for path in fold_dirs.values():
        path.mkdir()
    return {
        "visual_bundle_path": visual_path,
        "visual_contract_path": visual_contract,
        "registry_path": registry_path,
        "baseline_dirs": fold_dirs,
        "runtime_dirs": fold_dirs,
        "artifact_dirs": fold_dirs,
        "threshold_contract_path": threshold,
        "accepted_623_report_path": accepted,
        "output_path": tmp_path / "route.json",
        "manifest_path": tmp_path / "manifest.json",
        "enforce_frozen_registry": False,
    }


def test_accepts_only_full_route_gain_and_emits_manifest_for_625(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _synthetic(monkeypatch, tmp_path)
    result = evaluator.evaluate(**arguments)
    assert result["passed"] is True
    assert result["decision"] == "ACCEPT"
    assert result["winning_folds"] == 5
    assert result["macro_delta"] >= 0.003
    assert result["corrected"] > 0 and result["regressed"] == 0
    assert result["sealed_rows_loaded"] == 0
    assert result["candidate_variants_evaluated"] == 1
    assert all(result["gates"].values())
    assert result["route_threshold_contract"]["target_fold_labels_used_for_own_threshold"] == 0
    for fold, values in result["route_threshold_contract"]["candidate"].items():
        assert all(item["excluded_target_fold"] == int(fold) for item in values.values())

    manifest_path = arguments["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    assert manifest["manifest_sha256"] == evaluator.canonical_sha256(payload)
    seed625.verify_route_manifest(manifest_path, seed625.load_spec())


def test_rejection_keeps_original_route_and_emits_no_recipe_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _synthetic(monkeypatch, tmp_path, improve=False)
    result = evaluator.evaluate(**arguments)
    assert result["passed"] is False
    assert result["decision"] == "REJECT_KEEP_ORIGINAL_603"
    assert result["macro_delta"] == 0.0
    assert not arguments["manifest_path"].exists()


def test_upstream_report_must_equal_strict_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declared_path = tmp_path / "declared.json"
    declared_path.write_text(json.dumps({"passed": True}), encoding="utf-8")
    threshold = tmp_path / "threshold.json"
    threshold.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluator, "sha256_file", lambda path: "expected")
    monkeypatch.setattr(
        evaluator.full623,
        "evaluate_full",
        lambda **kwargs: {"passed": True, "decision": "GO_INTEGRATE_624"},
    )
    spec = {"experiment_623_full_threshold_contract_sha256": "expected"}
    folds = {fold: tmp_path for fold in evaluator.FOLDS}
    with pytest.raises(ValueError, match="strict recomputation"):
        evaluator._verify_accepted_623(
            report_path=declared_path,
            registry_path=tmp_path / "registry.csv",
            baseline_dirs=folds,
            runtime_dirs=folds,
            artifact_dirs=folds,
            threshold_contract_path=threshold,
            spec=spec,
        )


def test_missing_fold_and_existing_outputs_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _synthetic(monkeypatch, tmp_path)
    del arguments["artifact_dirs"][4]
    with pytest.raises(ValueError, match="folds 0..4"):
        evaluator.evaluate(**arguments)
    arguments = _synthetic(monkeypatch, tmp_path / "second")
    arguments["output_path"].parent.mkdir(parents=True, exist_ok=True)
    arguments["output_path"].write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluator.evaluate(**arguments)

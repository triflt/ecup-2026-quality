from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "621_semantic_v3_fisher_blockwise_merge" / "evaluate_screen.py"
spec = importlib.util.spec_from_file_location("exp621_screen_test", MODULE_PATH)
assert spec is not None and spec.loader is not None
screen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = screen
spec.loader.exec_module(screen)


def _write_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = []
    item = 0
    for fold in screen.FOLDS:
        for category in screen.CATEGORIES:
            for label in (1, 1, 0, 0):
                rows.append(
                    {
                        "id": str(item),
                        "category": category,
                        "label": label,
                        "semantic_component": f"component-{item}",
                        "component_size": 1,
                        "split": "development",
                        "development_fold": fold,
                    }
                )
                item += 1
    path = tmp_path / "folds.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(screen, "EXPECTED_FOLDS_SHA256", screen.sha256_file(path))
    monkeypatch.setattr(screen, "EXPECTED_BASELINE_PREDICTION_SHA256", None)
    return path


def _self_contract(payload: dict) -> dict:
    result = dict(payload)
    result["contract_sha256"] = screen.canonical_sha256(result)
    return result


def _write_baselines(tmp_path: Path, registry_path: Path) -> dict[int, Path]:
    registry = pd.read_csv(registry_path, dtype={"id": str, "category": str})
    result = {}
    for fold in screen.FOLDS:
        directory = tmp_path / f"baseline-{fold}"
        directory.mkdir()
        expected = registry.loc[registry["development_fold"].eq(fold)].reset_index(drop=True)
        predictions = expected[["id", "category", "label"]].copy()
        predictions["fold"] = fold
        predictions["lora_score"] = [0.9, 0.4, 0.8, 0.1] * len(screen.CATEGORIES)
        predictions_path = directory / "lora_holdout_predictions.csv"
        predictions.to_csv(predictions_path, index=False)
        contract = _self_contract(
            {
                "experiment_id": "600",
                "protocol_version": "semantic_family_v3",
                "component": "original",
                "outer_fold": fold,
                "prediction_rows": len(expected),
                "prediction_ids_sha256": screen.canonical_sha256(expected["id"].tolist()),
                "predictions_sha256": screen.sha256_file(predictions_path),
                "sealed_rows_in_predictions": 0,
                "sealed_rows_used_for_threshold": 0,
                "sealed_rows_used_for_evaluation": 0,
                "decision": "GO",
            }
        )
        (directory / "output_contract.runtime.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
        result[fold] = directory
    return result


def _write_candidates(
    tmp_path: Path, registry_path: Path, *, unchanged_fold: int | None = None
) -> dict[int, Path]:
    registry = pd.read_csv(registry_path, dtype={"id": str, "category": str})
    result = {}
    for fold in screen.SCREEN_FOLDS:
        directory = tmp_path / f"candidate-{fold}"
        merged = directory / "merged_adapter"
        merged.mkdir(parents=True)
        expected = registry.loc[registry["development_fold"].eq(fold)].reset_index(drop=True)
        predictions = expected[["id", "category"]].copy()
        predictions["fold"] = fold
        scores = [0.9, 0.4, 0.8, 0.1] if fold == unchanged_fold else [0.9, 0.4, 0.0, 0.1]
        predictions["lora_score"] = scores * len(screen.CATEGORIES)
        predictions_path = directory / "validation_predictions.csv"
        predictions.to_csv(predictions_path, index=False)
        manifest = {
            "protocol": "621_train_only_fisher_v1",
            "outer_fold": fold,
            "validation_rows": len(expected),
            "validation_ids_sha256": screen.canonical_sha256(expected["id"].tolist()),
            "sealed_rows": 0,
            "validation_labels_loaded": False,
            "sealed_labels_loaded": False,
            "decision": "READY_FOR_EXTERNAL_FOLD_SCORING",
        }
        manifest_path = merged / "merge_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        contract = {
            "protocol": "621_semantic_v3_fold_runtime_v1",
            "outer_fold": fold,
            "validation_rows": len(expected),
            "predictions_sha256": screen.sha256_file(predictions_path),
            "merge_manifest_sha256": screen.sha256_file(manifest_path),
            "validation_labels_loaded": False,
            "sealed_labels_loaded": False,
            "sealed_rows_loaded": 0,
            "decision": "READY_FOR_FROZEN_EXTERNAL_EVALUATION",
        }
        (directory / "fold_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        result[fold] = directory
    return result


def _freeze(
    tmp_path: Path, registry: Path, baselines: dict[int, Path], *, name: str = "thresholds.json"
) -> Path:
    path = tmp_path / name
    screen.freeze_donor_thresholds(
        registry_path=registry, baseline_dirs=baselines, output_path=path
    )
    return path


def test_donor_threshold_freeze_has_no_candidate_input_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, monkeypatch)
    baselines = _write_baselines(tmp_path, registry)
    path = _freeze(tmp_path, registry, baselines)
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert contract["candidate_inputs_read"] == 0
    assert contract["sealed_rows"] == 0
    assert contract["screen_folds"] == [0, 3]
    assert all(
        contract["thresholds"][str(fold)][category]["excluded_target_fold"] == fold
        for fold in screen.SCREEN_FOLDS
        for category in screen.CATEGORIES
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        screen.freeze_donor_thresholds(
            registry_path=registry, baseline_dirs=baselines, output_path=path
        )


def test_screen_passes_all_frozen_gates_for_two_improved_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, monkeypatch)
    baselines = _write_baselines(tmp_path, registry)
    thresholds = _freeze(tmp_path, registry, baselines)
    candidates = _write_candidates(tmp_path, registry)
    result = screen.evaluate_screen(
        registry_path=registry,
        baseline_dirs=baselines,
        candidate_dirs=candidates,
        threshold_contract_path=thresholds,
        output_path=tmp_path / "screen.json",
    )
    assert result["passed"] is True
    assert result["decision"] == "GO_LAUNCH_FOLDS_1_2_4"
    assert result["thresholds_tuned_after_candidate"] is False
    assert result["sealed_rows_loaded"] == 0
    assert result["corrected"] == 4
    assert result["regressed"] == 0
    assert result["corrected_to_regressed_infinite"] is True
    assert all(result["gates"].values())


def test_screen_accepts_and_recomputes_pre_result_legacy_threshold_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, monkeypatch)
    baselines = _write_baselines(tmp_path, registry)
    modern_path = _freeze(tmp_path, registry, baselines, name="modern.json")
    modern = json.loads(modern_path.read_text(encoding="utf-8"))
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    for fold in screen.SCREEN_FOLDS:
        (legacy_dir / f"fold{fold}.json").write_text(
            json.dumps(
                {
                    category: modern["thresholds"][str(fold)][category]["threshold"]
                    for category in screen.CATEGORIES
                }
            ),
            encoding="utf-8",
        )
    legacy = {
        "protocol": "exp621_original_seed42_leave_one_fold_out_thresholds_v1",
        "frozen_before_candidate_results": True,
        "screen_folds": [0, 3],
        "algorithm": (
            "700 unique quantiles from 0.002 through 0.998; maximize donor F1 "
            "with score >= threshold; first maximum wins"
        ),
        "source_prediction_sha256": {
            str(fold): modern["baseline_provenance"][str(fold)]["predictions_sha256"]
            for fold in screen.FOLDS
        },
        "donor_metrics": {
            str(fold): {
                category: {
                    "rows": modern["thresholds"][str(fold)][category]["donor_rows"],
                    "f1": modern["thresholds"][str(fold)][category]["donor_f1"],
                }
                for category in screen.CATEGORIES
            }
            for fold in screen.SCREEN_FOLDS
        },
        "sealed_holdout_used": False,
    }
    legacy_path = legacy_dir / "contract.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    result = screen.evaluate_screen(
        registry_path=registry,
        baseline_dirs=baselines,
        candidate_dirs=_write_candidates(tmp_path, registry),
        threshold_contract_path=legacy_path,
        output_path=tmp_path / "screen.json",
    )
    assert result["passed"] is True
    assert result["thresholds_tuned_after_candidate"] is False


def test_screen_rejects_when_either_fold_does_not_improve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, monkeypatch)
    baselines = _write_baselines(tmp_path, registry)
    thresholds = _freeze(tmp_path, registry, baselines)
    candidates = _write_candidates(tmp_path, registry, unchanged_fold=3)
    result = screen.evaluate_screen(
        registry_path=registry,
        baseline_dirs=baselines,
        candidate_dirs=candidates,
        threshold_contract_path=thresholds,
        output_path=tmp_path / "screen.json",
    )
    assert result["passed"] is False
    assert result["decision"] == "NO_GO_REJECT_621"
    assert result["gates"]["each_fold_delta_gt_0"] is False


def test_screen_rejects_tampered_threshold_contract_before_candidate_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, monkeypatch)
    baselines = _write_baselines(tmp_path, registry)
    thresholds = _freeze(tmp_path, registry, baselines)
    payload = json.loads(thresholds.read_text(encoding="utf-8"))
    payload["thresholds"]["0"][screen.FLAMMABLE]["threshold"] += 0.01
    thresholds.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold_contract_sha256 mismatch"):
        screen.evaluate_screen(
            registry_path=registry,
            baseline_dirs=baselines,
            candidate_dirs={0: tmp_path / "absent-0", 3: tmp_path / "absent-3"},
            threshold_contract_path=thresholds,
            output_path=tmp_path / "screen.json",
        )


def test_screen_rejects_candidate_labels_and_hash_or_order_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path, monkeypatch)
    baselines = _write_baselines(tmp_path, registry)
    thresholds = _freeze(tmp_path, registry, baselines)
    candidates = _write_candidates(tmp_path, registry)
    path = candidates[0] / "validation_predictions.csv"
    frame = pd.read_csv(path, dtype={"id": str})
    frame.insert(2, "label", 0)
    frame.to_csv(path, index=False)
    contract_path = candidates[0] / "fold_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["predictions_sha256"] = screen.sha256_file(path)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="label-free"):
        screen.evaluate_screen(
            registry_path=registry,
            baseline_dirs=baselines,
            candidate_dirs=candidates,
            threshold_contract_path=thresholds,
            output_path=tmp_path / "screen.json",
        )


def test_screen_rejects_non_exact_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _write_registry(tmp_path, monkeypatch)
    baselines = _write_baselines(tmp_path, registry)
    thresholds = _freeze(tmp_path, registry, baselines)
    registry.write_text(registry.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registry_sha256|registry checksum"):
        screen.evaluate_screen(
            registry_path=registry,
            baseline_dirs=baselines,
            candidate_dirs={0: tmp_path / "absent-0", 3: tmp_path / "absent-3"},
            threshold_contract_path=thresholds,
            output_path=tmp_path / "screen.json",
        )

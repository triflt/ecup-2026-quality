from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/623_semantic_v3_multitask_span_head"
SPEC = importlib.util.spec_from_file_location("exp623_screen_test", EXP / "evaluate_screen.py")
assert SPEC is not None and SPEC.loader is not None
screen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = screen
SPEC.loader.exec_module(screen)


def _registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, pd.DataFrame]:
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
    frame = pd.DataFrame(rows)
    path = tmp_path / "registry.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr(screen.parent_screen, "EXPECTED_FOLDS_SHA256", screen.sha256_file(path))
    monkeypatch.setattr(screen.parent_screen, "EXPECTED_BASELINE_PREDICTION_SHA256", None)
    return path, frame


def _self_contract(payload: dict) -> dict:
    result = dict(payload)
    result["contract_sha256"] = screen.canonical_sha256(result)
    return result


def _baselines(tmp_path: Path, registry: pd.DataFrame) -> dict[int, Path]:
    result = {}
    for fold in screen.FOLDS:
        directory = tmp_path / f"baseline-{fold}"
        directory.mkdir()
        expected = registry.loc[registry["development_fold"].eq(fold)].reset_index(drop=True)
        predictions = expected[["id", "category", "label"]].copy()
        predictions["fold"] = fold
        predictions["lora_score"] = [0.9, 0.4, 0.8, 0.1] * len(screen.CATEGORIES)
        prediction_path = directory / "lora_holdout_predictions.csv"
        predictions.to_csv(prediction_path, index=False)
        contract = _self_contract(
            {
                "experiment_id": "600",
                "protocol_version": "semantic_family_v3",
                "component": "original",
                "outer_fold": fold,
                "prediction_rows": len(expected),
                "prediction_ids_sha256": screen.canonical_sha256(expected["id"].tolist()),
                "predictions_sha256": screen.sha256_file(prediction_path),
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


def _runtimes(tmp_path: Path, registry: pd.DataFrame) -> dict[int, Path]:
    result = {}
    ordered = registry.reset_index(drop=True)
    for outer_fold in screen.SCREEN_FOLDS:
        directory = tmp_path / f"runtime-{outer_fold}"
        directory.mkdir()
        train = []
        validation = []
        for row_index, row in ordered.iterrows():
            common = {
                "id": str(row["id"]),
                "category": str(row["category"]),
                "name": "маркер",
                "description": "описание",
                "development_fold": int(row["development_fold"]),
                "row_index": row_index,
            }
            if int(row["development_fold"]) == outer_fold:
                validation.append(common)
            else:
                train.append(
                    {
                        **common,
                        "label": int(row["label"]),
                        "rationale": {"has_evidence": False, "quality_weight": 0.0},
                    }
                )
        for name, values in (("train.jsonl", train), ("validation.jsonl", validation)):
            (directory / name).write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values),
                encoding="utf-8",
            )
        np.savez_compressed(
            directory / "development_selector_oof.npz",
            ids=np.asarray(ordered["id"].astype(str).tolist(), dtype="U32"),
            fold_ids=ordered["development_fold"].to_numpy(np.int8),
            fused_scores=np.linspace(0.05, 0.95, len(ordered), dtype=np.float32),
        )
        (directory / "development_image_manifest.tsv.gz").write_bytes(b"fixture")
        output_names = (
            "train.jsonl",
            "validation.jsonl",
            "development_selector_oof.npz",
            "development_image_manifest.tsv.gz",
        )
        audit = {
            "experiment_id": "623",
            "outer_fold": outer_fold,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "validation_label_columns": [],
            "validation_rationale_columns": [],
            "train_validation_overlap": 0,
            "sealed_rows_written": 0,
            "input_sha256": screen.protocol.FROZEN_INPUT_SHA256,
            "output_sha256": {
                name: screen.sha256_file(directory / name) for name in output_names
            },
            "decision": "GO",
            "frozen_input_hashes_enforced": True,
        }
        (directory / "runtime_audit.json").write_text(json.dumps(audit), encoding="utf-8")
        result[outer_fold] = directory
    return result


def _artifacts(
    tmp_path: Path,
    registry: pd.DataFrame,
    runtimes: dict[int, Path],
    *,
    bad_exact_span: bool = False,
) -> dict[int, Path]:
    result = {}
    for fold in screen.SCREEN_FOLDS:
        directory = tmp_path / f"artifact-{fold}"
        directory.mkdir()
        expected = registry.loc[registry["development_fold"].eq(fold)].reset_index(drop=True)
        predictions = expected[["id", "category"]].copy()
        predictions["fold"] = fold
        predictions["lora_score"] = [0.9, 0.4, 0.0, 0.1] * len(screen.CATEGORIES)
        predictions["verdict_probability"] = 0.5
        predictions["evidence"] = "не маркер" if bad_exact_span and fold == 0 else "маркер"
        predictions["concept"] = "OBJECT_OF_SALE"
        predictions["explanation"] = "Решение основано на объекте продажи: «маркер»."
        predictions["char_start"] = 10
        predictions["char_end"] = 16
        prediction_path = directory / "validation_predictions.csv"
        predictions.to_csv(prediction_path, index=False)
        for name in ("adapter.zip", "auxiliary_head.safetensors", "auxiliary_head_config.json"):
            (directory / name).write_bytes(name.encode())
        runtime = runtimes[fold]
        prediction_audit = {
            "rows": len(expected),
            "prediction_ids_sha256": screen.canonical_sha256(expected["id"].tolist()),
            "predictions_sha256": screen.sha256_file(prediction_path),
            "validation_labels_read": 0,
            "decision": "GO",
        }
        selected_ids = screen._expected_selected_ids(runtime, fold=fold)
        selection = {
            "parent_sha256": screen.EXPECTED_PARENT_RUNNER_SHA256,
            "model_revision": screen.EXPECTED_MODEL_REVISION,
            "prediction_audit": prediction_audit,
            "outer_fold": fold,
            "training_records": len(selected_ids),
            "training_unique_rows": len(set(selected_ids)),
            "selected_id_multiset_sha256": screen.canonical_sha256(
                sorted(Counter(selected_ids).items())
            ),
            "outer_validation_occurrences": 0,
            "runtime_audit_sha256": screen.sha256_file(runtime / "runtime_audit.json"),
            "decision": "GO",
        }
        selection_path = directory / "selection_audit.json"
        selection_path.write_text(json.dumps(selection), encoding="utf-8")
        artifacts = {
            name: screen.sha256_file(directory / name) for name in screen.ARTIFACT_NAMES
        }
        contract = _self_contract(
            {
                "experiment_id": "623",
                "outer_fold": fold,
                "seed": 42,
                "training_records": len(selected_ids),
                "training_unique_rows": len(set(selected_ids)),
                "validation_rows": len(expected),
                "download_failures": 0,
                "optimizer_updates": 1,
                "alignment_masked_safe_candidates": 0,
                "runtime_minutes": 1.0,
                "parent_sha256": screen.EXPECTED_PARENT_RUNNER_SHA256,
                "artifacts": artifacts,
                "validation_labels_written": 0,
                "sealed_rows_used": 0,
                "decision": "GO",
            }
        )
        (directory / "output_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        result[fold] = directory
    return result


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, bad_span: bool = False):
    registry_path, registry = _registry(tmp_path, monkeypatch)
    baselines = _baselines(tmp_path, registry)
    threshold_path = tmp_path / "thresholds.json"
    screen.parent_screen.freeze_donor_thresholds(
        registry_path=registry_path, baseline_dirs=baselines, output_path=threshold_path
    )
    runtimes = _runtimes(tmp_path, registry)
    artifacts = _artifacts(tmp_path, registry, runtimes, bad_exact_span=bad_span)
    return registry_path, baselines, threshold_path, runtimes, artifacts


def test_screen_reuses_frozen_classification_gates_and_reports_structural_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, baselines, thresholds, runtimes, artifacts = _setup(tmp_path, monkeypatch)
    result = screen.evaluate_screen(
        registry_path=registry,
        baseline_dirs=baselines,
        runtime_dirs=runtimes,
        artifact_dirs=artifacts,
        threshold_contract_path=thresholds,
        output_path=tmp_path / "screen.json",
    )
    assert result["passed"] is True
    assert result["decision"] == "GO_LAUNCH_FOLDS_1_2_4"
    assert result["sealed_rows_loaded"] == 0
    assert result["thresholds_tuned_after_candidate"] is False
    assert result["grounding"]["overall"]["grounded_coverage"] == 1.0
    assert result["grounding"]["human_quality_evaluated"] is False
    assert result["grounding"]["human_quality"] is None
    assert all(result["gates"].values())


def test_screen_rejects_invalid_exact_substring_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, baselines, thresholds, runtimes, artifacts = _setup(
        tmp_path, monkeypatch, bad_span=True
    )
    with pytest.raises(ValueError, match="exact substring"):
        screen.evaluate_screen(
            registry_path=registry,
            baseline_dirs=baselines,
            runtime_dirs=runtimes,
            artifact_dirs=artifacts,
            threshold_contract_path=thresholds,
            output_path=tmp_path / "screen.json",
        )


def test_screen_rejects_contract_and_selection_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, baselines, thresholds, runtimes, artifacts = _setup(tmp_path, monkeypatch)
    contract_path = artifacts[0] / "output_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["sealed_rows_used"] = 1
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="contract_sha256 mismatch"):
        screen.evaluate_screen(
            registry_path=registry,
            baseline_dirs=baselines,
            runtime_dirs=runtimes,
            artifact_dirs=artifacts,
            threshold_contract_path=thresholds,
            output_path=tmp_path / "screen.json",
        )


def test_screen_rejects_runtime_claiming_sealed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, baselines, thresholds, runtimes, artifacts = _setup(tmp_path, monkeypatch)
    audit_path = runtimes[0] / "runtime_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["sealed_rows_written"] = 1
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed_rows_written"):
        screen.evaluate_screen(
            registry_path=registry,
            baseline_dirs=baselines,
            runtime_dirs=runtimes,
            artifact_dirs=artifacts,
            threshold_contract_path=thresholds,
            output_path=tmp_path / "screen.json",
        )


def test_selector_enumerates_only_boundary_tie_equivalents() -> None:
    indices = np.arange(8, dtype=np.int64)
    scores = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    options = screen._hard_random_tie_options(
        indices, scores, count=4, rng=np.random.default_rng(42)
    )
    hard_subsets = {tuple(sorted(selected[:2])) for selected, _ in options}
    assert hard_subsets == {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    }

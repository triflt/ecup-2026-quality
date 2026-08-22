from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "621_semantic_v3_fisher_blockwise_merge"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


worker = _load("exp621_worker_test", EXP / "fold_worker.py")
evaluator = _load("exp621_evaluator_test", EXP / "evaluate_fold.py")


def _membership(tmp_path: Path) -> tuple[Path, Path]:
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "category": ["A", "A", "B", "B"],
            "semantic_component": ["c0", "c1", "c2", "c3"],
            "component_size": [1, 1, 1, 1],
            "split": ["development"] * 4,
            "development_fold": [0, 1, 2, 3],
            "label": [1, 0, 1, 0],
        }
    )
    data = tmp_path / "data.csv"
    folds = tmp_path / "folds.csv"
    frame.to_csv(data, index=False)
    frame.drop(columns="label").to_csv(folds, index=False)
    return data, folds


def _adapter(
    tmp_path: Path,
    name: str,
    offset: float,
    target_modules: list[str] | None = None,
) -> Path:
    path = tmp_path / name
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "target_modules": target_modules or ["q_proj", "v_proj"],
                "r": 2,
                "lora_alpha": 2,
                "use_rslora": True,
                "task_type": "CAUSAL_LM",
                "lora_dropout": 0.05,
                "modules_to_save": ["lm_head"],
                "future_peft_default": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "base.model.q_proj.lora_A.weight": np.asarray(
                [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
            ),
            "base.model.q_proj.lora_B.weight": np.asarray(
                [[1.0 + offset, 0.0], [0.0, 1.0 + offset]], dtype=np.float32
            ),
        },
        str(path / "adapter_model.safetensors"),
    )
    return path


def test_membership_loader_rejects_sealed_rows_without_reading_labels(tmp_path: Path) -> None:
    data, folds = _membership(tmp_path)
    frame = pd.read_csv(data)
    frame.loc[0, "split"] = "sealed_holdout"
    frame.to_csv(data, index=False)
    with pytest.raises(ValueError, match="sealed"):
        worker.load_fold_membership(data, folds, outer_fold=0)


def test_train_only_fisher_report_has_no_validation_or_sealed_provenance() -> None:
    report = worker.estimate_fisher_from_gradients(
        [{"module": np.asarray([1.0, -2.0])}, {"module": np.asarray([3.0])}],
        train_rows=3,
        outer_fold=0,
        train_ids_sha256="train-hash",
    )
    assert report["fisher"]["module"] == pytest.approx(5.75)
    assert report["validation_rows"] == 0
    assert report["sealed_rows"] == 0
    assert report["validation_labels_loaded"] is False


def test_fold_merge_writes_hashed_adapter_and_preserves_scope(tmp_path: Path) -> None:
    data, folds = _membership(tmp_path)
    original = _adapter(tmp_path, "original", 0.0)
    specialist = _adapter(tmp_path, "specialist", 1.0, ["v_proj", "q_proj"])
    train_ids, _ = worker.load_fold_membership(data, folds, outer_fold=0)
    fisher = {
        "protocol": "621_train_only_fisher_v1",
        "outer_fold": 0,
        "train_rows": len(train_ids),
        "train_ids_sha256": worker.id_sequence_sha256(train_ids),
        "gradient_batches": 1,
        "validation_rows": 0,
        "sealed_rows": 0,
        "validation_labels_loaded": False,
        "sealed_labels_loaded": False,
        "fisher": {"base.model.q_proj": 1.0},
        "fisher_specialist": {"base.model.q_proj": 3.0},
    }
    fisher_path = tmp_path / "fisher.json"
    fisher_path.write_text(json.dumps(fisher), encoding="utf-8")
    output = tmp_path / "merged"
    manifest = worker.merge_fold_adapters(
        original_dir=original,
        specialist_dir=specialist,
        data_path=data,
        folds_path=folds,
        fisher_report_path=fisher_path,
        output_dir=output,
        outer_fold=0,
        base_model_id="public/model",
        base_model_revision="revision-1",
    )
    assert manifest["sealed_rows"] == 0
    assert manifest["validation_labels_loaded"] is False
    assert len(manifest["merged_weights_sha256"]) == 64
    assert (output / "adapter_config.json").is_file()
    assert (output / "merge_manifest.json").is_file()
    merged_config = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))
    assert merged_config["task_type"] == "CAUSAL_LM"
    assert merged_config["lora_dropout"] == 0.05
    assert merged_config["modules_to_save"] == ["lm_head"]
    assert merged_config["future_peft_default"] == {"enabled": True}
    assert merged_config["base_model_name_or_path"] == "public/model"
    assert merged_config["revision"] == "revision-1"
    assert merged_config["inference_mode"] is True
    assert merged_config["target_modules"] == ["q_proj", "v_proj"]


def test_evaluator_requires_external_predictions_without_labels(tmp_path: Path) -> None:
    data, _ = _membership(tmp_path)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "id": ["a"],
            "category": ["A"],
            "fold": [0],
            "lora_score": [1.0],
        }
    ).to_csv(predictions, index=False)
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"A": 0.5}), encoding="utf-8")
    manifest = tmp_path / "merge_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "outer_fold": 0,
                "sealed_rows": 0,
                "validation_labels_loaded": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation.json"
    result = evaluator.evaluate_fold(
        data_path=data,
        predictions_path=predictions,
        thresholds_path=thresholds,
        merged_manifest_path=manifest,
        output_path=output,
        outer_fold=0,
    )
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["thresholds_tuned_on_validation"] is False
    assert result["sealed_labels_loaded"] is False


def test_evaluator_rejects_prediction_labels(tmp_path: Path) -> None:
    data, _ = _membership(tmp_path)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "id": ["a"],
            "category": ["A"],
            "label": [1],
            "fold": [0],
            "lora_score": [1.0],
        }
    ).to_csv(predictions, index=False)
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"A": 0.5}), encoding="utf-8")
    manifest = tmp_path / "merge_manifest.json"
    manifest.write_text(
        json.dumps({"outer_fold": 0, "sealed_rows": 0, "validation_labels_loaded": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation labels"):
        evaluator.evaluate_fold(
            data_path=data,
            predictions_path=predictions,
            thresholds_path=thresholds,
            merged_manifest_path=manifest,
            output_path=tmp_path / "evaluation.json",
            outer_fold=0,
        )

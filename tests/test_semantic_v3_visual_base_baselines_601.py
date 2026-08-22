from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/601_semantic_v3_visual_base_baselines"
sys.path.insert(0, str(EXPERIMENT))

import assemble_components as assemble
import prepare_runtime_inputs as prepare
import semantic_v3_contract as contract
import train_robust_base as robust


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qwen = _load("exp601_train_qwen", "train_qwen3vl_fold.py")


def test_semantic_contract_is_frozen_and_sealed_fold_is_not_development() -> None:
    assert contract.DEVELOPMENT_FOLDS == (0, 1, 2, 3, 4)
    assert contract.SEALED_FOLD == -1
    assert contract.EXPECTED_DEVELOPMENT_ROWS == 11_118
    assert contract.EXPECTED_SEALED_ROWS == 1_853
    assert contract.QWEN_SEED == 42
    assert contract.QWEN_FIRST_IMAGE_MAX_EDGE == 448
    assert contract.QWEN_FIRST_IMAGE_MAX_PIXELS == 262_144
    assert qwen.sha256(qwen.PARENT_RUNNER) == contract.EXPECTED_QWEN_PARENT_SHA256
    with pytest.raises(ValueError):
        contract.require_development_fold(-1)


def test_empirical_projection_and_fusion_search_are_deterministic() -> None:
    reference = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    values = np.asarray([-3.0, -1.0, 0.5, 3.0], dtype=np.float32)
    projected = robust.project_empirical_rank(reference, values)
    assert projected.tolist() == pytest.approx([0.0, 0.25, 0.5, 1.0])

    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    heads = [np.asarray([0.0, 0.2, 0.8, 1.0], np.float32)] * 4
    first = robust.search_fusion(labels, heads)
    second = robust.search_fusion(labels, heads)
    assert first == second
    assert sum(first["weights"]) == pytest.approx(1.0)


def test_inner_text_vocabulary_excludes_inner_validation_only_token() -> None:
    train_texts = np.asarray(
        [f"common product {'alpha' if index % 2 else 'beta'} item{index}" for index in range(10)]
    )
    valid_texts = np.asarray(["common product heldoutuniquetoken"])

    vectorizer, train_matrix, valid_matrix = robust.vectorize_inner_split(train_texts, valid_texts)

    names = vectorizer.get_feature_names_out().astype(str)
    assert not any("heldoutuniquetoken" in name for name in names)
    assert train_matrix.shape[1] == valid_matrix.shape[1]


def test_runtime_preparation_physically_removes_sealed_rows(tmp_path: Path) -> None:
    ids = ["a", "b", "c", "sealed"]
    data = pd.DataFrame(
        {
            "id": ids,
            "name": ["n"] * 4,
            "description": ["d"] * 4,
            "category": ["БАД"] * 4,
            "label": [0, 1, 0, 1],
        }
    )
    folds = pd.DataFrame(
        {
            "id": ids,
            "category": ["БАД"] * 4,
            "label": [0, 1, 0, 1],
            "semantic_component": ["x", "y", "z", "s"],
            "component_size": [1] * 4,
            "split": ["development"] * 3 + ["sealed_holdout"],
            "development_fold": [0, 1, 2, -1],
        }
    )
    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"
    manifest_path = tmp_path / "manifest.tsv.gz"
    robust_path = tmp_path / "robust.npz"
    data.to_csv(data_path, index=False)
    folds.to_csv(folds_path, index=False)
    with gzip.open(manifest_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image_url"], delimiter="\t")
        writer.writeheader()
        for item_id in ids:
            writer.writerow({"id": item_id, "image_url": f"memory://{item_id}"})
    np.savez_compressed(
        robust_path,
        ids=np.asarray(ids[:3]),
        labels=np.asarray([0, 1, 0], np.int8),
        categories=np.asarray(["БАД"] * 3),
        folds=np.asarray([0, 1, 2], np.int8),
        protocol_version=np.asarray(contract.ROBUST_PROTOCOL_VERSION),
        robust_base_score=np.asarray([0.1, 0.9, 0.2], np.float32),
    )

    output = tmp_path / "runtime"
    report = prepare.prepare(
        data_path=data_path,
        folds_path=folds_path,
        manifest_path=manifest_path,
        robust_base_path=robust_path,
        output_dir=output,
    )

    assert report["sealed_ids_in_any_runtime_output"] == 0
    assert report["development_rows"] == 3
    assert "sealed" not in (output / "development_data.csv").read_text()
    with gzip.open(output / "development_first_image_manifest.tsv.gz", "rt") as stream:
        assert "sealed" not in stream.read()
    oof = np.load(output / "qwen_parent_oof.npz", allow_pickle=False)
    assert oof["ids"].astype(str).tolist() == ids[:3]

    legacy_path = tmp_path / "legacy_robust.npz"
    np.savez_compressed(
        legacy_path,
        ids=np.asarray(ids[:3]),
        labels=np.asarray([0, 1, 0], np.int8),
        categories=np.asarray(["БАД"] * 3),
        folds=np.asarray([0, 1, 2], np.int8),
        robust_base_score=np.asarray([0.1, 0.9, 0.2], np.float32),
    )
    with pytest.raises(ValueError, match="strict nested v2"):
        prepare.prepare(
            data_path=data_path,
            folds_path=folds_path,
            manifest_path=manifest_path,
            robust_base_path=legacy_path,
            output_dir=tmp_path / "legacy_runtime",
        )


def test_qwen_launcher_locks_exact_parent_recipe() -> None:
    environment: dict[str, str] = {}
    configured = qwen.configure_environment(
        fold=4,
        runtime_dir=Path("/tmp/runtime"),
        output_dir=Path("/tmp/output"),
        environment=environment,
    )
    assert configured["SEED"] == "42"
    assert configured["TRAINING_MODE"] == "hard"
    assert configured["MODEL_CLASS"] == "image_text"
    assert configured["QWEN3VL_FIRST_IMAGE_MAX_EDGE"] == "448"
    assert configured["QWEN3VL_FIRST_IMAGE_MAX_PIXELS"] == "262144"
    assert configured["HOLDOUT_FOLD"] == "4"
    with pytest.raises(ValueError, match="non-parent switches"):
        qwen.configure_environment(
            fold=0,
            runtime_dir=Path("/tmp/runtime"),
            output_dir=Path("/tmp/output"),
            environment={"FULL_TRAIN": "1"},
        )
    with pytest.raises(ValueError):
        qwen.configure_environment(
            fold=-1,
            runtime_dir=Path("/tmp/runtime"),
            output_dir=Path("/tmp/output"),
            environment={},
        )


def test_component_assembly_rejects_sealed_and_emits_route_order(tmp_path: Path) -> None:
    dev_ids = [f"d{i}" for i in range(5)]
    registry = pd.DataFrame(
        {
            "id": dev_ids + ["sealed"],
            "category": ["БАД"] * 6,
            "label": [0, 1, 0, 1, 1, 0],
            "semantic_component": [f"c{i}" for i in range(6)],
            "component_size": [1] * 6,
            "split": ["development"] * 5 + ["sealed_holdout"],
            "development_fold": [0, 1, 2, 3, 4, -1],
        }
    )
    folds_path = tmp_path / "folds.csv"
    registry.to_csv(folds_path, index=False)
    robust_path = tmp_path / "robust.npz"
    np.savez_compressed(
        robust_path,
        ids=np.asarray(dev_ids),
        labels=np.asarray([0, 1, 0, 1, 1], np.int8),
        categories=np.asarray(["БАД"] * 5),
        folds=np.arange(5, dtype=np.int8),
        protocol_version=np.asarray(contract.ROBUST_PROTOCOL_VERSION),
        robust_base_score=np.linspace(0.0, 1.0, 5, dtype=np.float32),
    )
    predictions, reports = [], []
    for fold, item_id in enumerate(dev_ids):
        prediction = tmp_path / f"p{fold}.csv"
        report = tmp_path / f"r{fold}.json"
        pd.DataFrame(
            {
                "id": [item_id],
                "category": ["БАД"],
                "label": [registry.loc[fold, "label"]],
                "fold": [fold],
                "lora_score": [float(fold)],
            }
        ).to_csv(prediction, index=False)
        report.write_text(
            json.dumps(
                {
                    "holdout_fold": fold,
                    "validation_version": "semantic_family_v3",
                    "development_only": True,
                    "sealed_rows_seen_by_train_selector_threshold_eval": 0,
                    "first_image_max_edge": 448,
                    "first_image_max_pixels": 262144,
                    "inference_image_index": 0,
                    "inference_images_per_row": 1,
                    "inference_passes": 1,
                    "parent_recipe": "experiment_110_exact",
                }
            )
        )
        predictions.append(prediction)
        reports.append(report)

    output = tmp_path / "assembled"
    result = assemble.assemble(
        folds_path=folds_path,
        robust_base_path=robust_path,
        prediction_paths=predictions,
        report_paths=reports,
        output_dir=output,
    )

    assert result["sealed_rows_in_outputs"] == 0
    assert result["route400_component_order"] == [
        "robust_base_rank",
        "qwen3vl_rank",
        "qwen35_rank",
    ]
    bundle = np.load(output / "semantic_v3_visual_base_components.npz", allow_pickle=False)
    assert "sealed" not in bundle["ids"].astype(str)


def test_card_records_completed_robust_base_and_running_qwen_jobs() -> None:
    config = tomllib.loads((EXPERIMENT / "experiment.toml").read_text())
    metrics = json.loads((EXPERIMENT / "results/metrics.json").read_text())
    assert config["validation"]["development_folds"] == [0, 1, 2, 3, 4]
    assert config["validation"]["sealed_holdout_excluded"] is True
    assert config["robust_base"]["status"] == "strict_nested_v2_completed_numpy_2_3_1"
    assert config["robust_base"]["macro_f1"] == pytest.approx(0.8935380094)
    assert config["qwen3vl"]["seed"] == 42
    assert config["qwen3vl"]["first_image_max_edge"] == 448
    assert config["qwen3vl"]["development_jobs"] == 5
    assert config["qwen3vl"]["gpu_per_job"] == 1
    assert config["execution"]["launch_authorized"] is True
    assert config["execution"]["commit_authorized"] is True
    assert metrics["training_launched"] is True
    assert metrics["sealed_holdout_used"] is False

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

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


builder = _load("exp621_builder_test", EXP / "build_fold_runtime.py")
worker = _load("exp621_worker_runtime_test", EXP / "run_fold.py")


def _runtime_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    data = tmp_path / "data.csv"
    folds = tmp_path / "folds.csv"
    image_manifest = tmp_path / "images.tsv"
    rows = []
    membership = []
    for index in range(6):
        item_id = f"id-{index}"
        category = "БАД" if index % 2 == 0 else "Легковоспламеняющиеся"
        label = index % 2
        split = "development" if index < 5 else "sealed_holdout"
        fold = index if index < 5 else -1
        rows.append(
            {
                "id": item_id,
                "name": f"name-{index}",
                "description": f"description-{index}",
                "category": category,
                "label": str(label),
            }
        )
        membership.append(
            {
                "id": item_id,
                "category": category,
                "semantic_component": f"component-{index}",
                "component_size": "1",
                "split": split,
                "development_fold": str(fold),
                "label": str(label),
            }
        )
    pd.DataFrame(rows).to_csv(data, index=False)
    pd.DataFrame(membership).to_csv(folds, index=False)
    with image_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image_url"], delimiter="\t")
        writer.writeheader()
        for row in rows[:5]:
            writer.writerow({"id": row["id"], "image_url": f"{row['id']}/0.jpg"})
    return data, folds, image_manifest


def test_runtime_builder_physically_removes_validation_labels(tmp_path: Path) -> None:
    data, folds, image_manifest = _runtime_inputs(tmp_path)
    runtime = tmp_path / "runtime"
    audit = builder.build_fold_runtime(
        data_path=data,
        folds_path=folds,
        image_manifest_path=image_manifest,
        output_dir=runtime,
        outer_fold=0,
    )
    assert audit["validation_label_column_present"] is False
    assert audit["validation_labels_written"] is False
    assert audit["sealed_rows_written"] == 0
    validation_header = (runtime / "validation_data.csv").read_text(encoding="utf-8").splitlines()[0]
    development_header = (runtime / "development_data_label_free.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    train_header = (runtime / "train_data.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "label" not in validation_header
    assert "label" not in development_header
    assert "label" in train_header
    assert "sealed" not in (runtime / "validation_data.csv").read_text(encoding="utf-8")
    assert json.loads((runtime / "runtime_audit.json").read_text())["files_sha256"]


def test_adapter_archive_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "adapter.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("adapter/adapter_config.json", "{}")
        output.writestr("adapter/adapter_model.safetensors", "not-a-model")
        output.writestr("../escape.txt", "must not extract")
    with pytest.raises(ValueError, match="traversal"):
        worker._safe_extract_zip(archive, tmp_path / "unpack")
    assert not (tmp_path / "escape.txt").exists()


def test_fisher_selection_is_deterministic_balanced_and_component_scoped() -> None:
    rows = [
        {
            "id": f"row-{index}",
            "category": "БАД" if index % 2 else "Легковоспламеняющиеся",
            "semantic_component": f"component-{index // 2}",
            "label": str(index % 2),
            "development_fold": "1",
        }
        for index in range(8)
    ]
    first = worker.select_fisher_rows(rows, task="original", max_rows=8)
    second = worker.select_fisher_rows(rows, task="original", max_rows=8)
    assert first == second
    assert len({(row["semantic_component"], row["label"]) for row in first}) == len(first)
    assert sum(row["label"] == "0" for row in first) == sum(row["label"] == "1" for row in first)


def test_label_free_prediction_contract_has_no_label_column(tmp_path: Path) -> None:
    path = tmp_path / "validation_predictions.csv"
    worker._write_predictions(
        path,
        [{"id": "a", "category": "БАД", "fold": "0", "lora_score": "0.2"}],
    )
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "id,category,fold,lora_score"
    assert "label" not in header


def test_worker_rejects_validation_label_input(tmp_path: Path) -> None:
    path = tmp_path / "validation.csv"
    pd.DataFrame(
        {
            "id": ["a"],
            "name": ["n"],
            "description": ["d"],
            "category": ["БАД"],
            "semantic_component": ["c"],
            "development_fold": ["0"],
            "label": ["1"],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="label column"):
        worker._load_rows(path, require_label=False)

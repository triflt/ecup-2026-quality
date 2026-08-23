from __future__ import annotations

import csv
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

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
    validation_header = (
        (runtime / "validation_data.csv").read_text(encoding="utf-8").splitlines()[0]
    )
    development_header = (
        (runtime / "development_data_label_free.csv").read_text(encoding="utf-8").splitlines()[0]
    )
    train_header = (runtime / "train_data.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "label" not in validation_header
    assert "label" not in development_header
    assert "label" in train_header
    assert "sealed" not in (runtime / "validation_data.csv").read_text(encoding="utf-8")
    assert json.loads((runtime / "runtime_audit.json").read_text())["files_sha256"]
    worker.verify_runtime_audit(runtime, json.loads((runtime / "runtime_audit.json").read_text()))


def test_runtime_audit_rejects_post_build_input_tampering(tmp_path: Path) -> None:
    data, folds, image_manifest = _runtime_inputs(tmp_path)
    runtime = tmp_path / "runtime"
    builder.build_fold_runtime(
        data_path=data,
        folds_path=folds,
        image_manifest_path=image_manifest,
        output_dir=runtime,
        outer_fold=0,
    )
    audit = json.loads((runtime / "runtime_audit.json").read_text())
    with (runtime / "validation_data.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="runtime input checksum mismatch"):
        worker.verify_runtime_audit(runtime, audit)


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
    assert len({row["semantic_component"] for row in first}) == len(first)


def test_fisher_selection_never_uses_two_labels_from_one_component() -> None:
    rows = [
        {
            "id": "duplicate-negative",
            "category": "БАД",
            "semantic_component": "duplicate",
            "label": "0",
            "development_fold": "1",
        },
        {
            "id": "duplicate-positive",
            "category": "БАД",
            "semantic_component": "duplicate",
            "label": "1",
            "development_fold": "2",
        },
        {
            "id": "other-negative",
            "category": "БАД",
            "semantic_component": "other-negative",
            "label": "0",
            "development_fold": "2",
        },
        {
            "id": "other-positive",
            "category": "БАД",
            "semantic_component": "other-positive",
            "label": "1",
            "development_fold": "3",
        },
    ]
    selected = worker.select_fisher_rows(rows, task="original", max_rows=4)
    assert len({row["semantic_component"] for row in selected}) == len(selected)


def test_predownload_accepts_an_existing_empty_directory(tmp_path: Path, monkeypatch) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()

    def fake_download(item_id: str, url: str, root: Path):
        Image.new("RGB", (2, 2), "white").save(root / f"{item_id}.jpg")
        return item_id, None

    monkeypatch.setattr(worker, "_download_image", fake_download)
    report = worker.predownload_images(["a"], {"a": "https://example.invalid/a"}, image_root)
    assert report["downloaded"] == 1
    assert (image_root / "a.jpg").is_file()


def test_collect_fisher_uses_explicit_outer_fold_and_combines_a_b_gradients(monkeypatch) -> None:
    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Parameter:
        def __init__(self, values):
            self.grad = FakeTensor(values)

    class Loss:
        def __truediv__(self, value):
            return self

        def backward(self):
            return None

    class Model:
        def train(self):
            return None

        def zero_grad(self, set_to_none=True):
            return None

        def named_parameters(self):
            return [
                ("base.q_proj.lora_A.default.weight", Parameter(np.ones((2, 3)))),
                ("base.q_proj.lora_B.default.weight", Parameter(np.ones((4, 2)) * 2)),
            ]

    monkeypatch.setattr(worker, "_target_ids", lambda processor: (0, 1))
    monkeypatch.setattr(worker, "_resolve_image", lambda *args: object())
    monkeypatch.setattr(worker, "_encode", lambda *args: {})
    monkeypatch.setattr(worker, "_binary_loss", lambda *args: (Loss(), 0.0))
    rows = [
        {
            "id": "train-from-fold-1",
            "label": "1",
            "development_fold": "1",
            "name": "n",
            "description": "d",
            "category": "БАД",
        }
    ]
    report = worker.collect_fisher(
        model=Model(),
        processor=object(),
        rows=rows,
        image_manifest={"train-from-fold-1": "unused"},
        image_root=Path("unused"),
        device="cpu",
        task="original",
        outer_fold=3,
        batch_size=1,
        torch=object(),
    )
    assert report["outer_fold"] == 3
    # Six A values of 1 and eight B values of 2 are concatenated, not added.
    assert report["fisher"]["base.q_proj"] == pytest.approx((6 + 8 * 4) / 14)


def test_parent_prompt_and_encoding_recipe_are_frozen() -> None:
    row = {
        "id": "a",
        "name": "<b>Имя</b>   товара",
        "description": "описание",
        "category": "БАД",
    }
    expected = (
        "Категория: БАД\n"
        "Название: Имя товара\n"
        "Описание: описание\n"
        "Правило: Метка 1 только если в описании или на упаковке есть прямое указание БАД "
        "или dietary supplement. Спортивное питание без такой маркировки, явное "
        "отрицание или отсутствие маркировки — метка 0.\n"
        "Определи правильность категории. Ответь только одной цифрой: 1 или 0."
    )
    assert worker.parent_user_text(row) == expected
    messages = worker.build_messages(row, object())

    class Value:
        def to(self, device):
            return ("moved", device)

    class Processor:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, value, **kwargs):
            assert value == messages
            self.kwargs = kwargs
            return {"input_ids": Value()}

    processor = Processor()
    encoded = worker._encode(processor, messages, object(), "cuda")
    assert encoded == {"input_ids": ("moved", "cuda")}
    assert processor.kwargs == {
        "enable_thinking": False,
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
        "padding": True,
        "truncation": True,
        "max_length": 1536,
    }


def test_parent_source_verification_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "parent.py"
    source.write_text("print('frozen')\n", encoding="utf-8")
    digest = builder.sha256_file(source)
    assert worker.verify_parent_source(source, digest, component="original") == digest
    with pytest.raises(ValueError, match="checksum mismatch"):
        worker.verify_parent_source(source, "0" * 64, component="original")


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

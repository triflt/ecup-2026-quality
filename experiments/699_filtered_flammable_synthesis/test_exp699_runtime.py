from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("exp699_builder", "build_synth_runtime.py")
append_builder = load("exp699_append_builder", "build_append_runtime.py")
refit_builder = load("exp699_refit_builder", "build_refit_runtime.py")
trainer = load("exp699_trainer", "train_synth_fold.py")
evaluator = load("exp699_evaluator", "evaluate_gpu_screen.py")
remote_compute_cache = load("exp699_remote_compute_cache", "prepare_remote_compute_image_cache.py")
assembler = load("exp699_assembler", "assemble_remote_compute_candidate.py")
submission_builder = load("exp699_submission_builder", "build_refit_submission.py")
smoke_builder = load("exp699_smoke_builder", "prepare_submission_smoke.py")
package_runner = load(
    "exp699_package_runner", "run_remote_compute_package_candidate.py"
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_text_only_conversation_has_no_image() -> None:
    row = SimpleNamespace(
        category="Легковоспламеняющиеся", name="Газ", description="Баллон с газом"
    )
    value = trainer.conversation(row, None)
    assert [item["type"] for item in value[0]["content"]] == ["text"]


def test_qwen3vl_conversation_adds_only_answer_suffix() -> None:
    row = SimpleNamespace(
        category="Легковоспламеняющиеся", name="Газ", description="Баллон с газом"
    )
    value = trainer.conversation(row, None, "1")
    assert [message["role"] for message in value] == ["user", "assistant"]
    assert value[-1]["content"] == [{"type": "text", "text": "1"}]


def test_frozen_architecture_recipe_contracts() -> None:
    assert trainer.MODEL_CONTRACTS["qwen35_4b"] == {
        "model_id": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "loader": "multimodal",
        "micro_batch": 2,
        "accumulation": 8,
    }


def test_assistant_suffix_labels_masks_prompt_tokens() -> None:
    import torch

    full = {
        "input_ids": torch.tensor([[0, 10, 11, 20], [12, 13, 21, 22]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]),
    }
    prompt = {
        "input_ids": torch.tensor([[0, 10, 11], [0, 12, 13]]),
        "attention_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
    }
    labels = trainer.assistant_suffix_labels(full, prompt)
    assert labels.tolist() == [[-100, -100, -100, 20], [-100, -100, 21, 22]]


def test_remote_candidate_artifact_exact_binding(tmp_path: Path) -> None:
    import numpy as np

    ids = np.asarray(["a", "b", "c", "d"])
    folds = np.asarray([0, 0, 3, 3])
    categories = np.asarray(["БАД", "Легковоспламеняющиеся"] * 2)
    root = tmp_path / "artifact"
    source_indices = {"a": 101, "b": 77, "c": 203, "d": 150}
    for fold, positions in ((0, (0, 1)), (3, (2, 3))):
        fold_root = root / f"fold{fold}"
        fold_root.mkdir(parents=True)
        predictions = fold_root / "predictions.jsonl"
        rows = [
            {
                # Runtime global_index belongs to its frozen source ordering and
                # intentionally is not the position in the evaluator OOF arrays.
                "global_index": source_indices[ids[index]],
                "id": ids[index],
                "fold": fold,
                "category": categories[index],
                "score": float(index) - 1.5,
                "prediction": int(index >= 2),
            }
            for index in positions
        ]
        write_jsonl(predictions, rows)
        adapter = fold_root / "adapter.zip"
        adapter.write_bytes(f"adapter-{fold}".encode())
        contract = {
            "experiment_id": "699",
            "architecture": "qwen35_4b",
            "fold": fold,
            "source": "v1",
            "mode": "positive_only",
            "cap": 40,
            "validation_labels_read": 0,
            "sealed_rows_used": 0,
            "public_rows_used": 0,
            "decision": "GO_EVALUATE",
            "optimizer_steps": 306,
            "loss_contract": "binary_bce_last_token",
            "runtime_minutes": 1.0,
            "peak_gpu_bytes": 10,
            "artifacts": {
                "predictions.jsonl": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                "adapter.zip": hashlib.sha256(adapter.read_bytes()).hexdigest(),
            },
        }
        contract["contract_sha256"] = evaluator.canonical_sha256(contract)
        (fold_root / "output_contract.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
    scores, audit = evaluator.load_candidate(
        {
            "architecture": "qwen35_4b",
            "source": "v1",
            "mode": "positive_only",
            "cap": 40,
            "path": root,
        },
        ids,
        folds,
        categories,
        np.asarray([source_indices[item] for item in ids]),
    )
    assert scores.tolist() == [-1.5, -0.5, 0.5, 1.5]
    assert audit["rows"] == 4
    assert trainer.MODEL_CONTRACTS["qwen3vl_2b"] == {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": "e2378df056d88153dc44616229fa371fcb87e236",
        "loader": "image_text",
        "micro_batch": 4,
        "accumulation": 4,
    }


def test_full_fivefold_candidate_artifact_binding(tmp_path: Path) -> None:
    import numpy as np

    ids = np.asarray([f"id-{fold}" for fold in range(5)])
    folds = np.arange(5, dtype=np.int8)
    categories = np.asarray(["Легковоспламеняющиеся"] * 5)
    global_indices = np.arange(100, 105, dtype=np.int64)
    root = tmp_path / "artifact"
    for fold in range(5):
        fold_root = root / f"fold{fold}"
        fold_root.mkdir(parents=True)
        predictions = fold_root / "predictions.jsonl"
        write_jsonl(
            predictions,
            [
                {
                    "global_index": int(global_indices[fold]),
                    "id": str(ids[fold]),
                    "fold": fold,
                    "category": str(categories[fold]),
                    "score": float(fold),
                    "prediction": 1,
                }
            ],
        )
        adapter = fold_root / "adapter.zip"
        adapter.write_bytes(f"adapter-{fold}".encode())
        contract = {
            "experiment_id": "699",
            "architecture": "qwen3vl_2b",
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "model_revision": "e2378df056d88153dc44616229fa371fcb87e236",
            "fold": fold,
            "source": "both",
            "mode": "balanced",
            "cap": 80,
            "image_policy": "first_image_448_plus_text_only_synthetic",
            "seed": 42,
            "epochs": 1,
            "learning_rate": 0.0002,
            "micro_batch": 4,
            "gradient_accumulation": 4,
            "effective_batch": 16,
            "validation_labels_read": 0,
            "sealed_rows_used": 0,
            "public_rows_used": 0,
            "decision": "GO_EVALUATE",
            "optimizer_steps": 306,
            "loss_contract": "assistant_suffix_lm",
            "threshold": 0.0,
            "threshold_tuned": False,
            "packages": {"torch": "test", "transformers": "test", "peft": "test"},
            "validation_rows": 1,
            "runtime_minutes": 1.0,
            "peak_gpu_bytes": 10,
            "artifacts": {
                "predictions.jsonl": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                "adapter.zip": hashlib.sha256(adapter.read_bytes()).hexdigest(),
            },
        }
        contract["contract_sha256"] = evaluator.canonical_sha256(contract)
        (fold_root / "output_contract.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
    scores, audit = evaluator.load_candidate(
        {
            "architecture": "qwen3vl_2b",
            "source": "both",
            "mode": "balanced",
            "cap": 80,
            "path": root,
        },
        ids,
        folds,
        categories,
        global_indices,
        evaluation_folds=evaluator.FULL_FOLDS,
    )
    assert scores.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert audit["rows"] == 5
    assert set(audit["folds"]) == {"0", "1", "2", "3", "4"}
    assembled = tmp_path / "assembled"
    manifest = assembler.assemble(
        {fold: root / f"fold{fold}" for fold in range(5)}, assembled
    )
    assert manifest["decision"] == "GO_FULL5_EVALUATE"
    assert manifest["self_sha256"] == assembler.canonical_sha256(
        {key: value for key, value in manifest.items() if key != "self_sha256"}
    )
    for fold in range(5):
        assert (assembled / f"fold{fold}" / "adapter.zip").stat().st_ino == (
            root / f"fold{fold}" / "adapter.zip"
        ).stat().st_ino


def test_source_split_rebinds_to_frozen_runtime(tmp_path: Path) -> None:
    import numpy as np

    oof_ids = np.asarray(["c", "a", "b", "d", "sealed"])
    oof_categories = np.asarray(
        ["БАД", "Легковоспламеняющиеся", "БАД", "БАД", "БАД"]
    )
    rows = [
        {"global_index": 0, "id": "a", "fold": 1, "category": "Легковоспламеняющиеся"},
        {"global_index": 1, "id": "b", "fold": 0, "category": "БАД"},
        {"global_index": 2, "id": "c", "fold": 3, "category": "БАД"},
        {"global_index": 3, "id": "d", "fold": 2, "category": "БАД"},
    ]
    runtime_map = tmp_path / "runtime_map.jsonl"
    write_jsonl(runtime_map, rows)
    contract = {
        "schema": "exp699_eval_runtime_map_v1",
        "experiment_id": "699",
        "rows": 4,
        "folds": {str(fold): {} for fold in range(5)},
        "map_sha256": evaluator.sha256(runtime_map),
        "labels_read": 0,
        "sealed_rows": 0,
        "public_used": False,
    }
    contract["self_sha256"] = evaluator.canonical_sha256(contract)
    contract_path = tmp_path / "runtime_map_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    positions, folds, global_indices = evaluator.load_source_split(
        runtime_map, contract_path, oof_ids, oof_categories
    )
    assert positions.tolist() == [1, 2, 0, 3]
    assert folds.tolist() == [1, 0, 3, 2]
    assert global_indices.tolist() == [0, 1, 2, 3]


def test_semantic_family_binding_and_sizes(tmp_path: Path) -> None:
    import numpy as np

    ids = np.asarray([f"id-{fold}" for fold in range(5)] + ["paired"])
    folds = np.asarray([0, 1, 2, 3, 4, 0], dtype=np.int8)
    categories = np.asarray(["БАД"] * 6)
    global_indices = np.arange(6, dtype=np.int64)
    for fold in range(5):
        root = tmp_path / f"fold{fold}"
        root.mkdir()
        rows = [
            {
                "global_index": fold,
                "id": f"id-{fold}",
                "fold": fold,
                "category": "БАД",
                "semantic_component": "shared" if fold == 1 else f"single-{fold}",
            }
        ]
        if fold == 0:
            rows.append(
                {
                    "global_index": 5,
                    "id": "paired",
                    "fold": 0,
                    "category": "БАД",
                    "semantic_component": "shared",
                }
            )
        write_jsonl(root / "validation.jsonl", rows)
    sizes, audit = evaluator.load_semantic_families(
        tmp_path, ids, folds, categories, global_indices
    )
    assert sizes.tolist() == [1, 2, 1, 1, 1, 2]
    assert audit == {
        "rows": 6,
        "singleton_rows": 4,
        "rare_le2_rows": 6,
        "repeated_rows": 2,
    }


def test_remote_compute_cache_uses_mounted_first_image(tmp_path: Path) -> None:
    from PIL import Image

    runtime = tmp_path / "runtime"
    for fold in (0, 3):
        root = runtime / f"fold{fold}"
        root.mkdir(parents=True)
        write_jsonl(
            root / "train.jsonl",
            [{"id": f"id-{fold}", "synthetic": False}],
        )
        write_jsonl(root / "validation.jsonl", [])
    images = tmp_path / "images"
    for fold in (0, 3):
        root = images / f"id-{fold}"
        root.mkdir(parents=True)
        Image.new("RGB", (800, 600), (fold, 1, 2)).save(root / "0.jpg")
    cache = tmp_path / "cache"
    report = remote_compute_cache.prepare(runtime, images, cache, "qwen3vl_2b")
    assert report["unique_ids"] == 2
    assert report["created"] == 2
    for path in cache.iterdir():
        with Image.open(path) as image:
            assert max(image.size) == 448


def test_training_cache_preflight_is_complete_and_network_free(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = [
        {"id": "real-a", "synthetic": False},
        {"id": "real-a", "synthetic": False},
        {"id": "synthetic-a", "synthetic": True},
    ]
    path = trainer.cache_path(cache, "real-a", "qwen35_4b")
    path.write_bytes(b"opaque-image")
    assert trainer.validate_image_cache(cache, rows, "qwen35_4b") == {
        "unique_real_ids": 1,
        "missing": 0,
    }
    path.unlink()
    try:
        trainer.validate_image_cache(cache, rows, "qwen35_4b")
    except ValueError as error:
        assert "image cache incomplete: 1 of 1" in str(error)
    else:
        raise AssertionError("missing image cache entry was accepted")


def test_two_epoch_training_shape_doubles_only_horizon() -> None:
    one = trainer.training_shape(4902, 2, 8, 1)
    two = trainer.training_shape(4902, 2, 8, 2)
    assert one == {
        "batches_per_epoch": 2451,
        "updates_per_epoch": 307,
        "optimizer_steps": 307,
        "training_occurrences_seen": 4902,
    }
    assert two == {
        "batches_per_epoch": 2451,
        "updates_per_epoch": 307,
        "optimizer_steps": 614,
        "training_occurrences_seen": 9804,
    }


def test_full_refit_runtime_reconstructs_all_development_and_appends_ten(
    tmp_path: Path,
) -> None:
    import numpy as np

    ids = np.asarray([f"id-{index:05d}" for index in range(11118)])
    labels = np.asarray(
        [1] * 1600 + [0] * 6000 + [1] * 170 + [0] * 3348,
        dtype=np.int8,
    )
    categories = np.asarray(
        ["БАД"] * 7600 + ["Легковоспламеняющиеся"] * 3518
    )
    fused = np.linspace(0.0, 1.0, len(ids), dtype=np.float32)
    parent_root = tmp_path / "parent"
    map_rows: list[dict] = []
    by_fold: dict[int, list[dict]] = {fold: [] for fold in range(5)}
    for index, row_id in enumerate(ids):
        fold = index % 5
        row = {
            "category": str(categories[index]),
            "description": f"description {row_id}",
            "fold": fold,
            "global_index": index,
            "id": str(row_id),
            "image_url": "https://example.invalid/image.jpg",
            "name": f"name {row_id}",
            "ocr_images": [],
            "semantic_component": f"component-{row_id}",
        }
        map_rows.append(
            {
                "global_index": index,
                "id": str(row_id),
                "fold": fold,
                "category": str(categories[index]),
            }
        )
        by_fold[fold].append(row)
    for fold, rows in by_fold.items():
        root = parent_root / f"fold{fold}"
        root.mkdir(parents=True)
        validation_path = root / "validation.jsonl"
        write_jsonl(validation_path, rows)
        audit = {
            "outer_fold": fold,
            "decision": "GO",
            "validation_labels_written": 0,
            "sealed_rows_written": 0,
            "output_sha256": {
                "validation.jsonl": refit_builder.sha256_file(validation_path)
            },
        }
        audit["contract_sha256"] = refit_builder.canonical_sha256(audit)
        (root / "runtime_audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
    runtime_map = tmp_path / "runtime_map.jsonl"
    write_jsonl(runtime_map, map_rows)
    runtime_map_contract = {
        "map_sha256": refit_builder.sha256_file(runtime_map),
        "labels_read": 0,
        "sealed_rows": 0,
        "public_used": False,
    }
    runtime_map_contract["self_sha256"] = refit_builder.canonical_sha256(
        runtime_map_contract
    )
    runtime_map_contract_path = tmp_path / "runtime_map_contract.json"
    runtime_map_contract_path.write_text(
        json.dumps(runtime_map_contract), encoding="utf-8"
    )
    oof = tmp_path / "oof.npz"
    np.savez(oof, ids=ids, categories=categories, labels=labels, fused=fused)
    ranked: list[tuple[int, Path]] = []
    for fold in range(5):
        path = tmp_path / f"ranked-fold{fold}.jsonl"
        write_jsonl(
            path,
            [
                {
                    "candidate_id": f"candidate-{index:02d}",
                    "category": "Легковоспламеняющиеся",
                    "description": f"synthetic description {index}",
                    "label": 1,
                    "label_rank": index + fold % 2,
                    "name": f"synthetic {index}",
                    "selector_fold": fold,
                    "source": "v2",
                }
                for index in range(15)
            ]
            + [
                {
                    "candidate_id": f"fold-only-{fold}",
                    "category": "Легковоспламеняющиеся",
                    "description": "fold-local synthetic candidate",
                    "label": 1,
                    "label_rank": 99,
                    "name": f"fold-only {fold}",
                    "selector_fold": fold,
                    "source": "v2",
                }
            ],
        )
        ranked.append((fold, path))
    output = tmp_path / "refit"
    report = refit_builder.build(
        SimpleNamespace(
            parent_root=parent_root,
            runtime_map=runtime_map,
            runtime_map_contract=runtime_map_contract_path,
            oof=oof,
            ranked=ranked,
            epochs=1,
            output_dir=output,
        )
    )
    assert report["decision"] == "GO_FULL_REFIT"
    assert report["original_real_occurrences"] == 5450
    assert report["train_occurrences"] == 5460
    assert report["synthetic_occurrences"] == 10
    assert report["expected_optimizer_steps"] == 342
    assert report["real_selector_counts"] == {
        "БАД:0": 1500,
        "БАД:1": 1500,
        "Легковоспламеняющиеся:0": 1600,
        "Легковоспламеняющиеся:1": 850,
    }
    train, validation, loaded = trainer.load_runtime(output, -1, refit=True)
    assert len(train) == 5460
    assert validation == []
    assert loaded["contract_sha256"] == report["contract_sha256"]
    assert {int(row["fold"]) for row in train if not row.get("synthetic")} == set(
        range(5)
    )


def test_balanced_v2_cap10_append_adds_ten_per_label(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    train = [
        {
            "category": "Легковоспламеняющиеся" if index < 1700 else "БАД",
            "description": "real description",
            "fold": 1,
            "global_index": index,
            "id": f"real-{index}",
            "image_url": "https://example.invalid/image.jpg",
            "label": index % 2,
            "name": "real",
            "occurrence_index": 0,
            "ocr_images": [],
            "semantic_component": f"component-{index}",
        }
        for index in range(4892)
    ]
    validation = [
        {
            "category": "Легковоспламеняющиеся",
            "description": "validation",
            "fold": 0,
            "global_index": 1000 + index,
            "id": f"validation-{index}",
            "image_url": "https://example.invalid/validation.jpg",
            "name": "validation",
            "occurrence_index": 0,
            "ocr_images": [],
            "semantic_component": f"validation-component-{index}",
        }
        for index in range(2224)
    ]
    write_jsonl(parent / "train.jsonl", train)
    write_jsonl(parent / "validation.jsonl", validation)
    audit = {
        "outer_fold": 0,
        "train_occurrences": len(train),
        "validation_rows": len(validation),
        "output_sha256": {
            "train.jsonl": builder.sha256_file(parent / "train.jsonl"),
            "validation.jsonl": builder.sha256_file(parent / "validation.jsonl"),
        },
    }
    audit["contract_sha256"] = builder.canonical_sha256(audit)
    (parent / "runtime_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    ranked = tmp_path / "ranked.jsonl"
    write_jsonl(
        ranked,
        [
            {
                "candidate_id": f"v2-{label}-{index}",
                "category": "Легковоспламеняющиеся",
                "description": "synthetic description long enough",
                "label": label,
                "label_rank": index + 1,
                "name": f"synthetic {label} {index}",
                "selector_fold": 0,
                "source": "v2",
            }
            for label in (0, 1)
            for index in range(12)
        ],
    )
    output = tmp_path / "append"
    report = append_builder.build_append_runtime(
        parent_dir=parent,
        ranked_path=ranked,
        output_dir=output,
        fold=0,
        source="v2",
        mode="balanced",
        cap=10,
    )
    loaded = builder.read_jsonl(output / "train.jsonl")
    loaded_train, loaded_validation, loaded_audit = trainer.load_runtime(output, 0)
    synthetic = [row for row in loaded if row.get("synthetic")]
    assert loaded_train == loaded
    assert len(loaded_validation) == 2224
    assert loaded_audit["contract_sha256"] == report["contract_sha256"]
    assert len(loaded) == len(train) + 20
    assert sum(row["label"] == 0 for row in synthetic) == 10
    assert sum(row["label"] == 1 for row in synthetic) == 10
    assert report["appended_negative_occurrences"] == 10
    assert report["appended_positive_occurrences"] == 10
    assert report["expected_optimizer_steps"] == 307

    repeat_output = tmp_path / "append-repeat4"
    repeat_report = append_builder.build_append_runtime(
        parent_dir=parent,
        ranked_path=ranked,
        output_dir=repeat_output,
        fold=0,
        source="v2",
        mode="positive_only",
        cap=10,
        synthetic_repeat=4,
    )
    repeat_train, _, _ = trainer.load_runtime(repeat_output, 0)
    repeated = [row for row in repeat_train if row.get("synthetic")]
    assert len(repeated) == 40
    assert len({row["id"] for row in repeated}) == 40
    assert len({row["semantic_component"] for row in repeated}) == 10
    assert {row["synthetic_repeat_index"] for row in repeated} == {0, 1, 2, 3}
    assert repeat_report["synthetic_unique"] == 10
    assert repeat_report["synthetic_occurrences"] == 40
    assert repeat_report["expected_optimizer_steps"] == 309

    repeat2_output = tmp_path / "append-repeat2"
    repeat2_report = append_builder.build_append_runtime(
        parent_dir=parent,
        ranked_path=ranked,
        output_dir=repeat2_output,
        fold=0,
        source="v2",
        mode="positive_only",
        cap=10,
        synthetic_repeat=2,
    )
    repeat2_train, _, _ = trainer.load_runtime(repeat2_output, 0)
    repeated2 = [row for row in repeat2_train if row.get("synthetic")]
    assert len(repeated2) == 20
    assert len({row["semantic_component"] for row in repeated2}) == 10
    assert {row["synthetic_repeat_index"] for row in repeated2} == {0, 1}
    assert repeat2_report["synthetic_occurrences"] == 20
    assert repeat2_report["expected_optimizer_steps"] == 307


def test_submission_runtime_patch_keeps_q3_and_matches_q35_parent_preprocessing() -> None:
    source = (
        ROOT.parent
        / "140_dual_lora_fusion"
        / "submission"
        / "run.py"
    ).read_text(encoding="utf-8")
    patched = submission_builder.patch_submission_run(source)
    assert "max_pixels=262144" in patched
    assert "if max_pixels is None:" in patched
    assert "image.thumbnail((448, 448)" in patched
    assert "scale = math.sqrt(max_pixels / (width * height))" in patched
    compile(patched, "candidate-run.py", "exec")


def test_submission_smoke_input_and_output_are_label_free_and_exact(tmp_path: Path) -> None:
    source = tmp_path / "validation.jsonl"
    rows = [
        {
            "id": f"{category_index}-{index}",
            "global_index": category_index * 10 + index,
            "fold": 0,
            "category": category,
            "name": f"name {index}",
            "description": f"description {index}",
        }
        for category_index, category in enumerate(smoke_builder.CATEGORIES)
        for index in range(5)
    ]
    write_jsonl(source, rows)
    mounted = tmp_path / "work" / "s3"
    mounted.mkdir(parents=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    for row in rows:
        target = mounted / f"{row['id']}.jpg"
        target.write_bytes(b"image")
        smoke_builder.image_cache_path(cache, row["id"]).symlink_to(target)
    output = tmp_path / "smoke"
    report = smoke_builder.prepare(
        source_validation=source,
        image_cache=cache,
        output_dir=output,
        approved_s3_root=mounted,
    )
    assert report["rows"] == 8
    assert report["categories"] == {
        "БАД": 4,
        "Легковоспламеняющиеся": 4,
    }
    with (output / "data.csv").open(encoding="utf-8", newline="") as stream:
        inputs = list(__import__("csv").DictReader(stream))
    submission = tmp_path / "submission.csv"
    with submission.open("w", encoding="utf-8", newline="") as stream:
        writer = __import__("csv").DictWriter(stream, fieldnames=["id", "result"])
        writer.writeheader()
        for row in inputs:
            writer.writerow(
                {
                    "id": row["id"],
                    "result": (
                        "<комментарий>Текст и изображения достаточно подробно "
                        "проверены для технического smoke теста.<вердикт>не бан"
                    ),
                }
            )
    accepted = smoke_builder.verify_output(output, submission)
    assert accepted["decision"] == "ACCEPT_PACKAGE_RUNTIME_SMOKE"
    assert accepted["rows"] == 8


def test_package_runner_safe_extract_is_exact_and_rejects_traversal(
    tmp_path: Path,
) -> None:
    import zipfile

    archive = tmp_path / "candidate.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("run.py", "print('ok')\n")
        output.writestr("adapter/file.bin", b"adapter")
    extracted = tmp_path / "extracted"
    inventory = package_runner.safe_extract_exact(archive, extracted)
    assert inventory == package_runner.regular_files(extracted)
    assert set(inventory) == {"run.py", "adapter/file.bin"}

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as output:
        output.writestr("run.py", "print('ok')\n")
        output.writestr("../escape", b"forbidden")
    try:
        package_runner.safe_extract_exact(unsafe, tmp_path / "unsafe-output")
    except ValueError as error:
        assert "unsafe ZIP member" in str(error)
    else:
        raise AssertionError("path traversal ZIP was accepted")


def test_replacement_positions_are_unique_items() -> None:
    rows = []
    for occurrence in range(5):
        for index in range(50):
            rows.append(
                {
                    "category": "Легковоспламеняющиеся",
                    "label": 1,
                    "id": str(index),
                    "semantic_component": f"c{index:03d}",
                    "occurrence_index": occurrence,
                }
            )
    positions = builder._replacement_positions(rows, label=1, count=40)
    assert len(positions) == 40
    assert len({rows[index]["id"] for index in positions}) == 40


def test_build_runtime_preserves_counts(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    train = []
    for index in range(4892):
        positive = index < 100
        train.append(
            {
                "category": "Легковоспламеняющиеся" if index < 1700 else "БАД",
                "description": "real description",
                "fold": 1,
                "global_index": index,
                "id": str(index),
                "image_url": "https://example.invalid/image.jpg",
                "label": int(positive),
                "name": "real",
                "occurrence_index": 0,
                "ocr_images": [],
                "semantic_component": f"component-{index}",
            }
        )
    validation = [
        {
            "category": "БАД",
            "description": "validation",
            "fold": 0,
            "global_index": 9999 + index,
            "id": f"validation-{index}",
            "image_url": "https://example.invalid/validation.jpg",
            "name": "validation",
            "occurrence_index": 0,
            "ocr_images": [],
            "semantic_component": f"validation-component-{index}",
        }
        for index in range(2224)
    ]
    write_jsonl(parent / "train.jsonl", train)
    write_jsonl(parent / "validation.jsonl", validation)
    audit = {
        "outer_fold": 0,
        "train_occurrences": 4892,
        "validation_rows": 2224,
        "output_sha256": {
            "train.jsonl": builder.sha256_file(parent / "train.jsonl"),
            "validation.jsonl": builder.sha256_file(parent / "validation.jsonl"),
        },
    }
    audit["contract_sha256"] = builder.canonical_sha256(audit)
    (parent / "runtime_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    ranked = tmp_path / "ranked.jsonl"
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "category": "Легковоспламеняющиеся",
            "description": "synthetic description long enough",
            "label": 1,
            "label_rank": index,
            "name": f"synthetic {index}",
            "selector_fold": 0,
            "source": "v1",
        }
        for index in range(40)
    ]
    write_jsonl(ranked, candidates)
    output = tmp_path / "output"
    result = builder.build_runtime(
        parent_dir=parent,
        ranked_path=ranked,
        output_dir=output,
        fold=0,
        source="v1",
        mode="positive_only",
        cap=40,
    )
    output_rows = builder.read_jsonl(output / "train.jsonl")
    assert len(output_rows) == 4892
    assert sum(bool(row.get("synthetic")) for row in output_rows) == 40
    assert result["bad_rows_changed"] == 0
    assert result["validation_rows_changed"] == 0
    assert result["contract_sha256"] == builder.canonical_sha256(
        {key: value for key, value in result.items() if key != "contract_sha256"}
    )

    balanced_ranked = tmp_path / "balanced.jsonl"
    balanced = [
        {
            "candidate_id": f"balanced-{label}-{index}",
            "category": "Легковоспламеняющиеся",
            "description": "balanced synthetic description long enough",
            "label": label,
            "label_rank": index,
            "name": f"balanced {label} {index}",
            "selector_fold": 0,
            "source": "v1" if index % 2 else "v2",
        }
        for label in (0, 1)
        for index in range(80)
    ]
    write_jsonl(balanced_ranked, balanced)
    balanced_output = tmp_path / "balanced-output"
    balanced_result = builder.build_runtime(
        parent_dir=parent,
        ranked_path=balanced_ranked,
        output_dir=balanced_output,
        fold=0,
        source="both",
        mode="balanced",
        cap=80,
    )
    assert balanced_result["synthetic_by_label"] == {"0": 80, "1": 80}
    assert balanced_result["synthetic_occurrences"] == 160

    v2_ranked = tmp_path / "v2-ranked.jsonl"
    v2_candidates = [
        {
            "candidate_id": f"v2-{index}",
            "category": "Легковоспламеняющиеся",
            "description": "v2 synthetic description long enough",
            "label": 1,
            "label_rank": index,
            "name": f"v2 synthetic {index}",
            "selector_fold": 0,
            "source": "v2",
        }
        for index in range(19)
    ]
    write_jsonl(v2_ranked, v2_candidates)
    v2_output = tmp_path / "v2-output"
    builder.build_runtime(
        parent_dir=parent,
        ranked_path=v2_ranked,
        output_dir=v2_output,
        fold=0,
        source="v2",
        mode="positive_only",
        cap=19,
    )
    loaded_train, loaded_validation, loaded_audit = trainer.load_runtime(v2_output, 0)
    assert len(loaded_train) == 4892
    assert len(loaded_validation) == 2224
    assert loaded_audit["synthetic_occurrences"] == 19


def test_candidate_uses_frozen_baseline_selection_without_retuning() -> None:
    import numpy as np

    categories = np.asarray(["БАД", "БАД", "БАД", "БАД"])
    folds = np.asarray([0, 0, 1, 1], dtype=np.int8)
    named_scores = {
        "robust_base": np.asarray([0.1, 0.9, 0.4, 0.8]),
        "qwen3vl": np.asarray([0.2, 0.8, 0.3, 0.9]),
        "qwen35": np.asarray([0.9, 0.1, 0.7, 0.2]),
    }
    selection = {
        "БАД": {
            "0": {
                "weights": {"robust_base": 0.5, "qwen3vl": 0.5, "qwen35": 0.0},
                "threshold": 0.5,
            },
            "1": {
                "weights": {"robust_base": 0.0, "qwen3vl": 0.0, "qwen35": 1.0},
                "threshold": 0.5,
            },
        }
    }
    predictions = evaluator.apply_frozen_selection(
        categories, folds, named_scores, selection
    )
    assert predictions.tolist() == [0, 1, 1, 0]

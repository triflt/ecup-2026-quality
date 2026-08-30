from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).parent
REPO = ROOT.parents[1]
TEST_DATA = Path(os.environ.get("EXP699_TEST_DATA", REPO / "research/data.csv"))
TEST_OOF = Path(
    os.environ.get(
        "EXP699_TEST_OOF", REPO / "research/four-head-r2-extracted/four_head_oof.npz"
    )
)
sys.path.insert(0, str(ROOT))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load(
    "exp699_solution140_full_support_builder",
    "build_solution140_full_support_refit.py",
)
trainer = load("exp699_solution140_full_support_trainer", "train_synth_fold.py")
cache_builder = load(
    "exp699_solution140_full_support_cache",
    "prepare_solution140_qwen35_image_cache.py",
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def ranked_inputs(tmp_path: Path) -> list[tuple[int, Path]]:
    result = []
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
            ],
        )
        result.append((fold, path))
    return result


@pytest.mark.parametrize(
    ("synth_cap", "preprocessing", "expected_train"),
    (
        (10, "thumbnail448", 6800),
        (5, "thumbnail448", 6795),
        (10, "area_cap262144", 6800),
    ),
)
def test_solution140_full_support_selection_and_trainer_gate(
    tmp_path: Path, synth_cap: int, preprocessing: str, expected_train: int
) -> None:
    output = tmp_path / f"runtime-{synth_cap}-{preprocessing}"
    report = builder.build(
        SimpleNamespace(
            data=TEST_DATA,
            oof=TEST_OOF,
            reference_code=REPO / "research/qwen35_family_balanced_lora.py",
            reference_report=(REPO / "research/qwen35-full-extracted/full_train_report.json"),
            ranked=ranked_inputs(tmp_path),
            synth_cap=synth_cap,
            preprocessing=preprocessing,
            epochs=1,
            output_dir=output,
        )
    )
    assert report["schema"] == "exp699_solution140_full_support_refit_runtime_v1"
    assert report["decision"] == "GO_FULL_REFIT_SOLUTION140_FULL_SUPPORT"
    assert report["original_real_occurrences"] == 6790
    assert report["original_real_unique"] == 5998
    assert report["train_occurrences"] == expected_train
    assert report["synthetic_occurrences"] == synth_cap
    assert report["preprocessing_key"] == preprocessing
    assert report["preprocessing_sha256"] == builder.canonical_sha256(
        builder.PREPROCESSING_CONTRACTS[preprocessing]
    )
    assert report["expected_optimizer_steps"] == 425
    assert report["real_selector_counts"] == builder.EXPECTED_REAL_COUNTS
    assert {
        key: report["input_sha256"][key] for key in builder.EXPECTED_INPUT_SHA256
    } == builder.EXPECTED_INPUT_SHA256

    train, validation, loaded = trainer.load_runtime(output, -1, refit=True)
    assert len(train) == expected_train
    assert validation == []
    assert loaded["contract_sha256"] == report["contract_sha256"]
    real = [row for row in train if not row.get("synthetic")]
    assert len(real) == 6790
    assert len({row["global_index"] for row in real}) == 5998
    assert (
        Counter(f"{row['category']}:{row['label']}" for row in real) == builder.EXPECTED_REAL_COUNTS
    )


def test_old_refit_gate_is_not_weakened(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    train_path = runtime / "train.jsonl"
    validation_path = runtime / "validation.jsonl"
    write_jsonl(train_path, [])
    write_jsonl(validation_path, [])
    audit = {
        "schema": "exp699_solution140_full_support_refit_runtime_v1",
        "decision": "GO_FULL_REFIT",
        "outer_fold": -1,
        "train_occurrences": 6800,
        "original_real_occurrences": 6790,
        "original_real_unique": 5998,
        "synthetic_occurrences": 10,
        "output_sha256": {
            "train.jsonl": trainer.sha256_file(train_path),
            "validation.jsonl": trainer.sha256_file(validation_path),
        },
    }
    audit["contract_sha256"] = trainer.canonical_sha256(audit)
    (runtime / "runtime_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    try:
        trainer.load_runtime(runtime, -1, refit=True)
    except ValueError as error:
        assert "occurrence count mismatch" in str(error)
    else:
        raise AssertionError("old GO_FULL_REFIT decision accepted corrected counts")


def test_qwen35_cache_materializes_original_448_thumbnail(tmp_path: Path, monkeypatch) -> None:
    audit = {
        "contract_sha256": "a" * 64,
        "decision": "GO_FULL_REFIT_SOLUTION140_FULL_SUPPORT",
        "preprocessing_key": "thumbnail448",
    }
    monkeypatch.setattr(cache_builder, "load_runtime_ids", lambda _: (["large", "small"], audit))
    mounted = tmp_path / "mounted"
    for row_id, size in (("large", (1000, 500)), ("small", (100, 50))):
        directory = mounted / row_id
        directory.mkdir(parents=True)
        Image.new("RGB", size, "red").save(directory / "0.jpg", quality=95)
    cache = tmp_path / "cache"
    report = cache_builder.prepare(
        tmp_path / "runtime",
        mounted,
        cache,
        tmp_path / "cache-report.json",
        "thumbnail448",
        workers=2,
    )
    assert report["decision"] == "ACCEPT_SOLUTION140_QWEN35_IMAGE_CACHE"
    assert report["unique_real_ids"] == 2
    with Image.open(cache_builder.cache_path(cache, "large", "qwen35_4b")) as image:
        assert image.size == (448, 224)
    with Image.open(cache_builder.cache_path(cache, "small", "qwen35_4b")) as image:
        assert image.size == (100, 50)
    assert not cache_builder.cache_path(cache, "large", "qwen35_4b").is_symlink()

    reused = cache_builder.prepare(
        tmp_path / "runtime",
        mounted,
        cache,
        tmp_path / "cache-report-reused.json",
        "thumbnail448",
        workers=2,
        reuse_existing=True,
    )
    assert reused["created"] == 0
    assert reused["reused"] == 2


def test_qwen35_area_cap_cache_keeps_mounted_sources_as_symlinks(
    tmp_path: Path, monkeypatch
) -> None:
    audit = {
        "contract_sha256": "b" * 64,
        "decision": "GO_FULL_REFIT_SOLUTION140_FULL_SUPPORT",
        "preprocessing_key": "area_cap262144",
    }
    monkeypatch.setattr(cache_builder, "load_runtime_ids", lambda _: (["large"], audit))
    mounted = tmp_path / "mounted" / "large"
    mounted.mkdir(parents=True)
    Image.new("RGB", (1000, 500), "blue").save(mounted / "0.jpg", quality=95)
    cache = tmp_path / "cache"
    report = cache_builder.prepare(
        tmp_path / "runtime",
        mounted.parent,
        cache,
        tmp_path / "cache-report.json",
        "area_cap262144",
        workers=1,
    )
    cached = cache_builder.cache_path(cache, "large", "qwen35_4b")
    assert cached.is_symlink()
    assert cached.resolve() == (mounted / "0.jpg").resolve()
    assert report["preprocessing_key"] == "area_cap262144"


def test_corrected_refit_training_shape_preserves_425_updates() -> None:
    assert trainer.training_shape(6800, 2, 8, 1) == {
        "batches_per_epoch": 3400,
        "updates_per_epoch": 425,
        "optimizer_steps": 425,
        "training_occurrences_seen": 6800,
    }


def test_full_support_hard_selector_uses_stable_tie_order() -> None:
    indices = np.arange(20, dtype=np.int64)[::-1]
    scores = np.zeros(20, dtype=np.float32)
    selected = builder.hard_random(indices, scores, 10, np.random.default_rng(42))
    assert selected[:5] == [19, 18, 17, 16, 15]

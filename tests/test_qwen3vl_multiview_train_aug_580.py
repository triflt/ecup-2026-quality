from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import json
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/580_qwen3vl_multiview_train_aug"
sys.path.insert(0, str(EXPERIMENT))

import gallery_augmentation as gallery
import multiview_contract as contract


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


launcher = _load("exp580_train_screen", "train_screen.py")
evaluator = _load("exp580_evaluate_screen", "evaluate_screen.py")


def _jpeg(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="JPEG", quality=95)
    return output.getvalue()


def _manifest(path: Path, rows: dict[str, int]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image_urls"], delimiter="\t")
        writer.writeheader()
        for item_id, count in rows.items():
            writer.writerow(
                {
                    "id": item_id,
                    "image_urls": json.dumps(
                        [f"memory://{item_id}/{index}" for index in range(count)]
                    ),
                }
            )


def test_plan_is_deterministic_and_cycles_every_gallery_position() -> None:
    first = [
        contract.training_image_index(
            item_id="A-12", seed=42, epoch=0, occurrence=i, gallery_size=5
        )
        for i in range(10)
    ]
    second = [
        contract.training_image_index(
            item_id="A-12", seed=42, epoch=0, occurrence=i, gallery_size=5
        )
        for i in range(10)
    ]

    assert first == second
    assert sorted(first[:5]) == [0, 1, 2, 3, 4]
    assert first[:5] == first[5:]
    assert contract.inference_image_index(gallery_size=5) == 0
    assert (
        contract.training_image_index(
            item_id="single", seed=42, epoch=0, occurrence=9, gallery_size=1
        )
        == 0
    )


def test_label_blind_full_archive_audit_and_first_image_null(tmp_path: Path) -> None:
    archive_path = tmp_path / "images.zip"
    manifest_path = tmp_path / "gallery.tsv.gz"
    _manifest(manifest_path, {"one": 1, "many": 3})
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("images/one/0.jpg", _jpeg((900, 600), (1, 2, 3)))
        archive.writestr("images/many/0.jpg", _jpeg((500, 700), (4, 5, 6)))
        archive.writestr("images/many/1.jpg", _jpeg((700, 500), (7, 8, 9)))
        archive.writestr("images/many/2.jpg", _jpeg((300, 300), (10, 11, 12)))

    report = gallery.audit_archive(
        images_zip=archive_path,
        gallery_manifest=manifest_path,
        progress_every=0,
    )

    assert report["status"] == "go_for_two_fold_screen"
    assert report["label_blind"] is True
    assert report["labels_categories_folds_read"] is False
    assert report["rows"] == 2
    assert report["images"] == 4
    assert report["multi_image_fraction"] == 0.5
    assert report["decode_failures"] == 0
    assert report["deterministic_plan_failures"] == 0
    assert report["gallery_cycle_coverage_failures"] == 0
    assert report["first_image_null_byte_mismatches"] == 0


def test_archive_audit_fails_closed_on_decode_error(tmp_path: Path) -> None:
    archive_path = tmp_path / "images.zip"
    manifest_path = tmp_path / "gallery.tsv.gz"
    _manifest(manifest_path, {"broken": 1})
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("images/broken/0.jpg", b"not-an-image")

    report = gallery.audit_archive(
        images_zip=archive_path,
        gallery_manifest=manifest_path,
        progress_every=0,
    )

    assert report["status"] == "no_go"
    assert report["decode_failures"] == 1
    assert report["decode_failure_records"][0]["error_type"]


def test_launcher_locks_parent_recipe_and_only_screen_folds() -> None:
    environment: dict[str, str] = {}
    configured = launcher.configure_environment(0, environment)

    assert configured["SEED"] == "42"
    assert configured["TRAINING_MODE"] == "hard"
    assert configured["MODEL_CLASS"] == "image_text"
    assert configured["DESCRIPTION_LIMIT"] == "1800"
    assert configured["QWEN3VL_FIRST_IMAGE_MAX_EDGE"] == "448"
    assert configured["QWEN3VL_FIRST_IMAGE_MAX_PIXELS"] == "262144"
    assert configured["HOLDOUT_FOLD"] == "0"
    with pytest.raises(ValueError, match="predeclared screen"):
        launcher.configure_environment(1, {})
    with pytest.raises(ValueError, match="non-parent switches"):
        launcher.configure_environment(3, {"FULL_TRAIN": "1"})
    with pytest.raises(ValueError, match="parent value"):
        launcher.configure_environment(3, {"SEED": "7"})
    with pytest.raises(ValueError, match="parent value"):
        launcher.configure_environment(3, {"QWEN3VL_FIRST_IMAGE_MAX_EDGE": "672"})


def test_occurrence_planner_reports_coverage_without_changing_count() -> None:
    planner = gallery.OccurrencePlanner({"a": 3, "b": 1})
    selected = [planner.choose("a") for _ in range(3)]
    selected += [planner.choose("b") for _ in range(2)]
    report = planner.report()

    assert sorted(selected[:3]) == [0, 1, 2]
    assert selected[3:] == [0, 0]
    assert report["training_occurrences"] == 5
    assert report["training_unique_rows"] == 2
    assert report["repeated_rows_with_maximal_cycle_coverage"] == 2
    planner.training = False
    assert planner.choose("a") == 0
    assert planner.report()["training_occurrences"] == 5


def test_candidate_report_proves_train_only_augmentation(tmp_path: Path) -> None:
    path = tmp_path / "lora_holdout_report.json"
    path.write_text(
        json.dumps(
            {
                "holdout_fold": 0,
                "train_records": 17,
                "soft_targets": False,
                "last_logit_only": False,
                "training_view_policy": contract.TRAINING_VIEW_POLICY,
                "gallery_seed": 42,
                "gallery_epoch": 0,
                "training_images_per_occurrence": 1,
                "inference_view_policy": contract.INFERENCE_VIEW_POLICY,
                "inference_image_index": 0,
                "inference_image_count": 1,
                "inference_passes": 1,
                "first_image_max_edge": 448,
                "first_image_max_pixels": 262144,
                "download_failures": 0,
                "gallery_plan": {"training_occurrences": 17},
            }
        )
    )

    checked = evaluator.validate_candidate_report(path, fold=0)

    assert checked["training_view_policy"] == contract.TRAINING_VIEW_POLICY
    assert checked["inference_image_index"] == 0
    raw = json.loads(path.read_text())
    raw["inference_image_index"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="inference_image_index"):
        evaluator.validate_candidate_report(path, fold=0)


def test_acceptance_uses_only_predeclared_causal_gates() -> None:
    fold = {
        "fold": 0,
        "delta_macro_f1": 0.002,
        "corrected": 3,
        "regressed": 1,
        "flammable_false_negatives": {"delta": 0},
        "safety_union_false_negatives": {"delta": -1},
    }
    result = evaluator.apply_acceptance(
        {
            "mode": "candidate",
            "folds": [dict(fold), {**fold, "fold": 3}],
            "evaluated_folds": [0, 3],
            "mean_screen_delta_macro_f1": 0.002,
        }
    )

    assert result["screen_passed"] is True
    assert result["experiment_id"] == "580"
    result["folds"][1]["corrected"] = 1
    result["folds"][1]["regressed"] = 1
    assert evaluator.apply_acceptance(result)["screen_passed"] is False


def test_experiment_card_keeps_inference_and_parent_recipe_locked() -> None:
    config = tomllib.loads((EXPERIMENT / "experiment.toml").read_text())
    metrics = json.loads((EXPERIMENT / "results/metrics.json").read_text())

    assert config["validation"]["screen_folds"] == [0, 3]
    assert config["training"]["images_per_occurrence"] == 1
    assert config["training"]["first_image_max_edge"] == 448
    assert config["training"]["prompt"] == "byte-identical training parent"
    assert config["training"]["selector"] == "byte-identical training parent"
    assert config["training"]["steps"] == "byte-identical training parent"
    assert config["training"]["weights"] == "byte-identical training parent"
    assert config["inference"]["image_index"] == 0
    assert config["inference"]["images_per_row"] == 1
    assert config["inference"]["passes"] == 1
    assert config["execution"]["gpu_count"] == 1
    assert config["execution"]["launch_authorized"] is True
    assert metrics["training_launched"] is True

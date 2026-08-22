from __future__ import annotations

import io
import sys
import tomllib
import zipfile
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/510_qwen3vl_first_image_672"
sys.path.insert(0, str(EXPERIMENT))
contract = import_module("resolution_contract")
rendering = import_module("audit_rendering")
runtime = import_module("runtime_projection")
launcher = import_module("train_screen")
evaluator = import_module("evaluate_screen")


def _jpeg(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="JPEG", quality=95)
    return stream.getvalue()


def test_resolution_contract_is_the_only_changed_parent_factor() -> None:
    assert contract.BASELINE_MAX_EDGE == 448
    assert contract.CANDIDATE_MAX_EDGE == 672
    assert contract.CANDIDATE_MAX_PIXELS == 451_584
    assert contract.SCREEN_FOLDS == (0, 3)

    parent_source = (ROOT / "research/qwen3vl_lora_holdout.py").read_text()
    assert 'os.environ.get("QWEN3VL_FIRST_IMAGE_MAX_EDGE", "448")' in parent_source
    assert 'os.environ.get("QWEN3VL_FIRST_IMAGE_MAX_PIXELS", "262144")' in parent_source
    assert "max_pixels=FIRST_IMAGE_MAX_PIXELS" in parent_source
    assert "(FIRST_IMAGE_MAX_EDGE, FIRST_IMAGE_MAX_EDGE)" in parent_source


def test_launcher_locks_two_folds_and_parent_recipe() -> None:
    environment: dict[str, str] = {}
    configured = launcher.configure_environment(0, environment)

    assert configured["HOLDOUT_FOLD"] == "0"
    assert configured["SEED"] == "42"
    assert configured["TRAINING_MODE"] == "hard"
    assert configured["MODEL_CLASS"] == "image_text"
    assert configured["QWEN3VL_FIRST_IMAGE_MAX_EDGE"] == "672"
    assert configured["QWEN3VL_FIRST_IMAGE_MAX_PIXELS"] == "451584"
    with pytest.raises(ValueError, match="predeclared screen"):
        launcher.configure_environment(1, {})
    with pytest.raises(ValueError, match="non-parent switches"):
        launcher.configure_environment(3, {"FULL_TRAIN": "1"})
    with pytest.raises(ValueError, match="non-parent switches"):
        launcher.configure_environment(3, {"SOFT_TARGETS": "0"})
    with pytest.raises(ValueError, match="parent value"):
        launcher.configure_environment(3, {"SEED": "31415"})
    with pytest.raises(ValueError, match="locked resolution"):
        launcher.configure_environment(
            3, {"QWEN3VL_FIRST_IMAGE_MAX_PIXELS": "262144"}
        )


def test_label_blind_rendering_audit_checks_coverage_and_geometry(tmp_path: Path) -> None:
    archive_path = tmp_path / "images.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("images/large-wide/0.jpg", _jpeg((1200, 800), (100, 30, 20)))
        archive.writestr("images/small/0.jpg", _jpeg((200, 100), (20, 100, 30)))
        archive.writestr("images/large-tall/0.jpg", _jpeg((900, 1200), (30, 20, 100)))
        archive.writestr("images/large-wide/1.jpg", b"not part of first-image audit")

    report = rendering.audit_archive(archive_path, sample_size=3)

    assert report["status"] == "go_for_two_fold_screen"
    assert report["label_blind"] is True
    assert report["label_or_category_columns_read"] is False
    assert report["first_image_only"] is True
    assert report["decode_failures"] == 0
    assert report["same_cover_failures"] == 0
    assert report["aspect_failures"] == 0
    assert report["more_than_25_percent_additional_pixels"] == 2
    assert all(record["same_cover_image"] for record in report["records"])


def test_rendering_audit_fails_closed_on_decode_error(tmp_path: Path) -> None:
    archive_path = tmp_path / "broken.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("images/broken/0.jpg", b"not-an-image")

    report = rendering.audit_archive(archive_path, sample_size=1)

    assert report["status"] == "no_go"
    assert report["decode_failures"] == 1
    assert report["records"][0]["decode_ok"] is False
    assert report["records"][0]["error_type"]


def test_runtime_projection_requires_exactly_600_rows() -> None:
    passed = runtime.project_runtime(300.0)
    failed = runtime.project_runtime(330.0)

    assert passed["status"] == "runtime_gate_passed"
    assert passed["projected_public_minutes"] == pytest.approx(13.3333333333)
    assert passed["projected_private_minutes"] == pytest.approx(31.6666666667)
    assert failed["status"] == "runtime_gate_failed"
    with pytest.raises(ValueError, match="exactly 600"):
        runtime.project_runtime(300.0, observed_rows=599)


def test_experiment_card_records_only_the_predeclared_reject_screen() -> None:
    config = tomllib.loads((EXPERIMENT / "experiment.toml").read_text())
    metrics = (EXPERIMENT / "results/metrics.json").read_text()

    assert config["validation"]["screen_folds"] == [0, 3]
    assert config["acceptance"]["screen"]["folds"] == [0, 3]
    assert config["training"]["candidate_max_edge"] == 672
    assert config["training"]["candidate_max_pixels"] == 451_584
    assert config["training"]["crop"] is False
    assert config["training"]["gallery_images"] is False
    assert config["training"]["tta"] is False
    assert config["execution"]["launch_authorized"] is False
    assert '"training_launched": true' in metrics
    assert '"status": "rejected_two_fold_screen"' in metrics
    assert '"decision": "reject_higher_resolution_regressed_both_folds"' in metrics


def test_candidate_report_must_prove_locked_resolution(tmp_path: Path) -> None:
    report_path = tmp_path / "lora_holdout_report.json"
    report_path.write_text(
        '{"holdout_fold": 0, "first_image_max_edge": 672, '
        '"first_image_max_pixels": 451584}'
    )

    assert evaluator.validate_candidate_report(report_path, fold=0) == {
        "holdout_fold": 0,
        "first_image_max_edge": 672,
        "first_image_max_pixels": 451_584,
    }
    report_path.write_text(
        '{"holdout_fold": 0, "first_image_max_edge": 448, '
        '"first_image_max_pixels": 451584}'
    )
    with pytest.raises(ValueError, match="expected 672"):
        evaluator.validate_candidate_report(report_path, fold=0)


def test_h2_fold_gate_reuses_route_audit_with_stricter_bad_guard() -> None:
    labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    categories = np.asarray(
        [evaluator.base.BAD] * 4 + [evaluator.base.FLAMMABLE] * 4
    )
    folds = np.zeros(8, dtype=np.int8)
    baseline = np.asarray([0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8)
    candidate = labels.copy()
    safe = np.ones(8, dtype=bool)
    safety_union = np.asarray(
        [False, False, False, False, True, False, False, False]
    )

    report = evaluator.h2_fold_audit(
        fold=0,
        labels=labels,
        categories=categories,
        folds=folds,
        baseline=baseline,
        candidate=candidate,
        safe=safe,
        safety_union=safety_union,
    )

    assert report["passed"] is True
    assert report["corrected"] == 2
    assert report["regressed"] == 0
    assert report["flammable_false_negatives"]["delta"] == -1
    assert report["gates"]["bad_f1_drop_not_over_0_003"] is True


def test_screen_outputs_are_fail_closed_against_overwrite(tmp_path: Path) -> None:
    paths = evaluator.output_paths(tmp_path, null_control=True)
    assert paths[0].name == "null_screen_control.json"
    paths[0].write_text("existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator.ensure_output_targets_absent(paths)

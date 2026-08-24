from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/677_qwen35_pr_auc_dynamics_screen"


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD = load("experiment_677_build", "build_inner_runtimes.py")
TRAIN = load("experiment_677_train", "train_dynamics.py")
EVALUATE = load("experiment_677_evaluate", "evaluate_inner.py")
GATE = load("experiment_677_gate", "verify_launch_gate.py")
PRESET = load("experiment_677_preset", "build_private_preset.py")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def write_source_runtime(path: Path, fold: int, train: list[dict], validation: list[dict]) -> None:
    path.mkdir()
    write_jsonl(path / "train.jsonl", train)
    write_jsonl(path / "validation.jsonl", validation)
    audit = {
        "outer_fold": fold,
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "output_sha256": {
            "train.jsonl": BUILD.sha256_file(path / "train.jsonl"),
            "validation.jsonl": BUILD.sha256_file(path / "validation.jsonl"),
        },
    }
    audit["contract_sha256"] = BUILD.canonical_sha256(audit)
    (path / "runtime_audit.json").write_text(json.dumps(audit), encoding="utf-8")


def test_builder_excludes_outer_and_inner_folds(tmp_path: Path) -> None:
    outer_train = [
        {
            "id": str(fold * 10 + index),
            "fold": fold,
            "category": "БАД" if index < 2 else "Легковоспламеняющиеся",
            "label": index % 2,
        }
        for fold in (1, 2, 3, 4)
        for index in range(4)
    ]
    outer = tmp_path / "outer"
    write_source_runtime(outer, 0, outer_train, [{"id": "outer", "fold": 0}])
    validation_runtimes = []
    for fold in (1, 2, 3, 4):
        runtime = tmp_path / f"source{fold}"
        validation = [{"id": str(fold * 100 + index), "fold": fold} for index in range(2)]
        write_source_runtime(runtime, fold, [], validation)
        validation_runtimes.append(runtime)
    output = tmp_path / "nested"
    result = BUILD.build(
        outer_runtime=outer, validation_runtimes=validation_runtimes, output_dir=output
    )
    assert result["public_used"] is False
    for fold in (1, 2, 3, 4):
        train = BUILD.read_jsonl(output / f"inner_fold{fold}" / "train.jsonl")
        validation = BUILD.read_jsonl(output / f"inner_fold{fold}" / "validation.jsonl")
        assert len(train) == 12
        assert all(row["fold"] not in {0, fold} for row in train)
        assert all(row["fold"] == fold and "label" not in row for row in validation)


def test_tail_trim_only_removes_bad_negatives() -> None:
    rows = [
        {"id": "a", "category": "Легковоспламеняющиеся", "label": 1},
        {"id": "b", "category": "БАД", "label": 0},
        {"id": "c", "category": "БАД", "label": 1},
        {"id": "d", "category": "БАД", "label": 0},
        {"id": "e", "category": "БАД", "label": 0},
    ]
    trimmed, removed = BUILD.trim_to_frozen_micro_batch(rows)
    assert len(trimmed) == 4
    assert removed == ["e"]


def test_checkpoint_schedule_is_fractional_and_final() -> None:
    assert TRAIN.checkpoint_steps(229) == [57, 114, 172, 229]
    assert TRAIN.checkpoint_steps(2) == [1, 2]


def test_average_precision_groups_ties() -> None:
    assert EVALUATE.average_precision([1, 0, 1, 0], [0.9, 0.8, 0.8, 0.1]) == pytest.approx(
        5 / 6
    )


def test_evaluator_selects_one_shared_nonfinal_fraction(tmp_path: Path) -> None:
    registry_path = tmp_path / "folds.csv"
    fields = ["id", "category", "label", "semantic_component", "split", "development_fold"]
    registry_rows = []
    run_dirs = []
    for fold in (1, 2, 3, 4):
        ids = [fold * 10 + index for index in range(4)]
        for item_id, category, label in zip(
            ids,
            ("БАД", "БАД", "Легковоспламеняющиеся", "Легковоспламеняющиеся"),
            (1, 0, 1, 0),
            strict=True,
        ):
            registry_rows.append(
                {
                    "id": item_id,
                    "category": category,
                    "label": label,
                    "semantic_component": str(item_id),
                    "split": "development",
                    "development_fold": fold,
                }
            )
        run_dir = tmp_path / f"run{fold}"
        run_dir.mkdir()
        checkpoints = []
        for rank, fraction in enumerate(EVALUATE.REQUESTED_FRACTIONS, start=1):
            checkpoint_dir = run_dir / f"step_{rank:04d}"
            checkpoint_dir.mkdir()
            improve = fraction == 0.5 and fold in {1, 2, 3}
            predictions = [
                {"id": ids[0], "score": 1.0},
                {"id": ids[1], "score": 0.0},
                {"id": ids[2], "score": 1.0 if improve else 0.0},
                {"id": ids[3], "score": 0.0 if improve else 1.0},
            ]
            prediction_path = checkpoint_dir / "predictions.jsonl"
            write_jsonl(prediction_path, predictions)
            checkpoints.append(
                {
                    "requested_training_fraction": fraction,
                    "training_fraction": fraction,
                    "optimizer_step": rank,
                    "predictions": str(prediction_path.relative_to(run_dir)),
                    "predictions_sha256": EVALUATE.sha256_file(prediction_path),
                }
            )
        report = {
            "experiment_id": "677",
            "outer_screen_fold": 0,
            "inner_validation_fold": fold,
            "changed_factor": "optimizer_stop_fraction_only",
            "technical_smoke": False,
            "validation_labels_read": 0,
            "sealed_rows": 0,
            "public_used": False,
            "threshold_tuned": False,
            "decision": "READY_FOR_INNER_EVALUATION",
            "checkpoint_fractions": list(EVALUATE.REQUESTED_FRACTIONS),
            "checkpoints": checkpoints,
        }
        report["contract_sha256"] = EVALUATE.canonical_sha256(report)
        (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        run_dirs.append(run_dir)
    with registry_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(registry_rows)
    result = EVALUATE.evaluate(registry_path, run_dirs, tmp_path / "metrics.json")
    assert result["selected_training_fraction"] == 0.5
    assert result["decision"] == "GO_CONFIRM_SELECTED_STOP_ON_OUTER_FOLDS_0_AND_3"
    assert result["public_used"] is False


def test_launch_gate_requires_terminal_accepted_parent(tmp_path: Path) -> None:
    gate_path = tmp_path / "gate.json"
    gate = {
        "experiment_id": "677",
        "parent_experiment_id": "659",
        "parent_result_sha256": "a" * 64,
        "parent_passed": True,
        "parent_decision": "ACCEPT_FULL_COMPONENT_ROUTE",
        "training_lane_open": True,
        "decision": "OPEN_FOUR_INNER_DYNAMICS_SCREENS",
        "gpu_jobs": 4,
        "gpus_per_job": 1,
        "inner_folds": [1, 2, 3, 4],
        "sealed_rows": 0,
        "public_used": False,
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    assert GATE.verify(gate_path, 3) == gate
    gate["parent_passed"] = False
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="closed"):
        GATE.verify(gate_path, 3)


def test_private_preset_reuses_proven_single_gpu_runtime(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    url = tmp_path / "url.txt"
    base.write_text(
        """job:
  time_limit: 8h
  flavor: h100-1x
  region: region
  image: image
  preemption: forbidden
  work_dir: /work
  env:
    TOKENIZERS_PARALLELISM: "false"
    PYTORCH_ALLOC_CONF: expandable_segments:True
  input:
    - {type: model_registry, src: model, dst: /hf_models/}
""",
        encoding="utf-8",
    )
    url.write_text("https://example.invalid/private-bundle", encoding="utf-8")
    output = tmp_path / "preset.yml"
    payload = PRESET.build(
        SimpleNamespace(
            base_preset=base,
            bundle_url_file=url,
            inner_fold=2,
            output=output,
        )
    )
    assert "flavor: h100-1x" in payload
    assert "--inner-fold 2" in payload
    assert "train_dynamics.py" in payload
    assert "pip install" not in payload
    assert payload.count("type: model_registry") == 1
    assert "example.invalid" not in payload


def test_committed_runtime_manifest_and_gate_are_fail_closed() -> None:
    results = EXPERIMENT / "results"
    manifest = json.loads((results / "runtime_manifest.json").read_text(encoding="utf-8"))
    payload = dict(manifest)
    digest = payload.pop("contract_sha256")
    assert digest == BUILD.canonical_sha256(payload)
    metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["runtime_manifest_sha256"] == BUILD.sha256_file(
        results / "runtime_manifest.json"
    )
    assert metrics["gpu_jobs_launched"] == 0
    with pytest.raises(ValueError, match="closed"):
        GATE.verify(results / "launch_gate.json", 1)

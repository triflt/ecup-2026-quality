from __future__ import annotations

import importlib.util
import csv
import json
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/679_qwen35_4b_lower_lr"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = load_module(EXPERIMENT / "train_fold.py", "experiment_679_train")
GATE = load_module(EXPERIMENT / "verify_launch_gate.py", "experiment_679_gate")
PRESET = load_module(EXPERIMENT / "build_private_preset.py", "experiment_679_preset")
EVALUATOR = load_module(EXPERIMENT / "evaluate.py", "experiment_679_evaluator")
ARTIFACT = load_module(EXPERIMENT / "verify_artifact.py", "experiment_679_artifact")


def canonical_without_hash(payload: dict) -> str:
    result = dict(payload)
    result.pop("contract_sha256", None)
    return TRAIN.control.canonical_sha256(result)


def test_contract_rewrite_changes_only_declared_scientific_factor() -> None:
    parent = {
        "schema_version": 1,
        "experiment_id": "641",
        "objective": "class_only",
        "model_id": "Qwen/Qwen3.5-4B",
        "optimizer_steps_executed": 306,
        "technical_smoke": False,
        "runtime_micro_batch_size": 2,
        "runtime_gradient_accumulation": 8,
        "effective_batch_size": 16,
        "seed": 42,
        "threshold": 0.0,
        "threshold_tuned": False,
        "artifacts": {"predictions.jsonl": "abc", "adapter.zip": "def"},
        "contract_sha256": "parent-sha",
    }
    result = TRAIN.rewrite_contract(parent)
    assert result["experiment_id"] == "679"
    assert result["control_experiment_id"] == "641"
    assert result["changed_factor"] == "learning_rate_only"
    assert result["control_learning_rate"] == 2e-4
    assert result["candidate_learning_rate"] == 1e-4
    assert result["uses_27b_at_training"] is False
    assert result["uses_27b_at_inference"] is False
    for key in (
        "objective",
        "model_id",
        "optimizer_steps_executed",
        "runtime_micro_batch_size",
        "runtime_gradient_accumulation",
        "effective_batch_size",
        "seed",
        "threshold",
        "threshold_tuned",
        "artifacts",
    ):
        assert result[key] == parent[key]
    assert result["contract_sha256"] == canonical_without_hash(result)


def write_runtime(path: Path, gate: dict, fold: int) -> None:
    path.mkdir(parents=True)
    train = b'{"id":"train"}\n'
    validation = b'{"id":"validation"}\n'
    (path / "train.jsonl").write_bytes(train)
    (path / "validation.jsonl").write_bytes(validation)
    gate["runtime_payload_sha256"][str(fold)] = {
        "train.jsonl": GATE.sha256_file(path / "train.jsonl"),
        "validation.jsonl": GATE.sha256_file(path / "validation.jsonl"),
    }
    (path / "runtime_audit.json").write_text(
        json.dumps({"contract_sha256": gate["runtime_contract_sha256"][str(fold)]}),
        encoding="utf-8",
    )


def test_launch_gate_is_fail_closed_until_explicitly_opened(tmp_path: Path) -> None:
    gate = json.loads((EXPERIMENT / "results/launch_gate.json").read_text(encoding="utf-8"))
    gate["decision"] = "PREPARED_WAIT_CAPACITY"
    gate["training_lane_open"] = False
    gate["allowed_folds"] = []
    runtime = tmp_path / "runtime"
    write_runtime(runtime, gate, 0)
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="closed"):
        GATE.verify(path, runtime, 0)

    gate["decision"] = "OPEN_SCREEN"
    gate["training_lane_open"] = True
    gate["allowed_folds"] = [0, 3]
    path.write_text(json.dumps(gate), encoding="utf-8")
    assert GATE.verify(path, runtime, 0)["candidate_learning_rate"] == 1e-4


def test_gate_rejects_runtime_payload_drift(tmp_path: Path) -> None:
    gate = json.loads((EXPERIMENT / "results/launch_gate.json").read_text(encoding="utf-8"))
    gate["decision"] = "PREPARED_WAIT_CAPACITY"
    gate["training_lane_open"] = False
    gate["allowed_folds"] = []
    gate["decision"] = "OPEN_SCREEN"
    gate["training_lane_open"] = True
    gate["allowed_folds"] = [0, 3]
    runtime = tmp_path / "runtime"
    write_runtime(runtime, gate, 0)
    (runtime / "train.jsonl").write_text("changed\n", encoding="utf-8")
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="payload mismatch"):
        GATE.verify(path, runtime, 0)


def test_private_preset_preserves_4b_recipe_and_rejects_closed_gate(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    base.write_text(
        """job:
  time_limit: 4h
  flavor: h100-1x
  region: frozen-region
  image: frozen-image
  preemption: never
  work_dir: /work
  env:
    TOKENIZERS_PARALLELISM: \"false\"
    PYTORCH_ALLOC_CONF: expandable_segments:True
  input:
    - {type: model_registry, name: frozen-model, dst: /hf_models}
""",
        encoding="utf-8",
    )
    url = tmp_path / "url.txt"
    url.write_text("https://example.invalid/unique-bundle.tar.gz\n", encoding="utf-8")
    patch = tmp_path / "train_lora.py"
    patch.write_text("# frozen", encoding="utf-8")
    gate = json.loads((EXPERIMENT / "results/launch_gate.json").read_text(encoding="utf-8"))
    gate["decision"] = "PREPARED_WAIT_CAPACITY"
    gate["training_lane_open"] = False
    gate["allowed_folds"] = []
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "base_preset": base,
            "bundle_url_file": url,
            "gate": gate_path,
            "train_lora_patch": patch,
            "technical_smoke": False,
            "fold": 0,
            "output": tmp_path / "preset.yml",
        },
    )()
    with pytest.raises(ValueError, match="not open"):
        PRESET.build(args)
    gate["decision"] = "OPEN_SCREEN"
    gate["training_lane_open"] = True
    gate["allowed_folds"] = [0, 3]
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    payload = PRESET.build(args)
    assert "h100-1x" in payload
    assert "679_qwen35_4b_lower_lr/train_fold.py" in payload
    assert "--micro-batch-size-override 2" in payload
    assert "Qwen3.6" not in payload and "27B" not in payload


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_evaluation_fixture(tmp_path: Path, folds: list[int]):
    registry_rows = []
    per_fold: dict[int, tuple[Path, Path]] = {}
    global_index = 0
    for fold in folds:
        baseline_rows, candidate_rows = [], []
        for category, labels, base_scores, candidate_scores in (
            ("БАД", [1, 0, 1, 0], [2.0, -2.0, 1.0, -1.0], [2.0, -2.0, 1.0, -1.0]),
            (
                "Легковоспламеняющиеся",
                [1, 0, 1, 0],
                [-0.5, 0.4, 0.3, -0.2],
                [1.0, -1.0, 0.8, -0.8],
            ),
        ):
            for local_index, (label, base_score, candidate_score) in enumerate(
                zip(labels, base_scores, candidate_scores, strict=True)
            ):
                row_id = f"{fold}{0 if category == 'БАД' else 1}{local_index}"
                registry_rows.append(
                    {
                        "id": row_id,
                        "category": category,
                        "label": label,
                        "semantic_component": f"component-{row_id}",
                        "component_size": 1,
                        "split": "development",
                        "development_fold": fold,
                    }
                )
                common = {
                    "global_index": global_index,
                    "id": row_id,
                    "fold": fold,
                    "category": category,
                }
                baseline_rows.append(
                    {
                        **common,
                        "score": base_score,
                        "prediction": int(base_score >= 0.0),
                    }
                )
                candidate_rows.append(
                    {
                        **common,
                        "score": candidate_score,
                        "prediction": int(candidate_score >= 0.0),
                    }
                )
                global_index += 1
        baseline_path = tmp_path / f"baseline{fold}.jsonl"
        candidate_path = tmp_path / f"candidate{fold}.jsonl"
        write_jsonl(baseline_path, baseline_rows)
        write_jsonl(candidate_path, candidate_rows)
        per_fold[fold] = (baseline_path, candidate_path)
    registry = tmp_path / "folds.csv"
    with registry.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(registry_rows[0]))
        writer.writeheader()
        writer.writerows(registry_rows)
    return registry, per_fold


def test_tie_aware_average_precision_matches_expected_grouping() -> None:
    labels = EVALUATOR.np.asarray([1, 0, 1, 0], dtype=EVALUATOR.np.int8)
    scores = EVALUATOR.np.asarray([1.0, 1.0, 0.0, 0.0])
    assert EVALUATOR.average_precision(labels, scores) == pytest.approx(0.5)


def test_screen_and_full_gates_use_ap_and_blind_confirmation(tmp_path: Path) -> None:
    registry, per_fold = make_evaluation_fixture(tmp_path, [0, 1, 2, 3, 4])
    screen = EVALUATOR.evaluate(
        registry_path=registry,
        baseline_paths=[per_fold[fold][0] for fold in (0, 3)],
        candidate_paths=[per_fold[fold][1] for fold in (0, 3)],
        mode="screen",
        output_path=tmp_path / "screen.json",
    )
    assert screen["passed"] is True
    assert screen["decision"] == "OPEN_CONFIRMATION_FOLDS"
    assert screen["gates"]["both_screen_folds_flammable_ap_positive"] is True
    full = EVALUATOR.evaluate(
        registry_path=registry,
        baseline_paths=[per_fold[fold][0] for fold in (0, 1, 2, 3, 4)],
        candidate_paths=[per_fold[fold][1] for fold in (0, 1, 2, 3, 4)],
        mode="full",
        screen_gate_path=tmp_path / "screen.json",
        output_path=tmp_path / "full.json",
    )
    assert full["passed"] is True
    assert full["decision"] == "ACCEPT_FOR_FULL_REFIT"
    assert full["gates"]["all_confirmation_folds_macro_positive"] is True
    assert full["gates"]["screen_reproduced_exactly"] is True


def make_full_artifact(tmp_path: Path, fold: int = 0) -> tuple[Path, Path]:
    runtime_path = tmp_path / "validation.jsonl"
    predictions: list[dict] = []
    runtime_rows: list[dict] = []
    for index in range(ARTIFACT.EXPECTED_VALIDATION_ROWS[fold]):
        common = {
            "global_index": index,
            "id": f"item-{index}",
            "fold": fold,
            "category": "БАД",
        }
        score = 1.0 if index % 2 == 0 else -1.0
        runtime_rows.append(common)
        predictions.append(
            {
                **common,
                "concept": "synthetic",
                "format_valid": True,
                "generated_verdict": "да" if score >= 0.0 else "нет",
                "grounded": True,
                "grounding_source": "text",
                "image_index": 0,
                "model_id": "Qwen/Qwen3.5-4B",
                "model_revision": ARTIFACT.MODEL_REVISION,
                "objective": "class_only",
                "prediction": int(score >= 0.0),
                "preprocessing_version": "v1",
                "prompt_version": "v1",
                "quote": "synthetic",
                "raw_generation": "synthetic",
                "region_index": 0,
                "score": score,
                "target_order": 0,
            }
        )
    write_jsonl(runtime_path, runtime_rows)
    predictions_payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in predictions
    ).encode()
    contract = {
        "schema_version": 1,
        "experiment_id": "679",
        "control_experiment_id": "641",
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": ARTIFACT.MODEL_REVISION,
        "objective": "class_only",
        "changed_factor": "learning_rate_only",
        "control_learning_rate": 0.0002,
        "candidate_learning_rate": 0.0001,
        "outer_fold": fold,
        "train_occurrences": ARTIFACT.EXPECTED_TRAIN_OCCURRENCES[fold],
        "validation_rows": ARTIFACT.EXPECTED_VALIDATION_ROWS[fold],
        "optimizer_steps_executed": 306,
        "runtime_micro_batch_size": 2,
        "runtime_gradient_accumulation": 8,
        "effective_batch_size": 16,
        "runtime_contract_sha256": ARTIFACT.EXPECTED_RUNTIME_CONTRACTS[fold],
        "technical_smoke": False,
        "threshold": 0.0,
        "threshold_tuned": False,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "uses_27b_at_training": False,
        "uses_27b_at_inference": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
        "decision": "GO_EVALUATE",
        "artifacts": {"predictions.jsonl": ARTIFACT.sha256_bytes(predictions_payload)},
    }
    contract["contract_sha256"] = ARTIFACT.canonical_sha256(contract)
    archive_path = tmp_path / "artifact.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("output_contract.json", json.dumps(contract, ensure_ascii=False))
        archive.writestr("predictions.jsonl", predictions_payload)
        archive.writestr(
            "adapter/adapter_config.json",
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B"}),
        )
        archive.writestr("adapter/adapter_model.safetensors", b"synthetic")
    return archive_path, runtime_path


def test_full_artifact_verifier_is_bound_to_frozen_4b_runtime(tmp_path: Path) -> None:
    archive_path, runtime_path = make_full_artifact(tmp_path)
    result = ARTIFACT.verify(archive_path, fold=0, runtime_validation=runtime_path)
    assert result["decision"] == "PASS"
    assert result["rows"] == 2224
    assert result["exact_runtime_binding"] is True
    assert result["deployable_4b_only"] is True


def test_full_artifact_verifier_rejects_row_reordering(tmp_path: Path) -> None:
    archive_path, runtime_path = make_full_artifact(tmp_path)
    rows = [json.loads(line) for line in runtime_path.read_text(encoding="utf-8").splitlines()]
    write_jsonl(runtime_path, list(reversed(rows)))
    with pytest.raises(ValueError, match="frozen runtime validation"):
        ARTIFACT.verify(archive_path, fold=0, runtime_validation=runtime_path)

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "experiments/645_qwen_scale_2x3_gate"
while str(GRID) in sys.path:
    sys.path.remove(str(GRID))
sys.path.insert(0, str(GRID))

for module_name in ("grid_contract", "verify_training_artifact"):
    sys.modules.pop(module_name, None)

import grid_contract
import verify_training_artifact


def build_artifact(
    path: Path, *, include_label: bool = False, fold: int = 0, rows: int = 1
) -> None:
    path.mkdir()
    predictions = path / "predictions.jsonl"
    prediction_rows = []
    for index in range(rows):
        prediction = {
            "id": f"row-{index}",
            "fold": fold,
            "model_id": grid_contract.CELL_SPECS["641"].model_id,
            "score": 0.25,
            "prediction": 1,
        }
        if include_label:
            prediction["label"] = 1
        prediction_rows.append(json.dumps(prediction) + "\n")
    predictions.write_text("".join(prediction_rows), encoding="utf-8")
    adapter = path / "adapter.zip"
    with zipfile.ZipFile(adapter, "w") as archive:
        archive.writestr("adapter/adapter_config.json", "{}")
        archive.writestr("adapter/adapter_model.safetensors", b"weights")
    contract = {
        "schema_version": 1,
        "experiment_id": "641",
        "outer_fold": fold,
        "model_id": grid_contract.CELL_SPECS["641"].model_id,
        "model_revision": grid_contract.CELL_SPECS["641"].model_revision,
        "objective": "class_only",
        "grid_contract_sha256": grid_contract.GRID_CONTRACT_SHA256,
        "runtime_contract_sha256": "runtime",
        "model_input_view_sha256": "view",
        "seed": 42,
        "epochs": 1,
        "effective_batch_size": 16,
        "evidence_auxiliary_weight": 0.0,
        "optimized_training_kernels": True,
        "fast_path_packages": verify_training_artifact.EXPECTED_FAST_PATH_PACKAGES,
        "fast_path_bindings": {
            "chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk",
            "recurrent_gated_delta_rule": "fla.ops.gated_delta_rule.fused_recurrent",
            "causal_conv1d_fn": "causal_conv1d.causal_conv1d_interface",
            "causal_conv1d_update": "causal_conv1d.causal_conv1d_interface",
        },
        "target_orders": [],
        "train_occurrences": 10,
        "validation_rows": rows,
        "technical_smoke": False,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "threshold": 0.0,
        "threshold_tuned": False,
        "runtime_minutes": 1.0,
        "artifacts": {
            "predictions.jsonl": grid_contract.sha256_file(predictions),
            "adapter.zip": grid_contract.sha256_file(adapter),
        },
        "decision": "GO_EVALUATE",
    }
    contract["contract_sha256"] = grid_contract.canonical_sha256(contract)
    (path / "output_contract.json").write_text(json.dumps(contract), encoding="utf-8")


def test_verifier_accepts_complete_class_only_output(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    build_artifact(artifact)
    report = verify_training_artifact.verify_artifact(
        artifact, experiment_id="641", fold=0, technical_smoke=False
    )
    assert report["decision"] == "ACCEPT_ARTIFACT"
    assert report["prediction_rows"] == 1


def test_verifier_accepts_frozen_641_legacy_eager_execution(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    build_artifact(artifact, rows=2224)
    contract_path = artifact / "output_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update(
        {
            "runtime_backend": "legacy_eager",
            "runtime_packages": verify_training_artifact.EXPECTED_LEGACY_PACKAGES,
            "fast_path_bindings": {},
            "optimized_training_kernels": False,
            "frozen_micro_batch_size": 4,
            "frozen_gradient_accumulation": 4,
            "runtime_micro_batch_size": 2,
            "runtime_gradient_accumulation": 8,
            "effective_batch_size": 16,
            "batching_override": True,
            "train_occurrences": 4892,
            "frozen_optimizer_updates": 306,
            "runtime_optimizer_updates": 306,
            "optimizer_steps_executed": 306,
            "tail_rows": 12,
            "selection_contract_changed": False,
            "numerical_accumulation_order_changed": True,
        }
    )
    contract.pop("fast_path_packages")
    contract.pop("contract_sha256")
    contract["contract_sha256"] = grid_contract.canonical_sha256(contract)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    report = verify_training_artifact.verify_artifact(
        artifact, experiment_id="641", fold=0, technical_smoke=False
    )
    assert report["runtime_backend"] == "legacy_eager"


def test_verifier_accepts_fold1_legacy_eager_row_counts(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    build_artifact(artifact, fold=1, rows=2223)
    predictions_path = artifact / "predictions.jsonl"
    contract_path = artifact / "output_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update(
        {
            "runtime_backend": "legacy_eager",
            "runtime_packages": verify_training_artifact.EXPECTED_LEGACY_PACKAGES,
            "fast_path_bindings": {},
            "optimized_training_kernels": False,
            "frozen_micro_batch_size": 4,
            "frozen_gradient_accumulation": 4,
            "runtime_micro_batch_size": 2,
            "runtime_gradient_accumulation": 8,
            "effective_batch_size": 16,
            "batching_override": True,
            "train_occurrences": 4894,
            "validation_rows": 2223,
            "frozen_optimizer_updates": 306,
            "runtime_optimizer_updates": 306,
            "optimizer_steps_executed": 306,
            "tail_rows": 14,
            "frozen_micro_batch_tail_rows": 2,
            "tail_loss_divisor": 4,
            "selection_contract_changed": False,
            "numerical_accumulation_order_changed": True,
        }
    )
    contract["artifacts"]["predictions.jsonl"] = grid_contract.sha256_file(predictions_path)
    contract.pop("fast_path_packages")
    contract.pop("contract_sha256")
    contract["contract_sha256"] = grid_contract.canonical_sha256(contract)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    report = verify_training_artifact.verify_artifact(
        artifact, experiment_id="641", fold=1, technical_smoke=False
    )
    assert report["prediction_rows"] == 2223


def test_verifier_rejects_legacy_eager_package_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    build_artifact(artifact)
    contract_path = artifact / "output_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update(
        {
            "runtime_backend": "legacy_eager",
            "runtime_packages": {**verify_training_artifact.EXPECTED_LEGACY_PACKAGES, "torch": "other"},
            "fast_path_bindings": {},
            "optimized_training_kernels": False,
        }
    )
    contract.pop("fast_path_packages")
    contract.pop("contract_sha256")
    contract["contract_sha256"] = grid_contract.canonical_sha256(contract)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="package versions"):
        verify_training_artifact.verify_artifact(
            artifact, experiment_id="641", fold=0, technical_smoke=False
        )


def test_verifier_rejects_prediction_supervision(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    build_artifact(artifact, include_label=True)
    with pytest.raises(ValueError, match="supervision fields"):
        verify_training_artifact.verify_artifact(
            artifact, experiment_id="641", fold=0, technical_smoke=False
        )


def test_verifier_rejects_unsafe_adapter_member(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    build_artifact(artifact)
    adapter = artifact / "adapter.zip"
    with zipfile.ZipFile(adapter, "a") as archive:
        archive.writestr("../escape", "bad")
    contract_path = artifact / "output_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["artifacts"]["adapter.zip"] = grid_contract.sha256_file(adapter)
    contract.pop("contract_sha256")
    contract["contract_sha256"] = grid_contract.canonical_sha256(contract)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe path"):
        verify_training_artifact.verify_artifact(
            artifact, experiment_id="641", fold=0, technical_smoke=False
        )

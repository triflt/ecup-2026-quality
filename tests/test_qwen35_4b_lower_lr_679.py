from __future__ import annotations

import importlib.util
import json
import sys
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

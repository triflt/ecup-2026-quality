from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/683_gemma4_e4b_class_only_lora_screen"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_gate_is_self_hashed_and_smoke_only(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(HERE))
    contract = load("contract683", HERE / "contract.py")
    gate = json.loads((HERE / "results/launch_gate.json").read_text())
    assert contract.verify_self_hash(gate)
    assert gate["decision"] == "OPEN_TECHNICAL_SMOKE_ONLY"
    assert gate["allowed_folds"] == [0]
    assert gate["max_gpu_jobs"] == 1
    assert gate["public_used"] is False


def test_frozen_source_runtime_hashes(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(HERE))
    contract = load("contract683b", HERE / "contract.py")
    for fold in (0, 3):
        source = ROOT / f"experiments/641_qwen35_4b_class_only_lora/.local/runtime/fold{fold}"
        if not source.exists():
            continue
        assert contract.sha256_file(source / "train.jsonl") == contract.EXPECTED_TRAIN_SHA256[fold]
        assert contract.sha256_file(source / "validation.jsonl") == contract.EXPECTED_VALIDATION_SHA256[fold]


def test_target_topology_is_text_only(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(HERE))
    trainer = load("trainer683", HERE / "train_fold.py")

    class Linear:
        pass

    class Dummy:
        def named_modules(self):
            for layer in range(42):
                yield f"model.language_model.layers.{layer}.self_attn.q_proj", Linear()
                yield f"model.language_model.layers.{layer}.self_attn.o_proj", Linear()
                if layer < 24:
                    yield f"model.language_model.layers.{layer}.self_attn.k_proj", Linear()
                    yield f"model.language_model.layers.{layer}.self_attn.v_proj", Linear()
            yield "model.vision_tower.layers.0.self_attn.q_proj", Linear()

    targets, counts = trainer.resolve_text_targets(Dummy())
    assert len(targets) == 132
    assert counts == {"k_proj": 24, "o_proj": 42, "q_proj": 42, "v_proj": 24}
    assert all("vision" not in name and "audio" not in name for name in targets)

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP500_SELECTOR = ROOT / "experiments/500_transaction_scope_position_aug/parent_selector.py"
EXP500_SELECTOR_SHA256 = "04e87a97b9e124f1bef48adbdf9416cfb15cc95a1cdc4c404145a610555373a3"
EXP600_PROTOCOL = ROOT / "experiments/600_semantic_v3_qwen35_baselines/protocol.py"
EXP600_TRAINER = ROOT / "experiments/600_semantic_v3_qwen35_baselines/train_component.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_exp600_dependencies():
    protocol = load_module("_exp610_shared_semantic_v3_protocol", EXP600_PROTOCOL)
    previous_protocol = sys.modules.get("protocol")
    sys.modules["protocol"] = protocol
    try:
        trainer = load_module("_exp610_shared_semantic_v3_trainer", EXP600_TRAINER)
    finally:
        if previous_protocol is None:
            sys.modules.pop("protocol", None)
        else:
            sys.modules["protocol"] = previous_protocol
    expected = {
        "PROTOCOL_VERSION": "semantic_family_v3",
        "SELECTOR_SOURCE_PROTOCOL": "semantic_v3_robust_base_strict_nested_v2",
        "DEVELOPMENT_ROWS": 11_118,
        "SEALED_ROWS": 1_853,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(protocol, key, None)}
        for key, value in expected.items()
        if getattr(protocol, key, None) != value
    }
    if mismatches:
        raise RuntimeError(f"experiment-600 protocol changed: {mismatches}")
    if trainer.PARENT_SHA256.get("specialist") != (
        "404f6d07965f551dfe7c7ee0120ce0a16132e1103f1db7d13b3622993f4fd6c0"
    ):
        raise RuntimeError("exact route-400 specialist parent contract changed")
    return protocol, trainer


def load_dependency_light_selector():
    if sha256_file(EXP500_SELECTOR) != EXP500_SELECTOR_SHA256:
        raise RuntimeError("dependency-light specialist selector changed")
    return load_module("_exp610_dependency_light_selector", EXP500_SELECTOR)

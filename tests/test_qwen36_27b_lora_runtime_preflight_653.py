from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "experiments"
    / "653_qwen36_27b_lora_runtime_preflight"
    / "technical_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("qwen36_lora_smoke_653", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_device_map_accepts_four_cuda_devices() -> None:
    summary = MODULE.validate_device_map(
        {"a": 0, "b": "cuda:1", "c": 2, "d": "3"}, minimum_cuda_devices=4
    )
    assert set(summary) == {"cuda:0", "cuda:1", "cuda:2", "cuda:3"}


@pytest.mark.parametrize("device", ["cpu", "disk", "meta"])
def test_device_map_rejects_offload(device: str) -> None:
    with pytest.raises(RuntimeError, match="offloaded"):
        MODULE.validate_device_map({"a": "cuda:0", "b": device}, minimum_cuda_devices=1)


def test_device_map_requires_true_sharding() -> None:
    with pytest.raises(RuntimeError, match="expected at least 4"):
        MODULE.validate_device_map({"a": 0, "b": "cuda:0"}, minimum_cuda_devices=4)

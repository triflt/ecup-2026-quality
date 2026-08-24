from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "experiments/653_qwen36_27b_lora_runtime_preflight/verify_artifact.py"
)
SPEC = importlib.util.spec_from_file_location("verify_653_artifact", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def make_artifact(path: Path, *, device_map: dict[str, int] | None = None) -> None:
    adapter = {
        "README.md": b"adapter",
        "adapter_config.json": b"{}",
        "adapter_model.safetensors": b"weights",
    }
    report = {
        "schema_version": 1,
        "experiment_id": "653",
        "model_id": VERIFY.MODEL_ID,
        "model_revision": VERIFY.MODEL_REVISION,
        "seed": 42,
        "synthetic_rows": 2,
        "competition_rows": 0,
        "sealed_rows": 0,
        "cuda_device_count": 4,
        "device_map_summary": device_map
        or {"cuda:0": 1, "cuda:1": 1, "cuda:2": 1, "cuda:3": 1},
        "gradients_finite": True,
        "losses": [0.4, 0.2],
        "optimizer_steps": 1,
        "adapter_reloaded": True,
        "reload_score": 0.5,
        "cpu_or_disk_offload": False,
        "packages": VERIFY.EXPECTED_PACKAGES,
        "adapter_manifest": {
            name: hashlib.sha256(payload).hexdigest() for name, payload in adapter.items()
        },
        "decision": "TECHNICAL_GO",
    }
    inner_buffer = BytesIO()
    with zipfile.ZipFile(inner_buffer, "w") as inner:
        inner.writestr("report.json", json.dumps(report))
        for name, payload in adapter.items():
            inner.writestr(f"adapter/{name}", payload)
    inner_payload = inner_buffer.getvalue()
    delivery = {"archive_sha256": hashlib.sha256(inner_payload).hexdigest()}
    with zipfile.ZipFile(path, "w") as outer:
        outer.writestr("report.json", json.dumps(report))
        outer.writestr("delivery.json", json.dumps(delivery))
        outer.writestr("qwen36_27b_lora_smoke.zip", inner_payload)


def test_verifier_accepts_complete_technical_artifact(tmp_path: Path) -> None:
    path = tmp_path / "artifact.zip"
    make_artifact(path)
    result = VERIFY.verify(path)
    assert result["decision"] == "PASS_OPEN_EXPERIMENT_654"
    assert result["cuda_devices"] == 4


def test_verifier_rejects_cpu_offload_or_incomplete_cuda_map(tmp_path: Path) -> None:
    path = tmp_path / "artifact.zip"
    make_artifact(path, device_map={"cuda:0": 1, "cuda:1": 1, "cuda:2": 1, "cpu": 1})
    with pytest.raises(ValueError, match="four CUDA devices"):
        VERIFY.verify(path)

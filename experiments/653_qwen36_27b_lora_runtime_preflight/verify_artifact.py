from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
EXPECTED_PACKAGES = {
    "peft": "0.20.0",
    "torch": "2.10.0+cu128",
    "transformers": "5.14.1",
}
REQUIRED_ADAPTER_FILES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_members(archive: zipfile.ZipFile) -> set[str]:
    bad = archive.testzip()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    names = set(archive.namelist())
    if any(
        PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        for name in names
    ):
        raise ValueError("unsafe ZIP member")
    return names


def verify(path: Path) -> dict[str, Any]:
    outer_payload = path.read_bytes()
    with zipfile.ZipFile(path) as outer:
        outer_names = safe_members(outer)
        required_outer = {"delivery.json", "report.json", "qwen36_27b_lora_smoke.zip"}
        if not required_outer <= outer_names:
            raise ValueError("outer artifact members are incomplete")
        delivery = json.loads(outer.read("delivery.json"))
        outer_report = json.loads(outer.read("report.json"))
        inner_payload = outer.read("qwen36_27b_lora_smoke.zip")

    if delivery.get("archive_sha256") != sha256_bytes(inner_payload):
        raise ValueError("inner archive checksum differs from delivery report")

    from io import BytesIO

    with zipfile.ZipFile(BytesIO(inner_payload)) as inner:
        inner_names = safe_members(inner)
        required_inner = {"report.json"} | {
            f"adapter/{name}" for name in REQUIRED_ADAPTER_FILES
        }
        if not required_inner <= inner_names:
            raise ValueError("inner artifact members are incomplete")
        inner_report = json.loads(inner.read("report.json"))
        adapter_manifest = {
            name.removeprefix("adapter/"): sha256_bytes(inner.read(name))
            for name in inner_names
            if name.startswith("adapter/") and not name.endswith("/")
        }

    if outer_report != inner_report:
        raise ValueError("outer and inner reports differ")
    expected = {
        "schema_version": 1,
        "experiment_id": "653",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": 42,
        "synthetic_rows": 2,
        "competition_rows": 0,
        "sealed_rows": 0,
        "cuda_device_count": 4,
        "gradients_finite": True,
        "optimizer_steps": 1,
        "adapter_reloaded": True,
        "cpu_or_disk_offload": False,
        "packages": EXPECTED_PACKAGES,
        "decision": "TECHNICAL_GO",
    }
    mismatch = {
        key: {"expected": value, "actual": inner_report.get(key)}
        for key, value in expected.items()
        if inner_report.get(key) != value
    }
    if mismatch:
        raise ValueError(f"technical report contract mismatch: {mismatch}")
    device_map = inner_report.get("device_map_summary")
    if not isinstance(device_map, dict) or set(device_map) != {
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
    }:
        raise ValueError("model was not placed across the frozen four CUDA devices")
    if any(not isinstance(count, int) or count <= 0 for count in device_map.values()):
        raise ValueError("device map contains an empty CUDA shard")
    losses = inner_report.get("losses")
    if not isinstance(losses, list) or len(losses) != 2 or not all(
        math.isfinite(float(value)) for value in losses
    ):
        raise ValueError("training losses are missing or non-finite")
    if not math.isfinite(float(inner_report.get("reload_score", math.nan))):
        raise ValueError("adapter reload score is non-finite")
    if inner_report.get("adapter_manifest") != adapter_manifest:
        raise ValueError("adapter manifest differs from archive contents")
    if set(adapter_manifest) != REQUIRED_ADAPTER_FILES:
        raise ValueError("adapter file set differs from the frozen contract")

    return {
        "schema_version": 1,
        "experiment_id": "653",
        "outer_archive_sha256": sha256_bytes(outer_payload),
        "inner_archive_sha256": sha256_bytes(inner_payload),
        "cuda_devices": len(device_map),
        "finite_losses": len(losses),
        "optimizer_steps": 1,
        "adapter_reloaded": True,
        "decision": "PASS_OPEN_EXPERIMENT_654",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify experiment-653 ML artifact.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.archive)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite verification output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

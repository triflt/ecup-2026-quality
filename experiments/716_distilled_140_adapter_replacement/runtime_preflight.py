from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


EXPECTED = {
    "torch": "2.9.0",
    "paddlepaddle-gpu": "3.3.0",
    "paddleocr": "3.7.0",
    "paddlex": "3.7.2",
    "qwen-vl-utils": "0.0.14",
}


def installed_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in EXPECTED}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("refusing to overwrite runtime preflight report")
    versions = installed_versions()
    mismatches = {
        name: {"expected_prefix": expected, "actual": versions[name]}
        for name, expected in EXPECTED.items()
        if not versions[name].startswith(expected)
    }
    if mismatches:
        raise ValueError(f"runtime package version mismatch: {mismatches}")

    import paddle
    import torch
    from paddleocr import PaddleOCR  # noqa: F401
    from qwen_vl_utils.vision_process import process_vision_info  # noqa: F401

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("preflight requires exactly one visible CUDA GPU for torch")
    if not paddle.device.is_compiled_with_cuda() or paddle.device.cuda.device_count() != 1:
        raise RuntimeError("preflight requires exactly one visible CUDA GPU for paddle")
    torch_value = float((torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")).sum().cpu())
    paddle_left = paddle.ones([32, 32], dtype="float32").cuda()
    paddle_value = float(paddle.matmul(paddle_left, paddle_left).sum().cpu())
    if torch_value != 32768.0 or paddle_value != 32768.0:
        raise RuntimeError("CUDA numerical preflight failed")
    report = {
        "schema_version": "exp716_runtime_preflight_v1",
        "experiment_id": "716",
        "decision": "RUNTIME_PREFLIGHT_PASS",
        "visible_cuda_devices": 1,
        "torch_cuda": torch.version.cuda,
        "paddle_cuda": paddle.version.cuda(),
        "versions": versions,
        "torch_gpu_matmul_sum": torch_value,
        "paddle_gpu_matmul_sum": paddle_value,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

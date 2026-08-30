from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


EXPECTED = {
    "torch": "2.9.0",
    "transformers": "",
    "peft": "",
    "pandas": "",
    "pillow": "",
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

    import torch
    from peft import PeftModel  # noqa: F401
    from transformers import AutoModelForMultimodalLM, AutoProcessor  # noqa: F401

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("preflight requires exactly one visible CUDA GPU for torch")
    torch_value = float((torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")).sum().cpu())
    if torch_value != 32768.0:
        raise RuntimeError("CUDA numerical preflight failed")
    report = {
        "schema_version": "exp718_runtime_preflight_v1",
        "experiment_id": "718",
        "architecture": "qwen35_only",
        "decision": "RUNTIME_PREFLIGHT_PASS",
        "visible_cuda_devices": 1,
        "torch_cuda": torch.version.cuda,
        "versions": versions,
        "torch_gpu_matmul_sum": torch_value,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

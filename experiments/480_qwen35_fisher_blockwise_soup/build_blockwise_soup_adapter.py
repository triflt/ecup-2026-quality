from __future__ import annotations

import argparse
import json
import math
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from safetensors.numpy import load_file, save_file

MAX_UNCOMPRESSED_BYTES = 80 * 1024 * 1024


def materialize(source: Path, destination: Path) -> Path:
    if source.is_dir():
        return source
    with zipfile.ZipFile(source) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in adapter ZIP: {bad}")
        total = 0
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            file_type = (info.external_attr >> 16) & 0o170000
            if name.is_absolute() or ".." in name.parts or file_type == 0o120000:
                raise ValueError(f"unsafe adapter ZIP member: {info.filename}")
            total += info.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError(f"adapter ZIP expands above safe limit: {total}")
        archive.extractall(destination)
    return destination


def block_name(tensor_name: str) -> str:
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    raise ValueError(f"unsupported adapter tensor: {tensor_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-original", type=Path, required=True)
    parser.add_argument("--adapter-specialist", type=Path, required=True)
    parser.add_argument("--block-alphas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    alpha_report = json.loads(args.block_alphas.read_text())
    alphas = {str(key): float(value) for key, value in alpha_report["block_alphas"].items()}
    if not alphas or any(not 0.0 <= value <= 1.0 for value in alphas.values()):
        raise ValueError("invalid block alpha mapping")

    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        original = materialize(args.adapter_original, Path(first))
        specialist = materialize(args.adapter_specialist, Path(second))
        config_original = json.loads((original / "adapter_config.json").read_text())
        config_specialist = json.loads((specialist / "adapter_config.json").read_text())
        comparable = [
            "base_model_name_or_path",
            "bias",
            "lora_alpha",
            "r",
            "target_modules",
            "task_type",
            "use_dora",
            "use_rslora",
        ]
        for key in comparable:
            left, right = config_original.get(key), config_specialist.get(key)
            if key == "target_modules":
                left, right = sorted(left), sorted(right)
            if left != right:
                raise ValueError(f"adapter config mismatch for {key}: {left!r} != {right!r}")
        state_original = load_file(original / "adapter_model.safetensors")
        state_specialist = load_file(specialist / "adapter_model.safetensors")
        if set(state_original) != set(state_specialist):
            raise ValueError("adapter tensor keys differ")

        old_rank = int(config_original["r"])
        old_scale = float(config_original["lora_alpha"]) / (
            math.sqrt(old_rank) if config_original.get("use_rslora") else old_rank
        )
        merged: dict[str, np.ndarray] = {}
        used_blocks = set()
        for key in sorted(state_original):
            block = block_name(key)
            if block not in alphas:
                raise ValueError(f"missing alpha for block: {block}")
            alpha = alphas[block]
            used_blocks.add(block)
            left, right = state_original[key], state_specialist[key]
            if ".lora_A." in key:
                merged[key] = np.concatenate([left, right], axis=0)
            else:
                merged[key] = np.concatenate(
                    [left * old_scale * (1.0 - alpha), right * old_scale * alpha],
                    axis=1,
                )
        if used_blocks != set(alphas):
            raise ValueError("alpha report contains unknown blocks")

        config = dict(config_original)
        config["r"] = old_rank * 2
        config["lora_alpha"] = old_rank * 2
        config["use_rslora"] = False
        config["rank_pattern"] = {}
        config["alpha_pattern"] = {}
        args.output.mkdir(parents=True)
        save_file(merged, args.output / "adapter_model.safetensors")
        (args.output / "adapter_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report = {
            "experiment_id": "480",
            "method": "blockwise Fisher-weighted exact effective-LoRA concatenation",
            "old_rank": old_rank,
            "new_rank": old_rank * 2,
            "old_scale": old_scale,
            "new_scale": 1.0,
            "blocks": len(used_blocks),
            "alpha_min": min(alphas.values()),
            "alpha_median": float(np.median(list(alphas.values()))),
            "alpha_max": max(alphas.values()),
        }
        (args.output / "merge_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

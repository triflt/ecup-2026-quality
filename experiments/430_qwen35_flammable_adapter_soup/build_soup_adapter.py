from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from safetensors.numpy import load_file, save_file


MAX_UNCOMPRESSED_BYTES = 80 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-190", type=Path, required=True)
    parser.add_argument("--adapter-260", type=Path, required=True)
    parser.add_argument("--alpha-260", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha_260 <= 1.0:
        raise ValueError("--alpha-260 must be in [0, 1]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        adapter_190 = materialize(args.adapter_190, Path(first))
        adapter_260 = materialize(args.adapter_260, Path(second))
        config_190 = json.loads((adapter_190 / "adapter_config.json").read_text())
        config_260 = json.loads((adapter_260 / "adapter_config.json").read_text())
        comparable = [
            "base_model_name_or_path", "bias", "lora_alpha", "r",
            "target_modules", "task_type", "use_dora", "use_rslora",
        ]
        for key in comparable:
            left, right = config_190.get(key), config_260.get(key)
            if key == "target_modules":
                left, right = sorted(left), sorted(right)
            if left != right:
                raise ValueError(f"adapter config mismatch for {key}: {left!r} != {right!r}")
        state_190 = load_file(adapter_190 / "adapter_model.safetensors")
        state_260 = load_file(adapter_260 / "adapter_model.safetensors")
        if set(state_190) != set(state_260):
            raise ValueError("adapter tensor keys differ")

        old_rank = int(config_190["r"])
        old_scale = float(config_190["lora_alpha"]) / (
            math.sqrt(old_rank) if config_190.get("use_rslora") else old_rank
        )
        alpha = args.alpha_260
        merged: dict[str, np.ndarray] = {}
        max_factor_error = 0.0
        for key in sorted(state_190):
            left, right = state_190[key], state_260[key]
            if ".lora_A." in key:
                merged[key] = np.concatenate([left, right], axis=0)
                max_factor_error = max(
                    max_factor_error,
                    float(np.max(np.abs(merged[key][:old_rank] - left))),
                    float(np.max(np.abs(merged[key][old_rank:] - right))),
                )
            elif ".lora_B." in key:
                merged[key] = np.concatenate(
                    [left * (old_scale * (1.0 - alpha)), right * (old_scale * alpha)],
                    axis=1,
                )
            else:
                raise ValueError(f"unsupported adapter tensor: {key}")

        config = dict(config_190)
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
            "experiment_id": "430",
            "method": "exact rank-32 concatenation of weighted effective LoRA deltas",
            "alpha_190": 1.0 - alpha,
            "alpha_260": alpha,
            "adapter_190": str(args.adapter_190),
            "adapter_260": str(args.adapter_260),
            "adapter_190_sha256": sha256(args.adapter_190),
            "adapter_260_sha256": sha256(args.adapter_260),
            "old_rank": old_rank,
            "new_rank": old_rank * 2,
            "old_scale": old_scale,
            "new_scale": 1.0,
            "tensor_count": len(merged),
            "max_factor_reconstruction_error": max_factor_error,
        }
        (args.output / "merge_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

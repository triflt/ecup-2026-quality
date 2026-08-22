from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    if args.alpha_260 != 0.5:
        raise ValueError("experiment 470 freezes alpha at 0.5")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        left_dir = materialize(args.adapter_190, Path(first))
        right_dir = materialize(args.adapter_260, Path(second))
        left_config = json.loads((left_dir / "adapter_config.json").read_text())
        right_config = json.loads((right_dir / "adapter_config.json").read_text())
        comparable = [
            "base_model_name_or_path", "bias", "lora_alpha", "r",
            "target_modules", "task_type", "use_dora", "use_rslora",
        ]
        for key in comparable:
            left, right = left_config.get(key), right_config.get(key)
            if key == "target_modules":
                left, right = sorted(left), sorted(right)
            if left != right:
                raise ValueError(f"adapter config mismatch for {key}: {left!r} != {right!r}")
        left_state = load_file(left_dir / "adapter_model.safetensors")
        right_state = load_file(right_dir / "adapter_model.safetensors")
        if set(left_state) != set(right_state):
            raise ValueError("adapter tensor keys differ")

        old_rank = int(left_config["r"])
        old_scale = float(left_config["lora_alpha"]) / (
            math.sqrt(old_rank) if left_config.get("use_rslora") else old_rank
        )
        merged: dict[str, np.ndarray] = {}
        for key in sorted(left_state):
            left, right = left_state[key], right_state[key]
            if ".lora_A." in key:
                merged[key] = np.concatenate([left, right], axis=0)
            elif ".lora_B." in key:
                merged[key] = np.concatenate(
                    [left * old_scale * 0.5, right * old_scale * 0.5], axis=1
                )
            else:
                raise ValueError(f"unsupported adapter tensor: {key}")

        config = dict(left_config)
        config.update(r=old_rank * 2, lora_alpha=old_rank * 2, use_rslora=False)
        config["rank_pattern"] = {}
        config["alpha_pattern"] = {}
        args.output.mkdir(parents=True)
        save_file(merged, args.output / "adapter_model.safetensors")
        (args.output / "adapter_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report = {
            "experiment_id": "470",
            "method": "exact fixed-alpha rank-32 LoRA interpolation",
            "alpha_original": 0.5,
            "alpha_specialist": 0.5,
            "original_sha256": sha256(args.adapter_190),
            "specialist_sha256": sha256(args.adapter_260),
            "old_rank": old_rank,
            "new_rank": old_rank * 2,
            "tensor_count": len(merged),
        }
        (args.output / "merge_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

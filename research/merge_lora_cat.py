from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def materialize(source: Path, destination: Path) -> Path:
    if source.is_dir():
        return source
    with zipfile.ZipFile(source) as archive:
        archive.extractall(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-a", type=Path, required=True)
    parser.add_argument("--adapter-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-a", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.weight_a <= 1.0:
        raise ValueError("--weight-a must be in [0, 1]")
    weight_b = 1.0 - args.weight_a

    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        adapter_a = materialize(args.adapter_a, Path(first))
        adapter_b = materialize(args.adapter_b, Path(second))
        config_a = json.loads((adapter_a / "adapter_config.json").read_text())
        config_b = json.loads((adapter_b / "adapter_config.json").read_text())
        comparable = [
            "base_model_name_or_path", "bias", "lora_alpha", "r", "target_modules",
            "task_type", "use_dora", "use_rslora",
        ]
        for key in comparable:
            left, right = config_a.get(key), config_b.get(key)
            if key == "target_modules":
                left, right = sorted(left), sorted(right)
            if left != right:
                raise ValueError(f"adapter config mismatch for {key}: {left!r} != {right!r}")
        state_a = load_file(adapter_a / "adapter_model.safetensors")
        state_b = load_file(adapter_b / "adapter_model.safetensors")
        if set(state_a) != set(state_b):
            raise ValueError("adapter tensor keys differ")

        old_rank = int(config_a["r"])
        old_scale = float(config_a["lora_alpha"]) / (
            math.sqrt(old_rank) if config_a.get("use_rslora") else old_rank
        )
        new_rank = old_rank * 2
        # Standard LoRA scaling alpha/r with alpha=new_rank gives scale 1.
        # Put the original scaling and ensemble weights directly into B.
        merged = {}
        checked = 0
        max_factor_error = 0.0
        for key in sorted(state_a):
            left = state_a[key]
            right = state_b[key]
            if ".lora_A." in key:
                merged[key] = np.concatenate([left, right], axis=0)
                max_factor_error = max(
                    max_factor_error,
                    float(np.max(np.abs(merged[key][:old_rank] - left))),
                    float(np.max(np.abs(merged[key][old_rank:] - right))),
                )
            elif ".lora_B." in key:
                merged[key] = np.concatenate(
                    [left * (old_scale * args.weight_a), right * (old_scale * weight_b)],
                    axis=1,
                )
                max_factor_error = max(
                    max_factor_error,
                    float(
                        np.max(
                            np.abs(
                                merged[key][:, :old_rank]
                                - left * (old_scale * args.weight_a)
                            )
                        )
                    ),
                    float(
                        np.max(
                            np.abs(
                                merged[key][:, old_rank:]
                                - right * (old_scale * weight_b)
                            )
                        )
                    ),
                )
            else:
                raise ValueError(f"unsupported adapter tensor: {key}")
            checked += 1

        config = dict(config_a)
        config["r"] = new_rank
        config["lora_alpha"] = new_rank
        config["use_rslora"] = False
        config["rank_pattern"] = {}
        config["alpha_pattern"] = {}
        args.output.mkdir(parents=True, exist_ok=False)
        save_file(merged, args.output / "adapter_model.safetensors")
        (args.output / "adapter_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        source_readme = adapter_a / "README.md"
        if source_readme.is_file():
            shutil.copy2(source_readme, args.output / "README.md")
        report = {
            "method": "exact rank-concatenation of LoRA deltas",
            "adapter_a": str(args.adapter_a),
            "adapter_b": str(args.adapter_b),
            "weight_a": args.weight_a,
            "weight_b": weight_b,
            "old_rank": old_rank,
            "new_rank": new_rank,
            "old_scale": old_scale,
            "new_scale": 1.0,
            "tensor_count": checked,
            "max_factor_reconstruction_error": max_factor_error,
        }
        (args.output / "merge_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

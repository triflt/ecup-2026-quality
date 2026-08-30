from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent_builder() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "656_paddleocr_tiled_repair"
        / "build_private_presets.py"
    )
    return load_module(path, "exp656_presets")


def render(
    *,
    parent: ModuleType,
    base_text: str,
    manifest_artifact: str,
    shard_artifacts: dict[int, str],
    repair_shard: int,
    num_repair_shards: int,
    smoke: bool,
    time_limit: str,
) -> str:
    rendered = parent.render(
        base_text=base_text,
        manifest_artifact=manifest_artifact,
        shard_artifacts=shard_artifacts,
        repair_shard=repair_shard,
        num_repair_shards=num_repair_shards,
        smoke=smoke,
        time_limit=time_limit,
    )
    rendered = rendered.replace(
        "python -u /work/code/656/run_tiled_repair.py ",
        "python -u /work/code/657/run_recursive_repair.py ",
    )
    rendered = rendered.replace(
        "--spotting-module /work/code/633/run_spotting.py ",
        "--spotting-module /work/code/633/run_spotting.py "
        "--tiled-module /work/code/656/run_tiled_repair.py ",
    )
    anchor = "  input:\n"
    runner = (
        "    - {type: files, "
        "src: experiments/657_paddleocr_recursive_tiled_repair/run_recursive_repair.py, "
        "dst: /work/code/657/run_recursive_repair.py}\n"
    )
    if anchor not in rendered:
        raise ValueError("parent preset has no input section")
    rendered = rendered.replace(anchor, anchor + runner, 1)
    rendered = rendered.replace("ocr_tiled_smoke", "ocr_refine_smoke")
    rendered = rendered.replace("ocr_tiled_s", "ocr_refine_s")
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Build private presets for recursive OCR repair.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source-artifacts-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--num-repair-shards", type=int, default=32)
    parser.add_argument("--repair-shard-index", type=int)
    parser.add_argument("--smoke-time-limit", default="4h0m0s")
    parser.add_argument("--production-time-limit", default="16h0m0s")
    args = parser.parse_args()
    parent = parent_builder()
    for path, name in (
        (args.base, "base preset"),
        (args.source_artifacts_manifest, "artifact manifest"),
        (args.output_dir, "output directory"),
    ):
        if not parent._is_ignored_local_path(path):
            raise ValueError(f"{name} must remain under an ignored .local/compute directory")
    if not 1 <= args.num_repair_shards <= 100_000:
        raise ValueError("num repair shards must be in 1..100000")
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite a preset directory")
    manifest_artifact, shard_artifacts = parent.source_artifacts(args.source_artifacts_manifest)
    base_text = args.base.read_text(encoding="utf-8")
    if args.mode == "smoke":
        if args.repair_shard_index not in {None, 0}:
            raise ValueError("smoke has exactly one shard")
        grid = [(0, 1)]
        time_limit = args.smoke_time_limit
    else:
        if args.repair_shard_index is not None and not (
            0 <= args.repair_shard_index < args.num_repair_shards
        ):
            raise ValueError("repair shard index must be inside the production grid")
        indices = (
            [args.repair_shard_index]
            if args.repair_shard_index is not None
            else range(args.num_repair_shards)
        )
        grid = [(index, args.num_repair_shards) for index in indices]
        time_limit = args.production_time_limit
    args.output_dir.mkdir(parents=True)
    for repair_shard, num_repair_shards in grid:
        filename = "smoke.yml" if args.mode == "smoke" else f"shard{repair_shard:02d}.yml"
        (args.output_dir / filename).write_text(
            render(
                parent=parent,
                base_text=base_text,
                manifest_artifact=manifest_artifact,
                shard_artifacts=shard_artifacts,
                repair_shard=repair_shard,
                num_repair_shards=num_repair_shards,
                smoke=args.mode == "smoke",
                time_limit=time_limit,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

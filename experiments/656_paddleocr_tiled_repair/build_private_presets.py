from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXPECTED_SOURCE_SHARDS = 32
BOUNDED_SMOKE_ROWS = 8
SAFE_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]*")
SAFE_TIME_LIMIT = re.compile(r"[0-9]+h[0-9]+m[0-9]+s")


def _is_safe_value(value: str) -> bool:
    return SAFE_VALUE.fullmatch(value) is not None and "://" not in value


def scalar(text: str, key: str, *, indent: int = 2) -> str:
    match = re.search(rf"^{' ' * indent}{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing preset key: {key}")
    value = match.group(1)
    if not _is_safe_value(value):
        raise ValueError(f"unsafe preset value for {key}")
    return value


def model_registry_mrid(text: str) -> str:
    match = re.search(r"type:\s*model_registry,\s*mrid:\s*([^,}]+)", text)
    if match is None:
        raise ValueError("base preset has no model_registry input")
    value = match.group(1).strip()
    if not _is_safe_value(value):
        raise ValueError("unsafe model registry identifier")
    return value


def source_artifacts(path: Path) -> tuple[str, dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("source artifact manifest must be an object")
    if set(payload) != {"manifest_artifact", "records"}:
        raise ValueError("source artifact manifest schema mismatch")
    manifest_artifact = payload["manifest_artifact"]
    records = payload["records"]
    if not isinstance(manifest_artifact, str) or not _is_safe_value(manifest_artifact):
        raise ValueError("unsafe manifest artifact reference")
    if not isinstance(records, list):
        raise TypeError("source artifact manifest records must be a list")
    result: dict[int, str] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"shard", "artifact"}:
            raise ValueError("source artifact record schema mismatch")
        shard = record["shard"]
        artifact = record["artifact"]
        if type(shard) is not int or not 0 <= shard < EXPECTED_SOURCE_SHARDS:
            raise ValueError("source shard outside 0..31")
        if not isinstance(artifact, str) or not _is_safe_value(artifact):
            raise ValueError("unsafe source artifact reference")
        if shard in result:
            raise ValueError(f"duplicate source shard: {shard}")
        result[shard] = artifact
    if set(result) != set(range(EXPECTED_SOURCE_SHARDS)):
        raise ValueError("source artifacts must cover exactly shards 0..31")
    return manifest_artifact, result


def render(
    *,
    base_text: str,
    manifest_artifact: str,
    shard_artifacts: dict[int, str],
    repair_shard: int,
    num_repair_shards: int,
    smoke: bool,
    time_limit: str,
) -> str:
    if set(shard_artifacts) != set(range(EXPECTED_SOURCE_SHARDS)):
        raise ValueError("source artifacts must cover exactly shards 0..31")
    if not 0 <= repair_shard < num_repair_shards:
        raise ValueError("repair shard outside grid")
    if not _is_safe_value(manifest_artifact):
        raise ValueError("unsafe manifest artifact reference")
    if any(not _is_safe_value(value) for value in shard_artifacts.values()):
        raise ValueError("unsafe source artifact reference")
    if SAFE_TIME_LIMIT.fullmatch(time_limit) is None:
        raise ValueError("unsafe time limit")
    if smoke and (repair_shard != 0 or num_repair_shards != 1):
        raise ValueError("smoke must use the global candidate stream")

    values = {
        "flavor": scalar(base_text, "flavor"),
        "region": scalar(base_text, "region"),
        "image": scalar(base_text, "image"),
        "preemption": scalar(base_text, "preemption"),
        "model_mrid": model_registry_mrid(base_text),
    }
    source_inputs = "\n".join(
        (
            "    - type: artifact\n"
            f"      src: {shard_artifacts[shard]}\n"
            f"      dst: /work/source/shard{shard:02d}"
        )
        for shard in range(EXPECTED_SOURCE_SHARDS)
    )
    shard_args = " ".join(
        f"--shard-dir /work/source/shard{shard:02d}"
        for shard in range(EXPECTED_SOURCE_SHARDS)
    )
    smoke_arg = f" --candidate-limit {BOUNDED_SMOKE_ROWS}" if smoke else ""
    # File output names are limited to 20 characters by the execution platform.
    output_name = "ocr_tiled_smoke" if smoke else f"ocr_tiled_s{repair_shard:02d}"
    command = (
        "mkdir -p /work/output /work/images && "
        "python -u /work/code/656/run_tiled_repair.py "
        "--manifest /work/manifest/all_images.jsonl "
        f"{shard_args} "
        "--builder-module /work/code/634/build_dataset.py "
        "--spotting-module /work/code/633/run_spotting.py "
        "--model-root /hf_models --output-dir /work/output --image-cache /work/images "
        f"--repair-shard-index {repair_shard} --num-repair-shards {num_repair_shards}"
        f"{smoke_arg}"
    )
    return f"""job:
  generate_name: paddleocr
  time_limit: {time_limit}
  flavor: {values['flavor']}
  region: {values['region']}
  image: {values['image']}
  preemption: {values['preemption']}
  work_dir: /work
  env:
    TOKENIZERS_PARALLELISM: "false"
  entrypoint: /bin/bash
  args:
    - -lc
    - >-
      {command}
  input:
    - {{type: files, src: experiments/656_paddleocr_tiled_repair/run_tiled_repair.py, dst: /work/code/656/run_tiled_repair.py}}
    - {{type: files, src: experiments/634_immutable_ocr_dataset/build_dataset.py, dst: /work/code/634/build_dataset.py}}
    - {{type: files, src: experiments/633_paddleocr_vl16_spotting/run_spotting.py, dst: /work/code/633/run_spotting.py}}
    - {{type: model_registry, mrid: {values['model_mrid']}, dst: /hf_models/}}
    - type: artifact
      src: {manifest_artifact}
      dst: /work/manifest
{source_inputs}
  output:
    - {{type: files, name: {output_name}, src: /work/output, mask: "**/*"}}
"""


def _is_ignored_local_path(path: Path) -> bool:
    parts = path.resolve().parts
    return any(parts[index : index + 2] == (".local", "compute") for index in range(len(parts) - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build private native-artifact presets for 656.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source-artifacts-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--num-repair-shards", type=int, default=32)
    parser.add_argument("--repair-shard-index", type=int)
    parser.add_argument("--smoke-time-limit", default="4h0m0s")
    parser.add_argument("--production-time-limit", default="16h0m0s")
    args = parser.parse_args()

    if not _is_ignored_local_path(args.base):
        raise ValueError("base preset must remain under an ignored .local/compute directory")
    if not _is_ignored_local_path(args.source_artifacts_manifest):
        raise ValueError("artifact references must remain under an ignored .local/compute directory")
    if not _is_ignored_local_path(args.output_dir):
        raise ValueError("generated presets must remain under an ignored .local/compute directory")
    if not 1 <= args.num_repair_shards <= 100_000:
        raise ValueError("num repair shards must be in 1..100000")
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite a preset directory")

    base_text = args.base.read_text(encoding="utf-8")
    manifest_artifact, shard_artifacts = source_artifacts(args.source_artifacts_manifest)
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

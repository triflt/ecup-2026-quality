from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXPECTED_SOURCE_SHARDS = 32
SAFE_JOB_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def scalar(text: str, key: str, *, indent: int = 2) -> str:
    match = re.search(rf"^{' ' * indent}{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing preset key: {key}")
    return match.group(1)


def model_registry_mrid(text: str) -> str:
    match = re.search(r"type:\s*model_registry,\s*mrid:\s*([^,}]+)", text)
    if match is None:
        raise ValueError("base preset has no model_registry input")
    value = match.group(1).strip()
    if any(char.isspace() for char in value):
        raise ValueError("unsafe model registry identifier")
    return value


def source_jobs(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise TypeError("source jobs manifest must contain a records list")
    result: dict[int, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("source jobs manifest records must be objects")
        shard = record.get("shard")
        job_name = record.get("job_name")
        if type(shard) is not int or not 0 <= shard < EXPECTED_SOURCE_SHARDS:
            raise ValueError("source shard outside 0..31")
        if not isinstance(job_name, str) or SAFE_JOB_NAME.fullmatch(job_name) is None:
            raise ValueError("unsafe or missing source job name")
        if shard in result:
            raise ValueError(f"duplicate source shard: {shard}")
        result[shard] = job_name
    if set(result) != set(range(EXPECTED_SOURCE_SHARDS)):
        raise ValueError("source jobs manifest must cover exactly shards 0..31")
    return result


def render(
    *,
    base_text: str,
    jobs: dict[int, str],
    repair_shard: int,
    num_repair_shards: int,
    max_new_tokens: int,
    time_limit: str = "16h0m0s",
) -> str:
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
            f"      src: {jobs[shard]}/paddleocr_full_s{shard:02d}\n"
            f"      dst: /work/shards/shard{shard:02d}"
        )
        for shard in range(EXPECTED_SOURCE_SHARDS)
    )
    shard_args = " ".join(
        f"--shard-dir /work/shards/shard{shard:02d}"
        for shard in range(EXPECTED_SOURCE_SHARDS)
    )
    command = (
        "mkdir -p /work/input /work/repo/experiments/633_paddleocr_vl16_spotting "
        "/work/repo/experiments/655_paddleocr_truncation_repair /work/images /work/output && "
        "python -c 'import os, urllib.request; "
        "urllib.request.urlretrieve(os.environ[\"RUNTIME_URL\"], "
        "\"/work/input/runtime.tar.gz\")' && "
        "tar -xzf /work/input/runtime.tar.gz -C /work/input && "
        "python -m pip install -q --break-system-packages --upgrade 'transformers>=5,<6' && "
        "python -u /work/repo/experiments/655_paddleocr_truncation_repair/run_repair.py "
        "--manifest /work/input/experiments/633_paddleocr_vl16_spotting/.local/all_images.jsonl "
        f"{shard_args} "
        "--builder-module /work/input/experiments/634_immutable_ocr_dataset/build_dataset.py "
        "--spotting-module /work/repo/experiments/633_paddleocr_vl16_spotting/run_spotting.py "
        "--model-root /hf_models --output-dir /work/output --image-cache /work/images "
        f"--repair-shard-index {repair_shard} --num-repair-shards {num_repair_shards} "
        f"--max-new-tokens {max_new_tokens}"
    )
    return f"""job:
  generate_name: paddleocr-repair-s{repair_shard:02d}
  time_limit: {time_limit}
  flavor: {values['flavor']}
  region: {values['region']}
  image: {values['image']}
  preemption: {values['preemption']}
  work_dir: /work
  env:
    TOKENIZERS_PARALLELISM: "false"
    RUNTIME_URL: ${{RUNTIME_URL}}
  entrypoint: /bin/bash
  args:
    - -lc
    - >-
      {command}
  input:
    - {{type: files, src: experiments/655_paddleocr_truncation_repair/run_repair.py, dst: /work/repo/experiments/655_paddleocr_truncation_repair/run_repair.py}}
    - {{type: files, src: experiments/633_paddleocr_vl16_spotting/run_spotting.py, dst: /work/repo/experiments/633_paddleocr_vl16_spotting/run_spotting.py}}
    - {{type: model_registry, mrid: {values['model_mrid']}, dst: /hf_models/}}
{source_inputs}
  output:
    - {{type: files, name: paddleocr_repair_s{repair_shard:02d}, src: /work/output, mask: "**/*"}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build private OCR truncation repair presets.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--runtime-url-file", type=Path, required=True)
    parser.add_argument("--source-jobs-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-repair-shards", type=int, default=16)
    parser.add_argument(
        "--repair-shard-index",
        type=int,
        help="Render only one shard. Useful for a bounded technical smoke.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--time-limit", default="16h0m0s")
    args = parser.parse_args()
    if not 1 <= args.num_repair_shards <= 100_000:
        raise ValueError("num repair shards must be in 1..100000")
    if args.repair_shard_index is not None and not (
        0 <= args.repair_shard_index < args.num_repair_shards
    ):
        raise ValueError("repair shard index must be inside the declared grid")
    if SAFE_JOB_NAME.fullmatch(args.time_limit) is None:
        raise ValueError("unsafe time limit")
    if args.max_new_tokens <= 512:
        raise ValueError("repair generation limit must exceed 512")
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite a preset directory")
    runtime_url = args.runtime_url_file.read_text(encoding="utf-8").strip()
    if not runtime_url.startswith("https://") or any(char.isspace() for char in runtime_url):
        raise ValueError("runtime URL file must contain one HTTPS URL")
    base_text = args.base.read_text(encoding="utf-8")
    jobs = source_jobs(args.source_jobs_manifest)
    args.output_dir.mkdir(parents=True)
    repair_shards = (
        [args.repair_shard_index]
        if args.repair_shard_index is not None
        else range(args.num_repair_shards)
    )
    for repair_shard in repair_shards:
        output = args.output_dir / f"shard{repair_shard:02d}.yml"
        output.write_text(
            render(
                base_text=base_text,
                jobs=jobs,
                repair_shard=repair_shard,
                num_repair_shards=args.num_repair_shards,
                max_new_tokens=args.max_new_tokens,
                time_limit=args.time_limit,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

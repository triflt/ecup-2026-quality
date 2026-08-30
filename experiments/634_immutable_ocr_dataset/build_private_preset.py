from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXPECTED_SHARDS = 32
SAFE_JOB_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def scalar(text: str, key: str, *, indent: int = 2) -> str:
    match = re.search(rf"^{' ' * indent}{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing preset key: {key}")
    return match.group(1)


def artifact_inputs(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise TypeError("source jobs manifest must contain a records list")
    by_shard: dict[int, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("source jobs manifest records must be objects")
        shard = record.get("shard")
        job_name = record.get("job_name")
        if type(shard) is not int or not 0 <= shard < EXPECTED_SHARDS:
            raise ValueError("source job shard is outside 0..31")
        if not isinstance(job_name, str) or SAFE_JOB_NAME.fullmatch(job_name) is None:
            raise ValueError("unsafe or missing source job name")
        if shard in by_shard:
            raise ValueError(f"duplicate source job shard: {shard}")
        by_shard[shard] = job_name
    if set(by_shard) != set(range(EXPECTED_SHARDS)):
        raise ValueError("source jobs manifest must cover exactly shards 0..31")
    return "\n".join(
        (
            "    - type: artifact\n"
            f"      src: {by_shard[shard]}/paddleocr_full_s{shard:02d}\n"
            f"      dst: /work/shards/shard{shard:02d}"
        )
        for shard in range(EXPECTED_SHARDS)
    )


def repair_artifact_inputs(path: Path) -> tuple[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise TypeError("repair jobs manifest must contain a nonempty records list")
    by_shard: dict[int, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("repair jobs manifest records must be objects")
        shard = record.get("repair_shard")
        job_name = record.get("job_name")
        if type(shard) is not int or shard < 0:
            raise ValueError("repair shard must be a nonnegative integer")
        if not isinstance(job_name, str) or SAFE_JOB_NAME.fullmatch(job_name) is None:
            raise ValueError("unsafe or missing repair job name")
        if shard in by_shard:
            raise ValueError(f"duplicate repair shard: {shard}")
        by_shard[shard] = job_name
    if set(by_shard) != set(range(len(by_shard))):
        raise ValueError("repair jobs manifest must cover contiguous shards from zero")
    return (
        "\n".join(
            (
                "    - type: artifact\n"
                f"      src: {by_shard[shard]}/paddleocr_repair_s{shard:02d}\n"
                f"      dst: /work/repairs/shard{shard:02d}"
            )
            for shard in range(len(by_shard))
        ),
        len(by_shard),
    )


def build(args: argparse.Namespace) -> str:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing private preset")
    runtime_url = args.runtime_url_file.read_text(encoding="utf-8").strip()
    if not runtime_url.startswith("https://") or any(char.isspace() for char in runtime_url):
        raise ValueError("runtime URL file must contain one HTTPS URL")
    base = args.base.read_text(encoding="utf-8")
    values = {
        "flavor": scalar(base, "flavor"),
        "region": scalar(base, "region"),
        "image": scalar(base, "image"),
        "preemption": scalar(base, "preemption"),
        "work_dir": scalar(base, "work_dir"),
        "tokenizers": scalar(base, "TOKENIZERS_PARALLELISM", indent=4),
        "allocator": scalar(base, "PYTORCH_ALLOC_CONF", indent=4),
    }
    source_jobs_manifest = getattr(args, "source_jobs_manifest", None)
    if source_jobs_manifest is None:
        command = (
            "mkdir -p /work/input /work/output && "
            "python -c 'import os, urllib.request; "
            "urllib.request.urlretrieve(os.environ[\"RUNTIME_URL\"], "
            "\"/work/input/runtime.tar.gz\")' && "
            "tar -xzf /work/input/runtime.tar.gz -C /work/input && "
            "PYTHONPATH=/work/input/experiments/634_immutable_ocr_dataset "
            "python -u /work/input/experiments/634_immutable_ocr_dataset/build_remote.py "
            "--manifest /work/input/experiments/633_paddleocr_vl16_spotting/.local/all_images.jsonl "
            "--source-url-manifest /work/input/source_urls.json --output-dir /work/output"
        )
        input_block = ""
    else:
        repair_jobs_manifest = getattr(args, "repair_jobs_manifest", None)
        if repair_jobs_manifest is None:
            repair_input_block = ""
            repair_command = ""
        else:
            repair_input_block, num_repair_shards = repair_artifact_inputs(repair_jobs_manifest)
            repair_command = (
                f"repair_args=(); for repair_idx in $(seq -w 0 {num_repair_shards - 1}); do "
                "repair_dir=/work/repairs/shard${repair_idx}; "
                "test -f ${repair_dir}/spotting.jsonl && test -f ${repair_dir}/report.json && "
                "repair_args+=(--repair-dir ${repair_dir}); done && "
                f"test ${{#repair_args[@]}} -eq {num_repair_shards * 2} && "
            )
        command = (
            "mkdir -p /work/input /work/output && "
            "python -c 'import os, urllib.request; "
            "urllib.request.urlretrieve(os.environ[\"RUNTIME_URL\"], "
            "\"/work/input/runtime.tar.gz\")' && "
            "tar -xzf /work/input/runtime.tar.gz -C /work/input && "
            "shard_args=(); for shard_idx in $(seq -w 0 31); do "
            "shard_dir=/work/shards/shard${shard_idx}; "
            "test -f ${shard_dir}/spotting.jsonl && test -f ${shard_dir}/report.json && "
            "shard_args+=(--shard-dir ${shard_dir}); done && "
            "test ${#shard_args[@]} -eq 64 && "
            f"{repair_command}"
            "PYTHONPATH=/work/input/experiments/634_immutable_ocr_dataset "
            "python -u /work/input/experiments/634_immutable_ocr_dataset/build_dataset.py "
            "--manifest /work/input/experiments/633_paddleocr_vl16_spotting/.local/all_images.jsonl "
            "${shard_args[@]} ${repair_args[@]} --output-dir /work/output/dataset && "
            "python -u /work/input/experiments/634_immutable_ocr_dataset/build_dataset.py "
            "--verify-only --output-dir /work/output/dataset"
        )
        joined_inputs = artifact_inputs(source_jobs_manifest)
        if repair_input_block:
            joined_inputs += "\n" + repair_input_block
        input_block = f"\n  input:\n{joined_inputs}"
    return f"""job:
  generate_name: paddleocr
  time_limit: 1h0m0s
  flavor: {values['flavor']}
  region: {values['region']}
  image: {values['image']}
  preemption: {values['preemption']}
  work_dir: {values['work_dir']}
  env:
    TOKENIZERS_PARALLELISM: {values['tokenizers']}
    PYTORCH_ALLOC_CONF: {values['allocator']}
    RUNTIME_URL: ${{RUNTIME_URL}}
  entrypoint: /bin/bash
  args:
    - -lc
    - >-
      {command}
{input_block}
  output:
    - {{type: files, name: ocr_dataset_634, src: /work/output}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a private experiment-634 preset.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--runtime-url-file", type=Path, required=True)
    parser.add_argument("--source-jobs-manifest", type=Path)
    parser.add_argument("--repair-jobs-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import re
from pathlib import Path


def scalar(text: str, key: str, *, indent: int = 2) -> str:
    match = re.search(rf"^{' ' * indent}{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing preset key: {key}")
    return match.group(1)


def build(args: argparse.Namespace) -> str:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing private preset")
    if args.fold not in range(5):
        raise ValueError("fold must be 0..4")
    url = args.bundle_url_file.read_text(encoding="utf-8").strip()
    if not url.startswith("https://") or any(char.isspace() for char in url):
        raise ValueError("bundle URL file must contain one HTTPS URL")
    if not re.fullmatch(r"[0-9a-f]{64}", args.bundle_sha256):
        raise ValueError("bundle SHA-256 must be 64 lowercase hex characters")
    base = args.base_qwen36.read_text(encoding="utf-8")
    four_gpu = args.base_four_gpu.read_text(encoding="utf-8")
    model_inputs = [line.strip() for line in base.splitlines() if "type: model_registry" in line]
    if len(model_inputs) != 1:
        raise ValueError("Qwen3.6 base preset must contain exactly one model input")
    values = {
        "flavor": scalar(four_gpu, "flavor"),
        "region": scalar(base, "region"),
        "image": scalar(base, "image"),
        "preemption": scalar(base, "preemption"),
        "work_dir": scalar(base, "work_dir"),
        "tokenizers": scalar(base, "TOKENIZERS_PARALLELISM", indent=4),
        "allocator": scalar(base, "PYTORCH_ALLOC_CONF", indent=4),
    }
    runtime_name = f"fold{args.fold}_{args.mode}"
    command = (
        "python -m pip install -q --break-system-packages --upgrade "
        "'transformers>=5,<6' 'accelerate>=1.12,<2' && "
        "mkdir -p /work/input /work/vendor /work/images /work/output && "
        "python -c 'import os, urllib.request; "
        "urllib.request.urlretrieve(os.environ[\"BUNDLE_URL\"], \"/work/bundle.tar.gz\")' && "
        f"test \"$(sha256sum /work/bundle.tar.gz | cut -d' ' -f1)\" = \"{args.bundle_sha256}\" && "
        "tar -xzf /work/bundle.tar.gz -C /work/input && "
        "python -m zipfile -e /work/input/vendor/peft-0.20.0.zip /work/vendor && "
        "PYTHONPATH=/work/vendor python -u "
        "/work/input/experiments/662_qwen36_27b_outer_train_scoring/score_teacher.py "
        f"--runtime-dir /work/input/runtime/{runtime_name} "
        f"--adapter-dir /work/input/adapter/fold{args.fold} "
        "--images /work/images --model-root /hf_models --vendor /work/vendor "
        "--output-dir /work/output --expected-cuda-devices 4"
    )
    return f"""job:
  generate_name: teacher-score
  time_limit: 2h0m0s
  flavor: {values['flavor']}
  region: {values['region']}
  image: {values['image']}
  preemption: {values['preemption']}
  work_dir: {values['work_dir']}
  env:
    TOKENIZERS_PARALLELISM: {values['tokenizers']}
    PYTORCH_ALLOC_CONF: {values['allocator']}
    BUNDLE_URL: ${{BUNDLE_URL}}
  entrypoint: /bin/bash
  args:
    - -lc
    - >-
      {command}
  input:
    {model_inputs[0]}
  output:
    - {{type: files, name: teacher_f{args.fold}_{args.mode}, src: /work/output}}
"""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--base-qwen36", type=Path, required=True)
    result.add_argument("--base-four-gpu", type=Path, required=True)
    result.add_argument("--bundle-url-file", type=Path, required=True)
    result.add_argument("--bundle-sha256", required=True)
    result.add_argument("--fold", type=int, required=True)
    result.add_argument("--mode", choices=("smoke8", "full"), required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")

from __future__ import annotations

import argparse
import json
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
    if args.smoke_script.is_absolute() or args.peft_zip.is_absolute():
        raise ValueError("file inputs must be repository-relative for portable preset context")
    base = args.base_qwen36.read_text(encoding="utf-8")
    four_gpu = args.base_four_gpu.read_text(encoding="utf-8")
    model_inputs = [line.strip() for line in base.splitlines() if "type: model_registry" in line]
    if len(model_inputs) != 1:
        raise ValueError("Qwen3.6 base preset must contain exactly one model input")
    command = (
        "python -m pip install -q --break-system-packages --upgrade "
        "'transformers>=5,<6' 'accelerate>=1.12,<2' && "
        "mkdir -p /work/vendor /work/output && "
        "python -m zipfile -e /work/peft-0.20.0.zip /work/vendor && "
        "PYTHONPATH=/work/vendor python -u /work/technical_smoke.py "
        "--model-root /hf_models --vendor /work/vendor "
        "--output-dir /work/output --expected-cuda-devices 4"
    )
    values = {
        "flavor": scalar(four_gpu, "flavor"),
        "region": scalar(base, "region"),
        "image": scalar(base, "image"),
        "preemption": scalar(base, "preemption"),
        "work_dir": scalar(base, "work_dir"),
        "tokenizers": scalar(base, "TOKENIZERS_PARALLELISM", indent=4),
        "allocator": scalar(base, "PYTORCH_ALLOC_CONF", indent=4),
    }
    script_path = json.dumps(str(args.smoke_script))
    peft_path = json.dumps(str(args.peft_zip))
    return f"""job:
  generate_name: qwen-lora
  time_limit: 1h0m0s
  flavor: {values['flavor']}
  region: {values['region']}
  image: {values['image']}
  preemption: {values['preemption']}
  work_dir: {values['work_dir']}
  env:
    TOKENIZERS_PARALLELISM: {values['tokenizers']}
    PYTORCH_ALLOC_CONF: {values['allocator']}
  entrypoint: /bin/bash
  args:
    - -lc
    - >-
      {command}
  input:
    - {{type: files, src: {script_path}, dst: /work/technical_smoke.py}}
    - {{type: files, src: {peft_path}, dst: /work/peft-0.20.0.zip}}
    {model_inputs[0]}
  output:
    - {{type: files, name: qwen36_lora_smoke, src: /work/output}}
"""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--base-qwen36", type=Path, required=True)
    result.add_argument("--base-four-gpu", type=Path, required=True)
    result.add_argument("--smoke-script", type=Path, required=True)
    result.add_argument("--peft-zip", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    payload = build(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(payload, encoding="utf-8")

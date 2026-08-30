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
        raise FileExistsError("refusing to overwrite a private preset")
    if args.fold not in {1, 2, 4}:
        raise ValueError("only remaining folds 1/2/4 are allowed")
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if gate.get("decision") != "OPEN_REMAINING_CONTROL_FOLDS":
        raise ValueError("continuation gate is not open")
    url = args.bundle_url_file.read_text(encoding="utf-8").strip()
    if not url.startswith("https://") or any(character.isspace() for character in url):
        raise ValueError("bundle URL file must contain one HTTPS URL")
    base = args.base_preset.read_text(encoding="utf-8")
    model_inputs = [line.strip() for line in base.splitlines() if "type: model_registry" in line]
    if len(model_inputs) != 1:
        raise ValueError("base preset must contain exactly one model input")
    values = {
        "time_limit": scalar(base, "time_limit"),
        "flavor": scalar(base, "flavor"),
        "region": scalar(base, "region"),
        "image": scalar(base, "image"),
        "preemption": scalar(base, "preemption"),
        "work_dir": scalar(base, "work_dir"),
        "tokenizers": scalar(base, "TOKENIZERS_PARALLELISM", indent=4),
        "allocator": scalar(base, "PYTORCH_ALLOC_CONF", indent=4),
    }
    command = (
        "mkdir -p /work/input /work/vendor /work/images /work/output && "
        "python -c 'import os, urllib.request; urllib.request.urlretrieve(os.environ[\"BUNDLE_URL\"], \"/work/input/bundle.tar.gz\")' && "
        "tar -xzf /work/input/bundle.tar.gz -C /work/input && "
        "cp /work/patch/train_lora.py /work/input/experiments/645_qwen_scale_2x3_gate/train_lora.py && "
        "python /work/input/experiments/661_qwen35_4b_class_only_full_control/verify_gate.py "
        "--gate /work/input/experiments/661_qwen35_4b_class_only_full_control/results/continuation_gate.json "
        f"--fold {args.fold} && "
        "python -m zipfile -e /work/input/research/peft-vendor-extracted/peft-0.20.0.zip /work/vendor && "
        "PYTHONPATH=/work/input/experiments/645_qwen_scale_2x3_gate:/work/vendor "
        "python -u /work/input/experiments/641_qwen35_4b_class_only_lora/run_fold.py "
        f"--fold {args.fold} "
        f"--runtime-dir /work/input/experiments/641_qwen35_4b_class_only_lora/.local/runtime/fold{args.fold} "
        "--images /work/images --model-root /hf_models "
        "--model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a "
        "--vendor /work/vendor --output-dir /work/output "
        "--runtime-backend legacy_eager --micro-batch-size-override 2"
    )
    return f"""job:
  generate_name: qwen
  time_limit: {values['time_limit']}
  flavor: {values['flavor']}
  region: {values['region']}
  image: {values['image']}
  preemption: {values['preemption']}
  work_dir: {values['work_dir']}
  env:
    TOKENIZERS_PARALLELISM: {values['tokenizers']}
    PYTORCH_ALLOC_CONF: {values['allocator']}
    BUNDLE_URL: ${{BUNDLE_URL}}
  entrypoint: bash
  args:
    - -lc
    - >-
      {command}
  input:
    - {{type: files, src: experiments/645_qwen_scale_2x3_gate/train_lora.py, dst: /work/patch/train_lora.py}}
    {model_inputs[0]}
  output:
    - {{type: files, name: qwen_m2_f{args.fold}, src: /work/output/, mask: "**/*"}}
"""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--base-preset", type=Path, required=True)
    result.add_argument("--bundle-url-file", type=Path, required=True)
    result.add_argument("--gate", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")


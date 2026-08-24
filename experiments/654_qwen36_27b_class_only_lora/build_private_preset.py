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
    if args.fold not in {0, 3}:
        raise ValueError("only screen folds 0 and 3 are open")
    bundle = getattr(args, "bundle", None)
    bundle_url_file = getattr(args, "bundle_url_file", None)
    if (bundle is None) == (bundle_url_file is None):
        raise ValueError("choose exactly one bundle delivery mode")
    if bundle is not None and bundle.is_absolute():
        raise ValueError("bundle path must be repository-relative")
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
    if bundle_url_file is not None:
        bundle_url = bundle_url_file.read_text(encoding="utf-8").strip()
        if not bundle_url.startswith("https://") or any(char.isspace() for char in bundle_url):
            raise ValueError("bundle URL file must contain one HTTPS URL")
        bundle_delivery = (
            "python -c 'import os, urllib.request; "
            "urllib.request.urlretrieve(os.environ[\"BUNDLE_URL\"], "
            "\"/work/qwen36_class_bundle.tar.gz\")' && "
        )
        # Keep the signed URL out of the preset even though the generated file is
        # ignored. The submitter supplies it as an ephemeral CLI environment
        # override after the static preset has passed the privacy audit.
        bundle_env = "    BUNDLE_URL: ${BUNDLE_URL}\n"
        bundle_input = ""
    else:
        bundle_delivery = ""
        bundle_env = ""
        bundle_path = json.dumps(str(bundle))
        bundle_input = (
            f"    - {{type: files, src: {bundle_path}, "
            "dst: /work/qwen36_class_bundle.tar.gz}}\n"
        )
    command = (
        "python -m pip install -q --break-system-packages --upgrade "
        "'transformers>=5,<6' 'accelerate>=1.12,<2' && "
        "mkdir -p /work/input /work/vendor /work/images /work/output && "
        f"{bundle_delivery}"
        "tar -xzf /work/qwen36_class_bundle.tar.gz -C /work/input && "
        "python -m zipfile -e "
        "/work/input/experiments/653_qwen36_27b_lora_runtime_preflight/"
        ".local/vendor/peft-0.20.0.zip /work/vendor && "
        "PYTHONPATH=/work/vendor python -u "
        "/work/input/experiments/654_qwen36_27b_class_only_lora/train_fold.py "
        f"--fold {args.fold} "
        f"--runtime-dir /work/input/experiments/654_qwen36_27b_class_only_lora/.local/runtime/fold{args.fold} "
        "--images /work/images --model-root /hf_models --vendor /work/vendor "
        "--output-dir /work/output --expected-cuda-devices 4"
    )
    return f"""job:
  generate_name: qwen-train
  time_limit: 8h0m0s
  flavor: {values['flavor']}
  region: {values['region']}
  image: {values['image']}
  preemption: {values['preemption']}
  work_dir: {values['work_dir']}
  env:
    TOKENIZERS_PARALLELISM: {values['tokenizers']}
    PYTORCH_ALLOC_CONF: {values['allocator']}
{bundle_env.rstrip()}
  entrypoint: /bin/bash
  args:
    - -lc
    - >-
      {command}
  input:
{bundle_input.rstrip()}
    {model_inputs[0]}
  output:
    - {{type: files, name: qwen36_class_fold{args.fold}, src: /work/output}}
"""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--base-qwen36", type=Path, required=True)
    result.add_argument("--base-four-gpu", type=Path, required=True)
    delivery = result.add_mutually_exclusive_group(required=True)
    delivery.add_argument("--bundle", type=Path)
    delivery.add_argument("--bundle-url-file", type=Path)
    result.add_argument("--fold", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")

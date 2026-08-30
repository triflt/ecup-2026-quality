from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from contract import MODEL_REVISION


def scalar(text: str, key: str, indent: int = 2) -> str:
    match = re.search(rf"^{' ' * indent}{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing preset key: {key}")
    return match.group(1)


def build(args: argparse.Namespace) -> str:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite preset")
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    expected = "OPEN_TECHNICAL_SMOKE_ONLY" if args.technical_smoke else "OPEN_SCREEN"
    if gate.get("decision") != expected or args.fold not in gate.get("allowed_folds", []):
        raise ValueError("launch gate is closed")
    url = args.bundle_url_file.read_text(encoding="utf-8").strip()
    bundle_sha = args.bundle_sha256_file.read_text(encoding="utf-8").strip()
    model_input = args.model_input_line_file.read_text(encoding="utf-8").strip()
    if not url.startswith("https://") or any(char.isspace() for char in url):
        raise ValueError("bundle URL must be one HTTPS URL")
    if not re.fullmatch(r"[0-9a-f]{64}", bundle_sha):
        raise ValueError("invalid bundle SHA")
    if "type: model_registry" not in model_input or "/hf_models/" not in model_input:
        raise ValueError("model input line is invalid")
    base = args.base_preset.read_text(encoding="utf-8")
    values = {key: scalar(base, key) for key in ("time_limit", "flavor", "region", "image", "preemption", "work_dir")}
    tokenizers = scalar(base, "TOKENIZERS_PARALLELISM", 4)
    allocator = scalar(base, "PYTORCH_ALLOC_CONF", 4)
    smoke = " --technical-smoke" if args.technical_smoke else ""
    runtime = f"/work/input/experiments/683_gemma4_e4b_class_only_lora_screen/.local/runtime/fold{args.fold}"
    command = (
        "mkdir -p /work/input /work/vendor /work/images /work/output && "
        "python -c 'import os, urllib.request; urllib.request.urlretrieve(os.environ[\"BUNDLE_URL\"], \"/work/input/bundle.tar.gz\")' && "
        f"test \"$(sha256sum /work/input/bundle.tar.gz | cut -d' ' -f1)\" = \"{bundle_sha}\" && "
        "tar -xzf /work/input/bundle.tar.gz -C /work/input && "
        f"python /work/input/experiments/683_gemma4_e4b_class_only_lora_screen/verify_launch_gate.py --gate /work/input/experiments/683_gemma4_e4b_class_only_lora_screen/results/launch_gate.json --runtime-dir {runtime} --fold {args.fold}{smoke} && "
        "python -m zipfile -e /work/input/research/peft-vendor-extracted/peft-0.20.0.zip /work/vendor && "
        "PYTHONPATH=/work/input/experiments/683_gemma4_e4b_class_only_lora_screen:/work/input/experiments/645_qwen_scale_2x3_gate:/work/vendor python -u /work/input/experiments/683_gemma4_e4b_class_only_lora_screen/train_fold.py "
        f"--fold {args.fold} --runtime-dir {runtime} --images /work/images --model-root /hf_models --model-revision {MODEL_REVISION} --vendor /work/vendor --output-dir /work/output{smoke}"
    )
    output_name = "gemma_e4b_smk" if args.technical_smoke else f"gemma_e4b_f{args.fold}"
    return f'''job:
  generate_name: gemma
  time_limit: {values["time_limit"]}
  flavor: {values["flavor"]}
  region: {values["region"]}
  image: {values["image"]}
  preemption: {values["preemption"]}
  work_dir: {values["work_dir"]}
  env:
    TOKENIZERS_PARALLELISM: {tokenizers}
    PYTORCH_ALLOC_CONF: {allocator}
    BUNDLE_URL: {json.dumps(url)}
  entrypoint: bash
  args:
    - -lc
    - >-
      {command}
  input:
    - {model_input}
  output:
    - {{type: files, name: {output_name}, src: /work/output/, mask: "**/*"}}
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-preset", type=Path, required=True)
    parser.add_argument("--bundle-url-file", type=Path, required=True)
    parser.add_argument("--bundle-sha256-file", type=Path, required=True)
    parser.add_argument("--model-input-line-file", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args), encoding="utf-8")

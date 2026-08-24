from __future__ import annotations

import argparse
import re
from pathlib import Path


def scalar(text: str, key: str, *, indent: int = 2) -> str:
    match = re.search(rf"^{' ' * indent}{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing preset key: {key}")
    return match.group(1)


def build(args: argparse.Namespace) -> tuple[str, str]:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a private preset")
    if args.env_output.exists():
        raise FileExistsError("refusing to overwrite a private environment file")
    if not args.technical_smoke:
        raise ValueError("full inner wave remains closed until smoke artifact acceptance")
    if args.inner_fold not in {1, 2, 3, 4}:
        raise ValueError("inner fold must be 1/2/3/4")
    if args.inner_fold != 1:
        raise ValueError("pre-smoke gate authorizes only inner-fold-1 technical smoke")
    url = args.bundle_url_file.read_text(encoding="utf-8").strip()
    if not url.startswith("https://") or any(character.isspace() for character in url):
        raise ValueError("bundle URL file must contain one HTTPS URL")
    bundle_sha256 = args.bundle_sha256.strip().lower()
    if len(bundle_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in bundle_sha256
    ):
        raise ValueError("bundle SHA-256 must be 64 lowercase hex characters")
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
        f"test \"$(sha256sum /work/input/bundle.tar.gz | cut -d' ' -f1)\" = \"{bundle_sha256}\" && "
        "tar -xzf /work/input/bundle.tar.gz -C /work/input && "
        "python /work/input/experiments/677_qwen35_pr_auc_dynamics_screen/verify_launch_gate.py "
        "--gate /work/input/experiments/677_qwen35_pr_auc_dynamics_screen/results/launch_gate.json "
        "--parent-result /work/input/experiments/659_qwen36_27b_flammable_equal_logit_blend/results/full_acceptance_audit.json "
        f"--inner-fold {args.inner_fold} --technical-smoke && "
        "python -m zipfile -e /work/input/research/peft-vendor-extracted/peft-0.20.0.zip /work/vendor && "
        "PYTHONPATH=/work/input/experiments/645_qwen_scale_2x3_gate:/work/vendor "
        "python -u /work/input/experiments/677_qwen35_pr_auc_dynamics_screen/train_dynamics.py "
        f"--inner-fold {args.inner_fold} "
        f"--runtime-dir /work/input/experiments/677_qwen35_pr_auc_dynamics_screen/.local/inner_runtime_v2/inner_fold{args.inner_fold} "
        "--images /work/images --model-root /hf_models "
        "--model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a "
        "--vendor /work/vendor --output-dir /work/output"
        + (" --technical-smoke" if args.technical_smoke else "")
    )
    output_name = f"dyn_i{args.inner_fold}" + ("_smoke" if args.technical_smoke else "")
    preset = f"""job:
  generate_name: qwen-dynamics
  time_limit: {values['time_limit']}
  flavor: {values['flavor']}
  region: {values['region']}
  image: {values['image']}
  preemption: {values['preemption']}
  work_dir: {values['work_dir']}
  env:
    TOKENIZERS_PARALLELISM: {values['tokenizers']}
    PYTORCH_ALLOC_CONF: {values['allocator']}
    BUNDLE_URL: override_on_submit
  entrypoint: bash
  args:
    - -lc
    - >-
      {command}
  input:
    {model_inputs[0]}
  output:
    - {{type: files, name: {output_name}, src: /work/output/, mask: "**/*"}}
"""
    return preset, f"BUNDLE_URL={url}\n"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--base-preset", type=Path, required=True)
    result.add_argument("--bundle-url-file", type=Path, required=True)
    result.add_argument("--bundle-sha256", required=True)
    result.add_argument("--inner-fold", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--env-output", type=Path, required=True)
    result.add_argument("--technical-smoke", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    payload, env_payload = build(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.env_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(payload, encoding="utf-8")
    arguments.env_output.write_text(env_payload, encoding="utf-8")

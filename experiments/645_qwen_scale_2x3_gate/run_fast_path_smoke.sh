#!/usr/bin/env bash
set -euo pipefail

runtime_root="${1:?runtime root is required}"
output_dir="${2:?output directory is required}"
source "${runtime_root}/experiments/645_qwen_scale_2x3_gate/install_fast_path_dependencies.sh" \
  "${runtime_root}"
export PYTHONPATH="${runtime_root}/experiments/645_qwen_scale_2x3_gate"
python -u "${runtime_root}/experiments/645_qwen_scale_2x3_gate/fast_path_smoke.py" \
  --output-dir "${output_dir}"

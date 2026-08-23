#!/usr/bin/env bash
set -euo pipefail

qwen_runtime_root="${1:?runtime root is required}"
qwen_vendor_dir="${qwen_runtime_root}/experiments/645_qwen_scale_2x3_gate/.local/vendor"

python - "${qwen_vendor_dir}" <<'PY'
import hashlib
import sys
from pathlib import Path

vendor = Path(sys.argv[1])
expected = {
    "transformers-5.15.1-py3-none-any.whl": "b7cdf238ff583e3a58dbc7fa34da1aaf091ce063141f65a30538160bd5afe93f",
    "causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl": "c16c1c48d4fa63415cc797e02d69f97248c57c04627d99e394d5bb0ef266e288",
    "einops-0.8.2-py3-none-any.whl": "54058201ac7087911181bfec4af6091bb59380360f069276601256a76af08193",
    "fla_core-0.5.2-py3-none-any.whl": "5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761",
    "flash_linear_attention-0.5.2-py3-none-any.whl": "dcf405d81f5426393b59037097aa700d0f4a841465d5028d5aa543f4502f2400",
    "kernels-0.16.0-py3-none-any.whl": "794af6a10fd888bb4f46ad1b9b2f4f61b5b0b104475a6415c5322b58a7bf02ed",
    "kernels_data-0.16.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": "30efa0e6ee0261c6c5b581fed372321a79db923a3e1934d00b7675565baaffc3",
    "ziglang-0.16.0-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.musllinux_1_1_x86_64.whl": "9fcda73f62b851dd72a54b710ad40a209896db14cfb13649e62191243556342b",
}
for filename, expected_sha256 in expected.items():
    path = vendor / filename
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"vendored wheel checksum mismatch: {filename}")
PY

python -m pip install -q --break-system-packages --no-deps \
  "${qwen_vendor_dir}/transformers-5.15.1-py3-none-any.whl" \
  "${qwen_vendor_dir}/causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
  "${qwen_vendor_dir}/einops-0.8.2-py3-none-any.whl" \
  "${qwen_vendor_dir}/fla_core-0.5.2-py3-none-any.whl" \
  "${qwen_vendor_dir}/flash_linear_attention-0.5.2-py3-none-any.whl" \
  "${qwen_vendor_dir}/kernels-0.16.0-py3-none-any.whl" \
  "${qwen_vendor_dir}/kernels_data-0.16.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
  "${qwen_vendor_dir}/ziglang-0.16.0-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.musllinux_1_1_x86_64.whl"

export CC="${qwen_runtime_root}/experiments/645_qwen_scale_2x3_gate/zig_cc.sh"
export CXX="${qwen_runtime_root}/experiments/645_qwen_scale_2x3_gate/zig_cxx.sh"

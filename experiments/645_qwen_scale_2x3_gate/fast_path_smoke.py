from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from grid_contract import canonical_sha256
from train_lora import (
    verify_fast_linear_attention_dependencies,
    verify_qwen35_fast_path_binding,
)


def run_smoke(output_dir: Path) -> dict[str, object]:
    import torch
    from causal_conv1d import causal_conv1d_fn
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    started = time.monotonic()
    packages = verify_fast_linear_attention_dependencies()
    bindings = verify_qwen35_fast_path_binding()

    conv_x = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    conv_weight = torch.randn(8, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    conv_output = causal_conv1d_fn(conv_x, conv_weight, activation="silu")
    if conv_output.shape != conv_x.shape or not torch.isfinite(conv_output).all():
        raise RuntimeError("causal-conv1d functional smoke failed")
    conv_output.float().square().mean().backward()
    if conv_x.grad is None or not torch.isfinite(conv_x.grad).all():
        raise RuntimeError("causal-conv1d backward smoke failed")

    shape = (1, 64, 2, 16)
    q = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.nn.functional.normalize(
        torch.randn(*shape, device="cuda", dtype=torch.bfloat16), dim=-1
    ).requires_grad_()
    v = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = -torch.nn.functional.softplus(
        torch.randn(1, 64, 2, device="cuda", dtype=torch.float32)
    )
    beta = torch.sigmoid(torch.randn(1, 64, 2, device="cuda", dtype=torch.float32))
    fla_output, _state = chunk_gated_delta_rule(q, k, v, g, beta)
    if fla_output.shape != v.shape or not torch.isfinite(fla_output).all():
        raise RuntimeError("FLA functional smoke failed")
    fla_output.float().square().mean().backward()
    for name, tensor in {"q": q, "k": k, "v": v}.items():
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise RuntimeError(f"FLA backward smoke failed for {name}")

    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "645",
        "fast_path_packages": packages,
        "fast_path_bindings": bindings,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_capability": list(torch.cuda.get_device_capability()),
        "causal_conv1d_shape": list(conv_output.shape),
        "fla_shape": list(fla_output.shape),
        "forward_and_backward_finite": True,
        "runtime_seconds": time.monotonic() - started,
        "decision": "FAST_PATH_FUNCTIONAL_SMOKE_PASSED",
    }
    report["contract_sha256"] = canonical_sha256(report)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "fast_path_smoke.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a functional CUDA forward/backward smoke for the frozen Qwen fast path."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_smoke(args.output_dir)


if __name__ == "__main__":
    main()

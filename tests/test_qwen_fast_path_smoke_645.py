from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "experiments/645_qwen_scale_2x3_gate/fast_path_smoke.py"
RUNNER = ROOT / "experiments/645_qwen_scale_2x3_gate/run_fast_path_smoke.sh"
INSTALLER = ROOT / "experiments/645_qwen_scale_2x3_gate/install_fast_path_dependencies.sh"


def test_fast_path_smoke_is_functional_and_scheduler_independent() -> None:
    source = SMOKE.read_text(encoding="utf-8")
    ast.parse(source)
    assert "causal_conv1d_fn" in source
    assert "chunk_gated_delta_rule" in source
    assert ".backward()" in source
    assert "forward_and_backward_finite" in source
    assert "--output-dir" in source
    for forbidden in ("compute", "project", "region", "job_name", "endpoint"):
        assert forbidden not in source.lower()


def test_fast_path_runner_is_pinned_and_contains_no_platform_details() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "set -euo pipefail" in source
    assert "install_fast_path_dependencies.sh" in source
    assert "fast_path_smoke.py" in source
    for forbidden in ("compute", "project", "region", "job_name", "endpoint"):
        assert forbidden not in source.lower()


def test_fast_path_installer_is_pinned_and_contains_no_platform_details() -> None:
    source = INSTALLER.read_text(encoding="utf-8")
    assert "set -euo pipefail" in source
    assert "--no-deps" in source
    assert "export CC=python-zig" in source
    assert "9fcda73f62b851dd72a54b710ad40a209896db14cfb13649e62191243556342b" in source
    for forbidden in ("compute", "project", "region", "job_name", "endpoint"):
        assert forbidden not in source.lower()

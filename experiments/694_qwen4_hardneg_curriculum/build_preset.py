from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parents[1] / "693_qwen4_causal_distillation/build_preset.py"
spec = importlib.util.spec_from_file_location("qwen4_three_job_preset_builder_694", COMMON)
if spec is None or spec.loader is None:
    raise ImportError("cannot load the frozen three-job preset builder")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

if __name__ == "__main__":
    module.main()

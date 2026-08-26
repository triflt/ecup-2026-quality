from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "693_qwen4_causal_distillation/evaluate.py"
spec = importlib.util.spec_from_file_location("exp693_eval_for_694", BASE)
if spec is None or spec.loader is None:
    raise ImportError("cannot load shared frozen evaluator")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

if __name__ == "__main__":
    module.main(frozen_method="hardneg")

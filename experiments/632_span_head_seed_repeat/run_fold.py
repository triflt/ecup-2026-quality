from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "experiments/623_semantic_v3_multitask_span_head/run_fold.py"


def load_parent():
    spec = importlib.util.spec_from_file_location("_exp632_parent_runner", PARENT)
    if spec is None or spec.loader is None:
        raise ImportError(PARENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXPERIMENT_ID = "632"
    module.SEED = 31415
    return module


if __name__ == "__main__":
    parent = load_parent()
    print(json.dumps(parent.run(parent.parse_args()), ensure_ascii=False, indent=2, sort_keys=True))

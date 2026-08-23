from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / "experiments/623_semantic_v3_multitask_span_head/evaluate_screen.py"


def load_parent():
    spec = importlib.util.spec_from_file_location("_exp632_parent_screen", PARENT)
    if spec is None or spec.loader is None:
        raise ImportError(PARENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.CANDIDATE_EXPERIMENT_ID = "632"
    module.CANDIDATE_SEED = 31415
    module.RUNTIME_EXPERIMENT_ID = "623"
    return module


def main() -> None:
    parent = load_parent()
    args = parent.parse_args()
    result = parent.evaluate_screen(
        registry_path=args.registry,
        baseline_dirs=parent._fold_paths(args.baseline_fold, required=parent.FOLDS),
        runtime_dirs=parent._fold_paths(args.runtime_fold, required=parent.SCREEN_FOLDS),
        artifact_dirs=parent._fold_paths(args.artifact_fold, required=parent.SCREEN_FOLDS),
        threshold_contract_path=args.threshold_contract,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

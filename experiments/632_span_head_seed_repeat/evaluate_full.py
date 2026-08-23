from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / "experiments/623_semantic_v3_multitask_span_head/evaluate_full.py"


def load_parent():
    spec = importlib.util.spec_from_file_location("_exp632_parent_full", PARENT)
    if spec is None or spec.loader is None:
        raise ImportError(PARENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.screen.CANDIDATE_EXPERIMENT_ID = "632"
    module.screen.CANDIDATE_SEED = 31415
    module.screen.RUNTIME_EXPERIMENT_ID = "623"
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the full independent-seed repeat.")
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--baseline-fold", action="append", required=True)
    parser.add_argument("--runtime-fold", action="append", required=True)
    parser.add_argument("--artifact-fold", action="append", required=True)
    parser.add_argument("--threshold-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    parent = load_parent()
    args = parse_args()
    temporary = args.output.with_name(args.output.name + ".parent")
    if temporary.exists():
        raise FileExistsError("refusing to overwrite temporary parent report")
    result = parent.evaluate_full(
        registry_path=args.registry,
        baseline_dirs=parent._fold_paths(args.baseline_fold, required=parent.FOLDS),
        runtime_dirs=parent._fold_paths(args.runtime_fold, required=parent.FOLDS),
        artifact_dirs=parent._fold_paths(args.artifact_fold, required=parent.FOLDS),
        threshold_contract_path=args.threshold_contract,
        output_path=temporary,
    )
    result["protocol"] = "632_independent_seed_full_five_fold_v1"
    result["candidate_experiment_id"] = "632"
    result["candidate_seed"] = 31415
    result["decision"] = "GO_INTEGRATE_635" if result["passed"] else "NO_GO_REJECT_632"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.unlink()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

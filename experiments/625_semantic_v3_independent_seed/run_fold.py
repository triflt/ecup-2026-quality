from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_DIR = ROOT / "experiments/623_semantic_v3_multitask_span_head"
SPEC_PATH = HERE / "frozen_spec.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_spec() -> dict[str, Any]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec.get("experiment_id") != "625":
        raise ValueError("experiment-625 frozen spec identity mismatch")
    if spec.get("source_seed") == spec.get("independent_seed"):
        raise ValueError("independent seed must differ from the source seed")
    if spec.get("outer_folds") != [0, 1, 2, 3, 4]:
        raise ValueError("all five physical folds must be frozen")
    if spec.get("one_gpu_per_fold") is not True or spec.get("sealed_rows_allowed") != 0:
        raise ValueError("compute or sealed-data contract mismatch")
    return spec


def verify_source_recipe(spec: dict[str, Any]) -> None:
    expected = spec.get("source_recipe_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("source recipe checksum map is missing")
    observed = {}
    for name, checksum in expected.items():
        path = SOURCE_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"source recipe file is missing: {name}")
        observed[name] = sha256_file(path)
        if observed[name] != checksum:
            raise ValueError(f"source recipe checksum mismatch: {name}")


def verify_route_manifest(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "exp624_route_recipe_v1",
        "experiment_id": "624",
        "decision": "ACCEPT",
        "winner_component_experiment": "623",
        "validation": "semantic_family_v3",
        "source_seed": spec["source_seed"],
        "threshold_contract_sha256": spec["frozen_full_threshold_contract_sha256"],
        "uses_sealed_holdout": False,
        "reference_route_weights_unchanged": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": manifest.get(key)}
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"experiment-624 route manifest mismatch: {mismatches}")
    if manifest.get("source_recipe_sha256") != spec["source_recipe_sha256"]:
        raise ValueError("route manifest does not bind the exact experiment-623 recipe")
    declared = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if declared != canonical_sha256(unsigned):
        raise ValueError("route manifest canonical checksum mismatch")
    return manifest


def load_source_runner(spec: dict[str, Any]):
    verify_source_recipe(spec)
    if str(SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(SOURCE_DIR))
    module_spec = importlib.util.spec_from_file_location(
        "_exp625_exact_source_runner", SOURCE_DIR / "run_fold.py"
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError("cannot load the exact experiment-623 runner")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    module.SEED = int(spec["independent_seed"])
    return module


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    spec = load_spec()
    if args.fold not in spec["outer_folds"]:
        raise ValueError("fold is outside the frozen five-fold scope")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("experiment 625 requires exactly one visible GPU per fold")
    manifest = verify_route_manifest(args.recipe_manifest.resolve(), spec)
    source = load_source_runner(spec)
    source_args = argparse.Namespace(
        fold=args.fold,
        runtime_dir=args.runtime_dir,
        images=args.images,
        model_root=args.model_root,
        model_revision=args.model_revision,
        vendor=args.vendor,
        output_dir=args.output_dir,
    )
    report = source.run(source_args)
    if report.get("seed") != spec["independent_seed"]:
        raise ValueError("source runner did not apply the frozen independent seed")
    if report.get("sealed_rows_used") != 0 or report.get("validation_labels_written") != 0:
        raise ValueError("source output violates label/sealed isolation")
    if report.get("outer_fold") != args.fold or report.get("decision") != "GO":
        raise ValueError("source output contract identity mismatch")
    report.update(
        {
            "experiment_id": "625",
            "source_component_experiment": "623",
            "route_recipe_experiment": "624",
            "route_recipe_manifest_sha256": manifest["manifest_sha256"],
            "frozen_spec_sha256": sha256_file(SPEC_PATH),
            "threshold_contract_sha256": spec["frozen_full_threshold_contract_sha256"],
            "only_changed_factor": "training_seed",
            "one_gpu_required": True,
        }
    )
    report.pop("contract_sha256", None)
    report["contract_sha256"] = canonical_sha256(report)
    output_contract = args.output_dir.resolve() / "output_contract.json"
    output_contract.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the accepted experiment-623 recipe with frozen seed 31415."
    )
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--recipe-manifest", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--vendor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))

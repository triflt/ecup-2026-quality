"""Export exact experiment-140 development predictions without labels."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_evaluator(root: Path):
    path = root / "experiments/635_span_head_full140_integration/evaluate.py"
    spec = importlib.util.spec_from_file_location("exp635_evaluate_for_670", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import experiment 635 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def export(*, bundle: Path, contract: Path, registry: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    root = Path(__file__).resolve().parents[2]
    evaluator = _load_evaluator(root)
    frozen = evaluator.load_spec()
    evaluator.verify_source_recipe(frozen)
    frame = evaluator._load_registry(registry, spec=frozen)
    replay_contract = evaluator.verify_replay_contract(
        path=contract,
        bundle_path=bundle,
        registry_path=registry,
        spec=frozen,
    )
    arrays = evaluator.load_replay_bundle(
        path=bundle,
        contract=replay_contract,
        registry=frame,
    )
    valid = evaluator.structural_evidence_mask(arrays)
    probability = evaluator.variant_probabilities(arrays, valid)[evaluator.BASELINE]
    predictions, _ = evaluator.route_predictions(
        bundle=arrays,
        qwen_probability=probability,
        route=frozen["route"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "prediction", "source"))
        writer.writeheader()
        for row_id, prediction in zip(arrays["ids"].astype(str), predictions, strict=True):
            writer.writerow(
                {"id": row_id, "prediction": int(prediction), "source": "exact_full140_semantic_v3"}
            )
    return {
        "rows": len(predictions),
        "positive_predictions": int(predictions.sum()),
        "bundle_sha256": evaluator.sha256_file(bundle),
        "contract_sha256": evaluator.sha256_file(contract),
        "output_sha256": evaluator.sha256_file(output),
        "labels_written": 0,
        "sealed_rows_loaded": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(**vars(args)), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

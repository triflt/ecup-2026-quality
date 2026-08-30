from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from contract import (
    DEVELOPMENT_FOLDS,
    EXPECTED_DEVELOPMENT_ROWS,
    canonical_sha256,
    id_sequence_sha256,
    load_development_registry,
    sha256_file,
)
from score_hypothesis_shard import SAFE_DATA_COLUMNS


def prepare_runtime(
    *,
    data_path: Path,
    folds_path: Path,
    visual_bundle_path: Path,
    route_predictions_path: Path,
    evidence_manifest_path: Path,
    evidence_audit_path: Path,
    output_dir: Path,
    enforce_frozen: bool = True,
    expected_rows: int = EXPECTED_DEVELOPMENT_ROWS,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    registry = load_development_registry(
        folds_path,
        enforce_frozen=enforce_frozen,
        expected_rows=expected_rows,
    )
    data = pd.read_csv(data_path, usecols=list(SAFE_DATA_COLUMNS), dtype={"id": str})
    if set(data.columns) != set(SAFE_DATA_COLUMNS):
        raise ValueError("safe development data schema mismatch")
    data = data.loc[:, list(SAFE_DATA_COLUMNS)]
    if len(data) != expected_rows or data["id"].duplicated().any():
        raise ValueError("safe development data row count or ID uniqueness mismatch")
    expected_ids = registry["id"].astype(str).tolist()
    if data["id"].astype(str).tolist() != expected_ids:
        raise ValueError("safe development data order differs from frozen registry")
    if not data["category"].astype(str).equals(registry["category"].astype(str)):
        raise ValueError("safe development categories differ from frozen registry")

    output_dir.mkdir(parents=True, exist_ok=False)
    data_out = output_dir / "development_feature_data.csv"
    folds_out = output_dir / "development_feature_folds.csv"
    data.to_csv(data_out, index=False)
    registry.to_csv(folds_out, index=False)

    with np.load(visual_bundle_path, allow_pickle=False) as visual:
        visual_ids = visual["ids"].astype(str)
        visual_categories = visual["categories"].astype(str)
        robust_scores = visual["robust_base_score"].astype(np.float32)
        qwen3vl_scores = visual["qwen3vl_score"].astype(np.float32)
    with np.load(route_predictions_path, allow_pickle=False) as route:
        route_ids = route["ids"].astype(str)
        route_categories = route["categories"].astype(str)
        baseline_predictions = route["original_route_predictions"].astype(np.int8)
    expected_array_ids = np.asarray(expected_ids, dtype=str)
    expected_categories = registry["category"].astype(str).to_numpy(dtype=str)
    if not np.array_equal(visual_ids, expected_array_ids) or not np.array_equal(
        route_ids, expected_array_ids
    ):
        raise ValueError("selector source IDs/order differ from frozen registry")
    if not np.array_equal(visual_categories, expected_categories) or not np.array_equal(
        route_categories, expected_categories
    ):
        raise ValueError("selector source categories differ from frozen registry")
    if (
        not np.isfinite(robust_scores).all()
        or not np.isfinite(qwen3vl_scores).all()
        or not np.isin(baseline_predictions, [0, 1]).all()
    ):
        raise ValueError("selector source scores or predictions are invalid")
    selector_out = output_dir / "selector_components.npz"
    np.savez_compressed(
        selector_out,
        ids=expected_array_ids,
        categories=expected_categories,
        robust_base_score=robust_scores,
        qwen3vl_score=qwen3vl_scores,
        baseline_predictions=baseline_predictions,
    )

    evidence_audit = json.loads(evidence_audit_path.read_text(encoding="utf-8"))
    if evidence_audit.get("decision") != "GO":
        raise ValueError("evidence manifest audit did not pass")
    expected_manifest_sha = evidence_audit.get("output_sha256", {}).get(
        evidence_manifest_path.name
    )
    if expected_manifest_sha != sha256_file(evidence_manifest_path):
        raise ValueError("evidence manifest checksum mismatch")
    evidence_out = output_dir / "development_evidence_manifest.jsonl"
    evidence_audit_out = output_dir / "evidence_manifest_audit.json"
    shutil.copy2(evidence_manifest_path, evidence_out)
    shutil.copy2(evidence_audit_path, evidence_audit_out)

    audit = {
        "status": "complete",
        "experiment_id": "620",
        "stage": "physical_label_free_feature_runtime",
        "development_rows": len(data),
        "development_ids_sha256": id_sequence_sha256(expected_ids),
        "data_columns": list(SAFE_DATA_COLUMNS),
        "labels_present": False,
        "sealed_rows_present": 0,
        "folds": list(DEVELOPMENT_FOLDS),
        "data_sha256": sha256_file(data_out),
        "folds_sha256": sha256_file(folds_out),
        "selector_components_sha256": sha256_file(selector_out),
        "evidence_manifest_sha256": sha256_file(evidence_out),
        "evidence_audit_sha256": sha256_file(evidence_audit_out),
        "competition_trained_adapters_present": False,
        "selector_output_arrays": [
            "ids",
            "categories",
            "robust_base_score",
            "qwen3vl_score",
            "baseline_predictions"
        ],
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    (output_dir / "runtime_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--visual-bundle", type=Path, required=True)
    parser.add_argument("--route-predictions", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--evidence-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = prepare_runtime(
        data_path=args.data,
        folds_path=args.folds,
        visual_bundle_path=args.visual_bundle,
        route_predictions_path=args.route_predictions,
        evidence_manifest_path=args.evidence_manifest,
        evidence_audit_path=args.evidence_audit,
        output_dir=args.output_dir,
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

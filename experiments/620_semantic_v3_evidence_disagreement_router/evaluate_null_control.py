from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from contract import (
    DEFAULT_600_METRICS,
    DEFAULT_601_METRICS,
    DEFAULT_FOLDS,
    DEFAULT_RUNTIME_CONTRACT,
    canonical_sha256,
    load_development_registry,
    load_json,
    require_complete_metrics,
    sha256_file,
    validate_qwen35_component_bundle,
    validate_visual_component_bundle,
)

BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"
ROUTE = {
    BAD: {
        "weight_robust": 0.50,
        "weight_qwen3vl": 0.25,
        "weight_qwen35": 0.25,
        "threshold": 0.27193570137023926,
        "qwen35_component": "original",
    },
    FLAMMABLE: {
        "weight_robust": 0.15,
        "weight_qwen3vl": 0.10,
        "weight_qwen35": 0.75,
        "threshold": 0.953912615776062,
        "qwen35_component": "specialist",
    },
}


def _validate_evidence_manifest(manifest_path: Path, audit_path: Path, ids: np.ndarray) -> str:
    audit = load_json(audit_path)
    if audit.get("decision") != "GO":
        raise RuntimeError("evidence manifest preflight did not pass")
    expected_flags = {
        "labels_loaded": False,
        "sealed_rows_in_inputs": 0,
        "sealed_rows_in_outputs": 0,
        "exact_offset_failures": 0,
        "forbidden_flip_evidence": 0,
        "unresolved_conflicts": 0,
    }
    for key, expected in expected_flags.items():
        if audit.get(key) != expected:
            raise ValueError(f"evidence audit mismatch for {key}")
    if audit.get("output_sha256", {}).get(manifest_path.name) != sha256_file(manifest_path):
        raise ValueError("evidence manifest checksum mismatch")
    manifest_ids: list[str] = []
    forbidden_keys = {"label", "labels", "gold", "target", "targets"}
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if forbidden_keys & set(row):
                raise ValueError(f"manifest line {line_number} contains supervision")
            manifest_ids.append(str(row["id"]))
    if not np.array_equal(np.asarray(manifest_ids, dtype=str), ids.astype(str)):
        raise ValueError("evidence manifest IDs/order mismatch")
    return sha256_file(manifest_path)


def _baseline_route(
    *,
    categories: np.ndarray,
    robust_rank: np.ndarray,
    qwen3vl_rank: np.ndarray,
    original_rank: np.ndarray,
    specialist_rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty(len(categories), dtype=np.float32)
    predictions = np.empty(len(categories), dtype=np.int8)
    for category, config in ROUTE.items():
        mask = categories == category
        qwen_rank = original_rank if config["qwen35_component"] == "original" else specialist_rank
        scores[mask] = (
            config["weight_robust"] * robust_rank[mask]
            + config["weight_qwen3vl"] * qwen3vl_rank[mask]
            + config["weight_qwen35"] * qwen_rank[mask]
        )
        predictions[mask] = (scores[mask] >= config["threshold"]).astype(np.int8)
    return scores, predictions


def evaluate_null_control(
    *,
    folds_path: Path,
    metrics_600_path: Path,
    metrics_601_path: Path,
    qwen35_bundle_path: Path,
    qwen35_contract_path: Path,
    visual_bundle_path: Path,
    visual_contract_path: Path,
    evidence_manifest_path: Path,
    evidence_audit_path: Path,
    output_dir: Path,
    runtime_contract_path: Path = DEFAULT_RUNTIME_CONTRACT,
    enforce_frozen: bool = True,
    expected_rows: int = 11_118,
) -> dict[str, Any]:
    """Prove that the disabled router is an exact no-op, without loading labels."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")

    # Deliberately check experiment status before creating any output. An incomplete
    # semantic-v3 OOF set must fail closed rather than produce a partial baseline.
    require_complete_metrics(metrics_600_path, experiment_id="600")
    require_complete_metrics(metrics_601_path, experiment_id="601")
    registry = load_development_registry(
        folds_path,
        enforce_frozen=enforce_frozen,
        expected_rows=expected_rows,
    )
    visual = validate_visual_component_bundle(
        visual_bundle_path,
        contract_path=visual_contract_path,
        registry=registry,
    )
    qwen35 = validate_qwen35_component_bundle(
        qwen35_bundle_path,
        contract_path=qwen35_contract_path,
        registry=registry,
    )
    for key in ("ids", "categories", "folds", "semantic_components"):
        if not np.array_equal(visual[key], qwen35[key]):
            raise ValueError(f"experiment-600/601 component mismatch for {key}")
    manifest_sha256 = _validate_evidence_manifest(
        evidence_manifest_path, evidence_audit_path, visual["ids"]
    )
    runtime_contract = load_json(runtime_contract_path)
    if runtime_contract.get("status") != "placeholder":
        raise ValueError("unexpected runtime contract version/status")

    baseline_scores, baseline_predictions = _baseline_route(
        categories=visual["categories"],
        robust_rank=visual["robust_base_rank"],
        qwen3vl_rank=visual["qwen3vl_rank"],
        original_rank=qwen35["qwen35_original_rank"],
        specialist_rank=qwen35["qwen35_specialist_rank"],
    )
    candidate_predictions = baseline_predictions.copy()
    changed = baseline_predictions != candidate_predictions
    if changed.any():
        raise AssertionError("null router changed predictions")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "null_router_output.npz"
    np.savez_compressed(
        output_path,
        ids=visual["ids"],
        folds=visual["folds"],
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
        eligible=np.zeros(len(baseline_predictions), dtype=bool),
        router_probabilities=baseline_scores,
        router_proposals=baseline_predictions.copy(),
        support_from_robust=np.zeros(len(baseline_predictions), dtype=bool),
        support_from_qwen3vl=np.zeros(len(baseline_predictions), dtype=bool),
        evidence_manifest_sha256=np.asarray(manifest_sha256),
        feature_bundle_sha256=np.asarray("NULL_CONTROL_NO_FEATURE_BUNDLE"),
    )
    required_arrays = set(runtime_contract["router_output"]["required_arrays"])
    with np.load(output_path, allow_pickle=False) as output:
        missing = required_arrays - set(output.files)
        forbidden = {"labels", "gold", "targets", "sealed_ids"} & set(output.files)
        if missing or forbidden:
            raise AssertionError(
                f"null output contract failure: missing={sorted(missing)}, forbidden={sorted(forbidden)}"
            )
    audit = {
        "experiment_id": "620",
        "protocol": "semantic_v3_evidence_disagreement_router_null_v1",
        "status": "complete",
        "rows": len(baseline_predictions),
        "changed_predictions": int(changed.sum()),
        "eligible_rows": 0,
        "labels_loaded": False,
        "sealed_rows_loaded": 0,
        "route": ROUTE,
        "input_sha256": {
            "metrics_600": sha256_file(metrics_600_path),
            "metrics_601": sha256_file(metrics_601_path),
            "qwen35_bundle": sha256_file(qwen35_bundle_path),
            "qwen35_contract": sha256_file(qwen35_contract_path),
            "visual_bundle": sha256_file(visual_bundle_path),
            "visual_contract": sha256_file(visual_contract_path),
            "evidence_manifest": manifest_sha256,
            "evidence_audit": sha256_file(evidence_audit_path),
        },
        "output_sha256": {output_path.name: sha256_file(output_path)},
        "gates": {
            "complete_600_oof": True,
            "complete_601_oof": True,
            "exact_id_alignment": True,
            "null_predictions_bit_exact": True,
            "no_labels_loaded": True,
            "zero_sealed_rows": True,
        },
        "decision": "GO",
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    (output_dir / "null_control_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed label-free null control for the semantic-v3 router."
    )
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--metrics-600", type=Path, default=DEFAULT_600_METRICS)
    parser.add_argument("--metrics-601", type=Path, default=DEFAULT_601_METRICS)
    parser.add_argument("--qwen35-bundle", type=Path, required=True)
    parser.add_argument("--qwen35-contract", type=Path, required=True)
    parser.add_argument("--visual-bundle", type=Path, required=True)
    parser.add_argument("--visual-contract", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--evidence-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = evaluate_null_control(
        folds_path=args.folds.resolve(),
        metrics_600_path=args.metrics_600.resolve(),
        metrics_601_path=args.metrics_601.resolve(),
        qwen35_bundle_path=args.qwen35_bundle.resolve(),
        qwen35_contract_path=args.qwen35_contract.resolve(),
        visual_bundle_path=args.visual_bundle.resolve(),
        visual_contract_path=args.visual_contract.resolve(),
        evidence_manifest_path=args.evidence_manifest.resolve(),
        evidence_audit_path=args.evidence_audit.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

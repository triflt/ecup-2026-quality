from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from contract import PROTOCOL_VERSION, canonical_sha256, load_json, sha256_file
from score_hypothesis_shard import (
    load_category_rows,
    load_evidence_manifest,
    load_selector_components,
    select_candidate_rows,
    select_hypothesis,
)

SCREEN_FOLDS = (0, 3)
SCORE_THRESHOLD = 0.90


def macro_f1(labels: np.ndarray, predictions: np.ndarray, categories: np.ndarray) -> float:
    values = [
        f1_score(labels[categories == category], predictions[categories == category], zero_division=0)
        for category in ("БАД", "Легковоспламеняющиеся")
    ]
    return float(np.mean(values))


def correction_counts(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> tuple[int, int]:
    corrected = int(((baseline != labels) & (candidate == labels)).sum())
    regressed = int(((baseline == labels) & (candidate != labels)).sum())
    return corrected, regressed


def load_verified_shard(
    *,
    directory: Path,
    hypothesis_id: str,
    expected_ids: np.ndarray,
    spec_sha256: str,
    backbone_decision_sha256: str,
    runtime_audit_sha256: str,
) -> np.ndarray:
    report_path = directory / "hypothesis_score_report.json"
    bundle_path = directory / "hypothesis_score_shard.npz"
    report = load_json(report_path)
    supplied_hash = report.pop("report_sha256", None)
    if supplied_hash != canonical_sha256(report):
        raise ValueError(f"{hypothesis_id} report self-hash mismatch")
    expected = {
        "status": "complete",
        "experiment_id": "620",
        "protocol_version": PROTOCOL_VERSION,
        "hypothesis_id": hypothesis_id,
        "labels_loaded": False,
        "sealed_rows_loaded": 0,
        "competition_trained_adapter_used": False,
        "spec_sha256": spec_sha256,
        "backbone_decision_sha256": backbone_decision_sha256,
        "runtime_audit_sha256": runtime_audit_sha256,
        "scorer_sha256": sha256_file(HERE / "score_hypothesis_shard.py"),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{hypothesis_id} report mismatch for {key}")
    if report.get("bundle_sha256") != sha256_file(bundle_path):
        raise ValueError(f"{hypothesis_id} bundle checksum mismatch")
    with np.load(bundle_path, allow_pickle=False) as bundle:
        required = {
            "ids",
            "categories",
            "hypothesis_id",
            "supports_verdict",
            "hypothesis_scores",
            "protocol_version",
            "spec_sha256",
            "backbone_decision_sha256",
            "scorer_sha256",
        }
        if set(bundle.files) != required:
            raise ValueError(f"{hypothesis_id} bundle schema mismatch")
        if str(bundle["hypothesis_id"]) != hypothesis_id:
            raise ValueError(f"{hypothesis_id} bundle identity mismatch")
        if str(bundle["scorer_sha256"]) != expected["scorer_sha256"]:
            raise ValueError(f"{hypothesis_id} bundle scorer checksum mismatch")
        ids = bundle["ids"].astype(str)
        scores = bundle["hypothesis_scores"].astype(np.float32)
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"{hypothesis_id} selected IDs/order mismatch")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"{hypothesis_id} scores invalid")
    return scores


def screen_metrics(
    *,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    fold_deltas: dict[str, float] = {}
    for fold in SCREEN_FOLDS:
        mask = folds == fold
        fold_deltas[str(fold)] = macro_f1(labels[mask], candidate[mask], categories[mask]) - macro_f1(
            labels[mask], baseline[mask], categories[mask]
        )
    mask = np.isin(folds, SCREEN_FOLDS)
    corrected, regressed = correction_counts(labels[mask], baseline[mask], candidate[mask])
    ratio = None if regressed == 0 else corrected / regressed
    category_deltas = {}
    for category in ("БАД", "Легковоспламеняющиеся"):
        local = mask & (categories == category)
        category_deltas[category] = float(
            f1_score(labels[local], candidate[local], zero_division=0)
            - f1_score(labels[local], baseline[local], zero_division=0)
        )
    flammable = mask & (categories == "Легковоспламеняющиеся")
    flammable_fn_delta = int(
        ((labels[flammable] == 1) & (candidate[flammable] == 0)).sum()
        - ((labels[flammable] == 1) & (baseline[flammable] == 0)).sum()
    )
    safety_fn_delta = int(
        ((labels[mask] == 1) & (candidate[mask] == 0)).sum()
        - ((labels[mask] == 1) & (baseline[mask] == 0)).sum()
    )
    mean_delta = float(np.mean(list(fold_deltas.values())))
    gates = {
        "each_screen_fold_positive": all(value > 0 for value in fold_deltas.values()),
        "mean_macro_delta_at_least_0_0015": mean_delta >= 0.0015,
        "corrected_to_regressed_at_least_1_5": (
            corrected > 0 if ratio is None else ratio >= 1.5
        ),
        "maximum_category_drop_0_002": min(category_deltas.values()) >= -0.002,
        "flammable_false_negatives_do_not_increase": flammable_fn_delta <= 0,
        "safety_false_negatives_do_not_increase": safety_fn_delta <= 0,
    }
    return {
        "fold_deltas": fold_deltas,
        "mean_fold_delta": mean_delta,
        "category_deltas": category_deltas,
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": ratio,
        "flammable_false_negative_delta": flammable_fn_delta,
        "safety_false_negative_delta": safety_fn_delta,
        "gates": gates,
        "passed": all(gates.values()),
    }


def evaluate(
    *,
    runtime_dir: Path,
    route_predictions_path: Path,
    spec_path: Path,
    backbone_decision_path: Path,
    shards: dict[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    spec = load_json(spec_path)
    decision = load_json(backbone_decision_path)
    spec_hash = sha256_file(spec_path)
    decision_hash = canonical_sha256(decision)
    runtime_audit = load_json(runtime_dir / "runtime_audit.json")
    runtime_audit_hash = canonical_sha256(runtime_audit)
    expected_shards = set(spec["scored_hypothesis_ids"])
    if set(shards) != expected_shards:
        raise ValueError("shard set differs from the frozen scored hypothesis IDs")
    selector = load_selector_components(runtime_dir / "selector_components.npz")
    evidence = load_evidence_manifest(runtime_dir / "development_evidence_manifest.jsonl")
    with np.load(route_predictions_path, allow_pickle=False) as route:
        ids = route["ids"].astype(str)
        labels = route["labels"].astype(np.int8)
        categories = route["categories"].astype(str)
        folds = route["folds"].astype(np.int8)
        components = route["semantic_components"].astype(str)
        baseline = route["original_route_predictions"].astype(np.int8)
    if not np.array_equal(ids, selector["ids"].astype(str)):
        raise ValueError("route and label-free selector IDs/order mismatch")

    candidate = baseline.copy()
    high_support = np.zeros(len(ids), dtype=bool)
    shard_summary: dict[str, Any] = {}
    for hypothesis_id, directory in sorted(shards.items()):
        category, hypothesis = select_hypothesis(spec, hypothesis_id)
        rows = load_category_rows(
            data_path=runtime_dir / "development_feature_data.csv",
            folds_path=runtime_dir / "development_feature_folds.csv",
            category=category,
            enforce_frozen=False,
        )
        selected, selection = select_candidate_rows(
            rows=rows,
            selector=selector,
            evidence_rows=evidence,
            hypothesis=hypothesis,
            compatible_concepts=spec["hypothesis_concept_compatibility"][hypothesis_id],
        )
        selected_ids = selected["id"].astype(str).to_numpy()
        scores = load_verified_shard(
            directory=directory,
            hypothesis_id=hypothesis_id,
            expected_ids=selected_ids,
            spec_sha256=spec_hash,
            backbone_decision_sha256=decision_hash,
            runtime_audit_sha256=runtime_audit_hash,
        )
        positions = {value: index for index, value in enumerate(ids)}
        local_positions = np.asarray([positions[value] for value in selected_ids], dtype=np.int64)
        accepted = scores >= SCORE_THRESHOLD
        high_support[local_positions[accepted]] = True
        candidate[local_positions[accepted]] = int(hypothesis["supports_verdict"])
        shard_summary[hypothesis_id] = {
            "selected_rows": len(selected_ids),
            "high_support_rows": int(accepted.sum()),
            "selection": selection,
            "artifact_sha256": sha256_file(directory / "hypothesis_score_shard.npz"),
        }

    changed = candidate != baseline
    if not np.array_equal(changed, high_support):
        raise AssertionError("every changed row must have a high-support frozen hypothesis")
    screen = screen_metrics(
        labels=labels,
        categories=categories,
        folds=folds,
        baseline=baseline,
        candidate=candidate,
    )
    result = {
        "experiment_id": "620",
        "protocol_version": PROTOCOL_VERSION,
        "status": "screen_passed_full_audit_pending" if screen["passed"] else "rejected_at_screen",
        "score_threshold": SCORE_THRESHOLD,
        "score_semantics": "binary_normalized_support_score_not_calibrated_probability",
        "trainable_meta_router": False,
        "competition_trained_hypothesis_adapter": False,
        "rows": len(ids),
        "changed": int(changed.sum()),
        "shards": shard_summary,
        "screen": screen,
        "full_evaluation_released": False,
        "sealed_holdout_used": False,
        "spec_sha256": spec_hash,
        "backbone_decision_sha256": decision_hash,
        "semantic_components_sha256": canonical_sha256(components.tolist()),
        "decision": "CONTINUE_FULL_AUDIT" if screen["passed"] else "NO_GO",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "screen_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "screen_predictions.npz",
        ids=ids,
        folds=folds,
        semantic_components=components,
        baseline_predictions=baseline,
        candidate_predictions=candidate,
        changed=changed,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--route-predictions", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=HERE / "frozen_router_spec.json")
    parser.add_argument(
        "--backbone-decision", type=Path, default=HERE / "backbone_compatibility_decision.json"
    )
    parser.add_argument("--shard", nargs=2, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    shards = {hypothesis_id: Path(path) for hypothesis_id, path in args.shard}
    result = evaluate(
        runtime_dir=args.runtime_dir,
        route_predictions_path=args.route_predictions,
        spec_path=args.spec,
        backbone_decision_path=args.backbone_decision,
        shards=shards,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

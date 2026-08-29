from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from grid_contract import (
    CATEGORIES,
    FULL_FOLDS,
    GRID_CONTRACT_SHA256,
    MODEL_REVISIONS,
    NO_EVIDENCE,
    PREPROCESSING_VERSION,
    PROMPT_VERSION,
    SCREEN_FOLDS,
    canonical_sha256,
    resolve_grounding,
    sha256_file,
)

TRACKS = {
    "prompt_4b": ("Qwen/Qwen3.5-4B", "prompting", "none"),
    "prompt_27b": ("Qwen/Qwen3.8-27B", "prompting", "none"),
    "class_4b": ("Qwen/Qwen3.5-4B", "class_only", "none"),
    "class_27b": ("Qwen/Qwen3.8-27B", "class_only", "none"),
    "evidence_class_first_4b": (
        "Qwen/Qwen3.5-4B",
        "grounded_evidence",
        "class_first",
    ),
    "evidence_class_first_27b": (
        "Qwen/Qwen3.8-27B",
        "grounded_evidence",
        "class_first",
    ),
    "evidence_first_4b": (
        "Qwen/Qwen3.5-4B",
        "grounded_evidence",
        "evidence_first",
    ),
    "evidence_first_27b": (
        "Qwen/Qwen3.8-27B",
        "grounded_evidence",
        "evidence_first",
    ),
}
REQUIRED_PREDICTION_FIELDS = {
    "global_index",
    "id",
    "fold",
    "category",
    "score",
    "prediction",
    "model_id",
    "model_revision",
    "objective",
    "target_order",
    "prompt_version",
    "preprocessing_version",
    "raw_generation",
    "generated_verdict",
    "format_valid",
    "quote",
    "concept",
    "grounded",
    "grounding_source",
    "image_index",
    "region_index",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def _parse_assignments(values: list[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or name not in TRACKS or not raw_path:
            raise ValueError(f"expected TRACK=PATH with a frozen track name: {value}")
        result[name].append(Path(raw_path))
    missing = set(TRACKS) - set(result)
    if missing:
        raise ValueError(f"missing grid tracks: {sorted(missing)}")
    return dict(result)


def _load_runtimes(paths: list[Path], folds: tuple[int, ...]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            if int(row["fold"]) not in folds:
                continue
            row_id = str(row["id"])
            if row_id in rows:
                raise ValueError("runtime validation IDs overlap")
            if "label" in row or "evidence_target" in row:
                raise ValueError("evaluator runtime contains supervision")
            rows[row_id] = row
    return rows


def _verify_contracts(
    contracts: dict[str, list[Path]],
    track_paths: dict[str, list[Path]],
) -> str:
    input_hashes: set[str] = set()
    for track, paths in track_paths.items():
        contract_paths = contracts.get(track, [])
        if len(contract_paths) != len(paths):
            raise ValueError(f"each {track} prediction shard needs one output contract")
        expected_model, expected_objective, _expected_order = TRACKS[track]
        for path, contract_path in zip(paths, contract_paths, strict=True):
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            payload = dict(contract)
            digest = payload.pop("contract_sha256", None)
            if digest != canonical_sha256(payload):
                raise ValueError(f"contract self-hash mismatch: {contract_path}")
            expected = {
                "model_id": expected_model,
                "model_revision": MODEL_REVISIONS[expected_model],
                "objective": expected_objective,
                "grid_contract_sha256": GRID_CONTRACT_SHA256,
                "threshold": 0.0,
                "threshold_tuned": False,
                "sealed_rows_used": 0,
                "decision": "GO_EVALUATE",
            }
            mismatch = {
                key: {"expected": value, "actual": contract.get(key)}
                for key, value in expected.items()
                if contract.get(key) != value
            }
            if mismatch:
                raise ValueError(f"track contract mismatch for {track}: {mismatch}")
            if contract.get("artifacts", {}).get(path.name) != sha256_file(path):
                raise ValueError(f"prediction checksum mismatch for {path}")
            input_hashes.add(str(contract["model_input_view_sha256"]))
    if len(input_hashes) != 1:
        raise ValueError("grid cells did not use an identical model-input view")
    return input_hashes.pop()


def _load_track(
    *,
    name: str,
    paths: list[Path],
    expected: pd.DataFrame,
    runtime_by_id: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = [row for path in paths for row in read_jsonl(path)]
    if any(not REQUIRED_PREDICTION_FIELDS.issubset(row) for row in rows):
        raise ValueError(f"track {name} prediction schema is incomplete")
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["fold"].astype(int).isin(expected["development_fold"])].copy()
    frame = frame.sort_values("id", key=lambda values: values.astype(str)).reset_index(drop=True)
    expected_ids = expected["id"].astype(str).tolist()
    if len(frame) != len(expected) or frame["id"].astype(str).tolist() != expected_ids:
        raise ValueError(f"track {name} does not exactly cover evaluated IDs")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError(f"track {name} contains duplicate IDs")
    expected_model, expected_objective, expected_order = TRACKS[name]
    metadata = {
        "model_id": expected_model,
        "model_revision": MODEL_REVISIONS[expected_model],
        "objective": expected_objective,
        "target_order": expected_order,
        "prompt_version": PROMPT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    for key, value in metadata.items():
        if not frame[key].astype(str).eq(str(value)).all():
            raise ValueError(f"track {name} has inconsistent {key}")
    if not np.isfinite(pd.to_numeric(frame["score"], errors="coerce")).all():
        raise ValueError(f"track {name} contains non-finite scores")
    score = frame["score"].astype(float).to_numpy()
    prediction = frame["prediction"].astype(int).to_numpy()
    if not np.array_equal(prediction, (score >= 0.0).astype(np.int8)):
        raise ValueError(f"track {name} differs from the frozen zero threshold")
    if not np.array_equal(
        frame["fold"].astype(int).to_numpy(),
        expected["development_fold"].astype(int).to_numpy(),
    ):
        raise ValueError(f"track {name} fold assignments differ from registry")
    if frame["category"].astype(str).tolist() != expected["category"].astype(str).tolist():
        raise ValueError(f"track {name} categories differ from registry")

    if expected_objective != "grounded_evidence":
        if not frame["quote"].astype(str).eq(NO_EVIDENCE).all():
            raise ValueError(f"non-evidence track {name} emitted evidence")
        return frame
    for row in frame.to_dict("records"):
        runtime = runtime_by_id[str(row["id"])]
        resolved = resolve_grounding(runtime, str(row["quote"]))
        claims = {
            "grounded": bool(row["grounded"]),
            "grounding_source": str(row["grounding_source"]),
            "image_index": int(row["image_index"]),
            "region_index": int(row["region_index"]),
        }
        actual = {
            "grounded": bool(resolved["grounded"]),
            "grounding_source": str(resolved["source"]),
            "image_index": int(resolved["image_index"]),
            "region_index": int(resolved["region_index"]),
        }
        if claims != actual:
            raise ValueError(f"grounding claim mismatch for track={name}, id={row['id']}")
    return frame


def _track_metrics(
    labels: np.ndarray,
    folds: np.ndarray,
    categories: np.ndarray,
    prediction: np.ndarray,
    scope: np.ndarray,
) -> dict[str, Any]:
    category_metrics: dict[str, Any] = {}
    for category in CATEGORIES:
        local = scope & (categories == category)
        local_labels = labels[local]
        local_predictions = prediction[local]
        category_metrics[category] = {
            "f1": f1(local_labels, local_predictions),
            "tp": int(((local_labels == 1) & (local_predictions == 1)).sum()),
            "fp": int(((local_labels == 0) & (local_predictions == 1)).sum()),
            "fn": int(((local_labels == 1) & (local_predictions == 0)).sum()),
        }
    fold_metrics = {}
    for fold in sorted(set(folds[scope].tolist())):
        values = []
        for category in CATEGORIES:
            local = scope & (folds == fold) & (categories == category)
            values.append(f1(labels[local], prediction[local]))
        fold_metrics[str(fold)] = float(np.mean(values))
    return {
        "macro_f1": float(np.mean([value["f1"] for value in category_metrics.values()])),
        "categories": category_metrics,
        "fold_macro_f1": fold_metrics,
    }


def _comparison(
    *,
    labels: np.ndarray,
    folds: np.ndarray,
    categories: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    scope: np.ndarray,
) -> dict[str, Any]:
    left_metrics = _track_metrics(labels, folds, categories, left, scope)
    right_metrics = _track_metrics(labels, folds, categories, right, scope)
    fold_delta = {
        fold: right_metrics["fold_macro_f1"][fold] - left_metrics["fold_macro_f1"][fold]
        for fold in left_metrics["fold_macro_f1"]
    }
    category_delta = {
        category: right_metrics["categories"][category]["f1"]
        - left_metrics["categories"][category]["f1"]
        for category in CATEGORIES
    }
    corrected = int((scope & (left != labels) & (right == labels)).sum())
    regressed = int((scope & (left == labels) & (right != labels)).sum())
    return {
        "left_macro_f1": left_metrics["macro_f1"],
        "right_macro_f1": right_metrics["macro_f1"],
        "macro_delta": right_metrics["macro_f1"] - left_metrics["macro_f1"],
        "fold_delta": fold_delta,
        "winning_folds": sum(value > 0 for value in fold_delta.values()),
        "category_delta": category_delta,
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": (None if regressed == 0 else corrected / regressed),
    }


def _evidence_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    emitted = ~frame["quote"].astype(str).eq(NO_EVIDENCE)
    emitted_count = int(emitted.sum())
    grounded_emitted = int((emitted & frame["grounded"].astype(bool)).sum())
    return {
        "rows": len(frame),
        "format_valid_fraction": float(frame["format_valid"].astype(bool).mean()),
        "evidence_coverage": emitted_count / max(1, len(frame)),
        "grounded_fraction_of_emitted": grounded_emitted / max(1, emitted_count),
        "text_evidence": int(frame["grounding_source"].astype(str).eq("text").sum()),
        "ocr_evidence": int(frame["grounding_source"].astype(str).eq("ocr").sum()),
        "unresolved_evidence": int(frame["grounding_source"].astype(str).eq("unresolved").sum()),
        "generated_verdict_agreement": float(
            (frame["generated_verdict"].astype(int) == frame["prediction"].astype(int)).mean()
        ),
    }


def _manual_audit(
    manifest_path: Path | None, ratings_path: Path | None, *, expected_track: str
) -> dict[str, Any]:
    if manifest_path is None or ratings_path is None:
        return {"present": False, "passed": False, "reason": "manual_audit_missing"}
    manifest = pd.read_csv(manifest_path, dtype={"id": str})
    ratings = pd.read_csv(ratings_path, dtype={"id": str})
    required_manifest = {"id", "track"}
    required_ratings = {
        "id",
        "track",
        "relevant",
        "unsupported",
        "sold_object_correct",
        "negation_correct",
        "completeness_correct",
    }
    if set(manifest.columns) != required_manifest or set(ratings.columns) != required_ratings:
        raise ValueError("manual audit schema mismatch")
    if len(manifest) < 200 or manifest["id"].duplicated().any():
        raise ValueError("manual audit manifest needs at least 200 unique IDs")
    if not manifest["track"].astype(str).eq(expected_track).all():
        raise ValueError("manual audit manifest targets the wrong frozen track")
    merged = manifest.merge(ratings, on=["id", "track"], how="left", validate="one_to_one")
    rating_columns = sorted(required_ratings - {"id", "track"})
    if merged[rating_columns].isna().any().any():
        raise ValueError("manual audit ratings are incomplete")
    for column in rating_columns:
        if not merged[column].astype(int).isin([0, 1]).all():
            raise ValueError(f"manual audit {column} must be binary")
    relevant = float(merged["relevant"].astype(int).mean())
    unsupported = float(merged["unsupported"].astype(int).mean())
    scope_values = merged[
        ["sold_object_correct", "negation_correct", "completeness_correct"]
    ].to_numpy(np.int8)
    scope = float(scope_values.mean())
    gates = {
        "relevant_at_least_0_90": relevant >= 0.90,
        "unsupported_at_most_0_01": unsupported <= 0.01,
        "scope_correct_at_least_0_95": scope >= 0.95,
    }
    return {
        "present": True,
        "rows": len(merged),
        "relevant_fraction": relevant,
        "unsupported_fraction": unsupported,
        "scope_correct_fraction": scope,
        "gates": gates,
        "passed": all(gates.values()),
    }


def evaluate(
    *,
    registry_path: Path,
    runtime_paths: list[Path],
    track_paths: dict[str, list[Path]],
    contract_paths: dict[str, list[Path]],
    output_path: Path,
    folds_to_evaluate: tuple[int, ...],
    audit_manifest_path: Path | None = None,
    audit_ratings_path: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite grid evaluation")
    registry = pd.read_csv(registry_path, dtype={"id": str})
    registry = registry.loc[
        registry["split"].astype(str).eq("development")
        & registry["development_fold"].astype(int).isin(folds_to_evaluate)
    ].copy()
    registry = registry.sort_values("id", key=lambda values: values.astype(str)).reset_index(
        drop=True
    )
    runtime_by_id = _load_runtimes(runtime_paths, folds_to_evaluate)
    if set(runtime_by_id) != set(registry["id"].astype(str)):
        raise ValueError("label-free evaluator runtimes do not exactly cover registry scope")
    input_view_sha256 = _verify_contracts(contract_paths, track_paths)
    frames = {
        name: _load_track(
            name=name,
            paths=paths,
            expected=registry,
            runtime_by_id=runtime_by_id,
        )
        for name, paths in track_paths.items()
    }
    labels = registry["label"].to_numpy(np.int8)
    folds = registry["development_fold"].to_numpy(np.int8)
    categories = registry["category"].astype(str).to_numpy()
    scope = np.ones(len(registry), dtype=bool)
    predictions = {
        name: frame["prediction"].astype(int).to_numpy() for name, frame in frames.items()
    }
    track_metrics = {
        name: _track_metrics(labels, folds, categories, prediction, scope)
        for name, prediction in predictions.items()
    }
    comparison_pairs = {
        "prompt_scale_4b_to_27b": ("prompt_4b", "prompt_27b"),
        "class_scale_4b_to_27b": ("class_4b", "class_27b"),
        "evidence_first_scale_4b_to_27b": (
            "evidence_first_4b",
            "evidence_first_27b",
        ),
        "class_first_scale_4b_to_27b": (
            "evidence_class_first_4b",
            "evidence_class_first_27b",
        ),
        "4b_class_to_evidence_first": ("class_4b", "evidence_first_4b"),
        "27b_class_to_evidence_first": ("class_27b", "evidence_first_27b"),
        "4b_class_to_class_first_aux": ("class_4b", "evidence_class_first_4b"),
        "27b_class_to_class_first_aux": ("class_27b", "evidence_class_first_27b"),
    }
    comparisons = {
        name: _comparison(
            labels=labels,
            folds=folds,
            categories=categories,
            left=predictions[left],
            right=predictions[right],
            scope=scope,
        )
        for name, (left, right) in comparison_pairs.items()
    }
    prompt_small_score = frames["prompt_4b"]["score"].astype(float).to_numpy()
    prompt_large_score = frames["prompt_27b"]["score"].astype(float).to_numpy()
    hard_scope = (
        (predictions["prompt_4b"] != predictions["prompt_27b"])
        | (np.abs(prompt_small_score) < 0.5)
        | (np.abs(prompt_large_score) < 0.5)
    )
    hard_comparison = _comparison(
        labels=labels,
        folds=folds,
        categories=categories,
        left=predictions["evidence_first_4b"],
        right=predictions["evidence_first_27b"],
        scope=hard_scope,
    )
    hard_comparison["rows"] = int(hard_scope.sum())
    evidence = {
        name: _evidence_metrics(frame)
        for name, frame in frames.items()
        if TRACKS[name][1] == "grounded_evidence"
    }
    manual = _manual_audit(
        audit_manifest_path,
        audit_ratings_path,
        expected_track="evidence_first_27b",
    )
    primary = comparisons["evidence_first_scale_4b_to_27b"]
    screen_gates = {
        "both_screen_folds_positive": primary["winning_folds"] == len(folds_to_evaluate),
        "macro_delta_at_least_0_003": primary["macro_delta"] >= 0.003,
        "no_category_drop_below_minus_0_002": min(primary["category_delta"].values()) >= -0.002,
        "corrected_to_regressed_at_least_1_5": (
            primary["corrected"] > 0
            if primary["regressed"] == 0
            else primary["corrected"] / primary["regressed"] >= 1.5
        ),
        "hard_cohort_positive": hard_comparison["rows"] > 0 and hard_comparison["macro_delta"] > 0,
    }
    full = folds_to_evaluate == FULL_FOLDS
    distillation_gates = {
        "full_five_fold_evaluation": full,
        "wins_at_least_4_of_5": full and primary["winning_folds"] >= 4,
        "macro_delta_at_least_0_003": primary["macro_delta"] >= 0.003,
        "no_category_drop_below_minus_0_002": min(primary["category_delta"].values()) >= -0.002,
        "hard_cohort_positive": hard_comparison["rows"] > 0 and hard_comparison["macro_delta"] > 0,
        "corrected_to_regressed_at_least_1_5": (
            primary["corrected"] > 0
            if primary["regressed"] == 0
            else primary["corrected"] / primary["regressed"] >= 1.5
        ),
        "large_evidence_format_at_least_0_95": evidence["evidence_first_27b"][
            "format_valid_fraction"
        ]
        >= 0.95,
        "large_evidence_grounded_at_least_0_95": evidence["evidence_first_27b"][
            "grounded_fraction_of_emitted"
        ]
        >= 0.95,
        "large_evidence_coverage_at_least_0_50": evidence["evidence_first_27b"]["evidence_coverage"]
        >= 0.50,
        "manual_audit_passed": bool(manual["passed"]),
        "zero_threshold_frozen": True,
        "public_or_sealed_used": False,
    }
    distillation_allowed = all(distillation_gates.values())
    if not full:
        decision = (
            "GO_FULL_FOLDS_1_2_4" if all(screen_gates.values()) else "NO_GO_REJECT_TEACHER_SCREEN"
        )
    elif not manual["present"] and all(
        value for key, value in distillation_gates.items() if key != "manual_audit_passed"
    ):
        decision = "WAIT_MANUAL_AUDIT_NO_DISTILLATION"
    else:
        decision = "GO_DISTILLATION" if distillation_allowed else "NO_GO_DISTILLATION"
    result = {
        "schema_version": 1,
        "experiment_id": "645",
        "evaluation_version": "semantic_family_v3",
        "grid_contract_sha256": GRID_CONTRACT_SHA256,
        "model_input_view_sha256": input_view_sha256,
        "folds_evaluated": list(folds_to_evaluate),
        "screen_folds": list(SCREEN_FOLDS),
        "sealed_rows_loaded": 0,
        "public_used": False,
        "threshold": 0.0,
        "threshold_tuned": False,
        "primary_evidence_order": "evidence_first",
        "diagnostic_evidence_order": "class_first",
        "tracks": track_metrics,
        "comparisons": comparisons,
        "hard_cohort": hard_comparison,
        "evidence": evidence,
        "manual_audit": manual,
        "screen_gates": screen_gates,
        "distillation_gates": distillation_gates,
        "distillation_allowed": distillation_allowed,
        "decision": decision,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Qwen 2x3 scale grid.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, action="append", required=True)
    parser.add_argument("--track", action="append", required=True)
    parser.add_argument("--contract", action="append", required=True)
    parser.add_argument("--audit-manifest", type=Path)
    parser.add_argument("--audit-ratings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    tracks = _parse_assignments(args.track)
    contracts = _parse_assignments(args.contract)
    folds = FULL_FOLDS if args.full else SCREEN_FOLDS
    result = evaluate(
        registry_path=args.registry,
        runtime_paths=args.runtime,
        track_paths=tracks,
        contract_paths=contracts,
        output_path=args.output,
        folds_to_evaluate=folds,
        audit_manifest_path=args.audit_manifest,
        audit_ratings_path=args.audit_ratings,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

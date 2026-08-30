"""Fail-closed frozen two-fold screen for experiment 623.

The evaluator reuses experiment 621's exact experiment-600 baselines,
development registry, and donor-only thresholds.  Candidate artifacts remain
label-free until all integrity and grounding checks have passed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT_SCREEN_PATH = ROOT / "experiments/621_semantic_v3_fisher_blockwise_merge/evaluate_screen.py"
PROTOCOL_PATH = HERE / "protocol.py"
RUNNER_PATH = HERE / "run_fold.py"
EXPECTED_PARENT_RUNNER_SHA256 = "c30e690ad260af72fcc625c8d3e6d9ab9c5a096d8443d6d9f5f7adbcaa52123c"
EXPECTED_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
CANDIDATE_EXPERIMENT_ID = "623"
CANDIDATE_SEED = 42
RUNTIME_EXPERIMENT_ID = "623"
ARTIFACT_NAMES = (
    "adapter.zip",
    "auxiliary_head.safetensors",
    "auxiliary_head_config.json",
    "validation_predictions.csv",
    "selection_audit.json",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parent_screen = _load_module("_exp623_parent_screen", PARENT_SCREEN_PATH)
protocol = _load_module("_exp623_protocol", PROTOCOL_PATH)

FOLDS = parent_screen.FOLDS
SCREEN_FOLDS = parent_screen.SCREEN_FOLDS
CATEGORIES = parent_screen.CATEGORIES
FLAMMABLE = parent_screen.FLAMMABLE


def sha256_file(path: Path) -> str:
    return protocol.sha256_file(path)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path.name}")
    return value


def _verify_self_contract(contract: Mapping[str, Any]) -> None:
    expected = contract.get("contract_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("missing contract_sha256")
    payload = dict(contract)
    del payload["contract_sha256"]
    if canonical_sha256(payload) != expected:
        raise ValueError("contract_sha256 mismatch")


def _hard_random(
    indices: np.ndarray, scores: np.ndarray, count: int, rng: np.random.Generator
) -> list[int]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= count:
        return indices.tolist()
    hard_count = count // 2
    hard = indices[np.argsort(scores[indices])[:hard_count]]
    remaining = np.setdiff1d(indices, hard, assume_unique=False)
    random_part = rng.choice(remaining, size=count - hard_count, replace=False)
    return np.concatenate([hard, random_part]).tolist()


def _clone_rng(rng: np.random.Generator) -> np.random.Generator:
    clone = np.random.default_rng()
    clone.bit_generator.state = rng.bit_generator.state
    return clone


def _hard_random_tie_options(
    indices: np.ndarray,
    scores: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[list[int], np.random.Generator]]:
    """Enumerate selections allowed only by the parent's unstable argsort ties."""

    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= count:
        return [(indices.tolist(), _clone_rng(rng))]
    hard_count = count // 2
    values = scores[indices]
    boundary = np.partition(values, hard_count - 1)[hard_count - 1]
    fixed = indices[values < boundary]
    tied = indices[values == boundary]
    needed = hard_count - len(fixed)
    combinations = math.comb(len(tied), needed)
    if combinations > 256:
        raise ValueError("selector boundary has too many tie-equivalent branches")
    options: list[tuple[list[int], np.random.Generator]] = []
    for chosen in itertools.combinations(tied.tolist(), needed):
        branch = _clone_rng(rng)
        hard = np.concatenate([fixed, np.asarray(chosen, dtype=np.int64)])
        remaining = np.setdiff1d(indices, hard, assume_unique=False)
        random_part = branch.choice(remaining, size=count - hard_count, replace=False)
        options.append((np.concatenate([hard, random_part]).tolist(), branch))
    return options


def _expected_selected_ids(runtime_dir: Path, *, fold: int) -> list[str]:
    """Repeat the exact frozen experiment-600 hard selector without model imports."""

    train = protocol.read_jsonl(runtime_dir / "train.jsonl")
    validation = protocol.read_jsonl(runtime_dir / "validation.jsonl")
    ordered = sorted(train + validation, key=lambda row: int(row["row_index"]))
    if [int(row["row_index"]) for row in ordered] != list(range(len(ordered))):
        raise ValueError("runtime row indices are not a complete development scope")
    train_by_id = {str(row["id"]): row for row in train}
    if any("label" in row or "rationale" in row for row in validation):
        raise ValueError("validation runtime contains supervision")
    selector = np.load(runtime_dir / "development_selector_oof.npz", allow_pickle=False)
    required = {"ids", "fold_ids", "fused_scores"}
    if not required.issubset(selector.files):
        raise ValueError("frozen selector schema mismatch")
    ids = [str(value) for value in selector["ids"].tolist()]
    ordered_ids = [str(row["id"]) for row in ordered]
    if ids != ordered_ids:
        raise ValueError("frozen selector IDs/order differ from runtime")
    folds = selector["fold_ids"].astype(np.int8)
    if len(folds) != len(ordered) or not np.array_equal(folds, [row["development_fold"] for row in ordered]):
        raise ValueError("frozen selector folds differ from runtime")
    categories = np.asarray([str(row["category"]) for row in ordered])
    if set(categories) != set(CATEGORIES):
        raise ValueError("runtime category set mismatch")
    labels = np.asarray(
        [int(train_by_id[row_id]["label"]) if row_id in train_by_id else 0 for row_id in ids],
        dtype=np.int8,
    )
    scores = selector["fused_scores"].astype(np.float32)
    threshold_map = {"БАД": 0.24864045896205267, FLAMMABLE: 0.9591804083988902}
    thresholds = np.asarray([threshold_map[category] for category in categories], dtype=np.float32)
    uncertainty = np.abs(scores - thresholds)
    train_mask = folds != fold
    rng = np.random.default_rng(CANDIDATE_SEED)
    records: list[int] = []
    bad = np.flatnonzero(train_mask & (categories == "БАД"))
    bad_pos, bad_neg = bad[labels[bad] == 1], bad[labels[bad] == 0]
    bad_count = min(1500, len(bad_neg), len(bad_pos))
    records.extend(_hard_random(bad_pos, uncertainty, bad_count, rng))
    records.extend(_hard_random(bad_neg, uncertainty, bad_count, rng))
    flammable = np.flatnonzero(train_mask & (categories == FLAMMABLE))
    flam_pos, flam_neg = flammable[labels[flammable] == 1], flammable[labels[flammable] == 0]
    records.extend(np.repeat(flam_pos, 5).tolist())
    records.extend(_hard_random(flam_neg, uncertainty, min(1600, len(flam_neg)), rng))
    random.Random(CANDIDATE_SEED).shuffle(records)
    return [ids[index] for index in records]


def _expected_selected_multiset_hashes(runtime_dir: Path, *, fold: int) -> set[str]:
    """Return every exact-parent multiset allowed solely by boundary ties."""

    train = protocol.read_jsonl(runtime_dir / "train.jsonl")
    validation = protocol.read_jsonl(runtime_dir / "validation.jsonl")
    ordered = sorted(train + validation, key=lambda row: int(row["row_index"]))
    train_by_id = {str(row["id"]): row for row in train}
    selector = np.load(runtime_dir / "development_selector_oof.npz", allow_pickle=False)
    ids = [str(value) for value in selector["ids"].tolist()]
    folds = selector["fold_ids"].astype(np.int8)
    categories = np.asarray([str(row["category"]) for row in ordered])
    labels = np.asarray(
        [int(train_by_id[row_id]["label"]) if row_id in train_by_id else 0 for row_id in ids],
        dtype=np.int8,
    )
    scores = selector["fused_scores"].astype(np.float32)
    threshold_map = {"БАД": 0.24864045896205267, FLAMMABLE: 0.9591804083988902}
    thresholds = np.asarray(
        [threshold_map[category] for category in categories], dtype=np.float32
    )
    uncertainty = np.abs(scores - thresholds)
    train_mask = folds != fold
    bad = np.flatnonzero(train_mask & (categories == "БАД"))
    bad_pos, bad_neg = bad[labels[bad] == 1], bad[labels[bad] == 0]
    bad_count = min(1500, len(bad_neg), len(bad_pos))
    flammable = np.flatnonzero(train_mask & (categories == FLAMMABLE))
    flam_pos, flam_neg = flammable[labels[flammable] == 1], flammable[labels[flammable] == 0]
    stages = (
        (bad_pos, bad_count),
        (bad_neg, bad_count),
        (flam_neg, min(1600, len(flam_neg))),
    )
    states: list[tuple[list[int], np.random.Generator]] = [
        ([], np.random.default_rng(CANDIDATE_SEED))
    ]
    for stage_index, (indices, count) in enumerate(stages):
        expanded: list[tuple[list[int], np.random.Generator]] = []
        for records, rng in states:
            for selected, next_rng in _hard_random_tie_options(
                indices, uncertainty, count, rng
            ):
                current = records + selected
                if stage_index == 1:
                    current += np.repeat(flam_pos, 5).tolist()
                expanded.append((current, next_rng))
        if len(expanded) > 4096:
            raise ValueError("selector has too many combined tie-equivalent branches")
        states = expanded
    return {
        canonical_sha256(sorted(Counter(ids[index] for index in records).items()))
        for records, _ in states
    }


def _verify_runtime(runtime_dir: Path, *, fold: int, expected: pd.DataFrame) -> Path:
    audit_path = runtime_dir / "runtime_audit.json"
    validation_path = runtime_dir / "validation.jsonl"
    audit = _load_json(audit_path)
    expected_fields = {
        "experiment_id": RUNTIME_EXPERIMENT_ID,
        "outer_fold": fold,
        "validation_rows": len(expected),
        "validation_label_columns": [],
        "validation_rationale_columns": [],
        "train_validation_overlap": 0,
        "sealed_rows_written": 0,
        "decision": "GO",
        "frozen_input_hashes_enforced": True,
    }
    for key, value in expected_fields.items():
        if audit.get(key) != value:
            raise ValueError(f"runtime fold {fold} audit mismatch: {key}")
    if audit.get("input_sha256") != protocol.FROZEN_INPUT_SHA256:
        raise ValueError(f"runtime fold {fold} frozen input hashes mismatch")
    for name in (
        "train.jsonl",
        "validation.jsonl",
        "development_selector_oof.npz",
        "development_image_manifest.tsv.gz",
    ):
        path = runtime_dir / name
        if audit.get("output_sha256", {}).get(name) != sha256_file(path):
            raise ValueError(f"runtime fold {fold} checksum mismatch: {name}")
    rows = protocol.read_jsonl(validation_path)
    if [str(row["id"]) for row in rows] != expected["id"].astype(str).tolist():
        raise ValueError(f"runtime fold {fold} validation IDs/order mismatch")
    if [str(row["category"]) for row in rows] != expected["category"].astype(str).tolist():
        raise ValueError(f"runtime fold {fold} validation categories/order mismatch")
    if any(set(row) & protocol.FORBIDDEN_LABEL_COLUMNS or "rationale" in row for row in rows):
        raise ValueError(f"runtime fold {fold} validation contains supervision")
    return validation_path


def _load_candidate_fold(
    artifact_dir: Path,
    runtime_dir: Path,
    *,
    fold: int,
    expected: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    validation_path = _verify_runtime(runtime_dir, fold=fold, expected=expected)
    contract_path = artifact_dir / "output_contract.json"
    contract = _load_json(contract_path)
    _verify_self_contract(contract)
    expected_contract = {
        "experiment_id": CANDIDATE_EXPERIMENT_ID,
        "outer_fold": fold,
        "seed": CANDIDATE_SEED,
        "validation_rows": len(expected),
        "download_failures": 0,
        "parent_sha256": EXPECTED_PARENT_RUNNER_SHA256,
        "validation_labels_written": 0,
        "sealed_rows_used": 0,
        "decision": "GO",
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise ValueError(f"candidate fold {fold} contract mismatch: {key}")
    if set(contract.get("artifacts", {})) != set(ARTIFACT_NAMES):
        raise ValueError(f"candidate fold {fold} artifact manifest mismatch")
    for name in ARTIFACT_NAMES:
        path = artifact_dir / name
        if contract["artifacts"].get(name) != sha256_file(path):
            raise ValueError(f"candidate fold {fold} artifact checksum mismatch: {name}")
    prediction_path = artifact_dir / "validation_predictions.csv"
    prediction_audit = protocol.validate_label_free_predictions(
        prediction_path=prediction_path,
        validation_path=validation_path,
        outer_fold=fold,
    )
    selection_path = artifact_dir / "selection_audit.json"
    selection = _load_json(selection_path)
    expected_ids = _expected_selected_ids(runtime_dir, fold=fold)
    expected_selection = {
        "parent_sha256": EXPECTED_PARENT_RUNNER_SHA256,
        "model_revision": EXPECTED_MODEL_REVISION,
        "prediction_audit": prediction_audit,
        "outer_fold": fold,
        "training_records": len(expected_ids),
        "training_unique_rows": len(set(expected_ids)),
        "outer_validation_occurrences": 0,
        "runtime_audit_sha256": sha256_file(runtime_dir / "runtime_audit.json"),
        "decision": "GO",
    }
    observed_multiset = selection.get("selected_id_multiset_sha256")
    selection_without_multiset = dict(selection)
    selection_without_multiset.pop("selected_id_multiset_sha256", None)
    if selection_without_multiset != expected_selection:
        raise ValueError(f"candidate fold {fold} selection audit mismatch")
    if observed_multiset not in _expected_selected_multiset_hashes(runtime_dir, fold=fold):
        raise ValueError(f"candidate fold {fold} selection multiset is not tie-equivalent")
    frame = pd.read_csv(prediction_path, dtype={"id": str, "category": str})
    return frame, {
        "contract_sha256": sha256_file(contract_path),
        "predictions_sha256": sha256_file(prediction_path),
        "selection_audit_sha256": sha256_file(selection_path),
    }, prediction_audit


def evaluate_screen(
    *,
    registry_path: Path,
    baseline_dirs: Mapping[int, Path],
    runtime_dirs: Mapping[int, Path],
    artifact_dirs: Mapping[int, Path],
    threshold_contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite screen report")
    if set(baseline_dirs) != set(FOLDS):
        raise ValueError("exactly baseline folds 0..4 are required")
    if set(runtime_dirs) != set(SCREEN_FOLDS) or set(artifact_dirs) != set(SCREEN_FOLDS):
        raise ValueError("screen requires exactly runtime and artifact folds 0 and 3")
    thresholds, baseline_frames = parent_screen._verify_threshold_contract(
        contract_path=threshold_contract_path,
        registry_path=registry_path,
        baseline_dirs=baseline_dirs,
    )
    registry = parent_screen._registry(registry_path)
    candidate_frames: dict[int, pd.DataFrame] = {}
    candidate_provenance: dict[str, Any] = {}
    prediction_audits: dict[str, Any] = {}
    for fold in SCREEN_FOLDS:
        expected = parent_screen._expected_fold(registry, fold)
        candidate_frames[fold], candidate_provenance[str(fold)], prediction_audits[str(fold)] = (
            _load_candidate_fold(
                artifact_dirs[fold], runtime_dirs[fold], fold=fold, expected=expected
            )
        )

    rows: list[pd.DataFrame] = []
    fold_metrics: dict[str, Any] = {}
    coverage_folds: dict[str, Any] = {}
    for fold in SCREEN_FOLDS:
        expected = parent_screen._expected_fold(registry, fold)
        baseline = baseline_frames[fold]
        candidate = candidate_frames[fold]
        assembled = expected.copy()
        assembled["baseline_score"] = baseline["lora_score"].to_numpy(np.float64)
        assembled["candidate_score"] = candidate["lora_score"].to_numpy(np.float64)
        baseline_pred = np.zeros(len(assembled), dtype=np.int8)
        candidate_pred = np.zeros(len(assembled), dtype=np.int8)
        categories: dict[str, Any] = {}
        coverage_categories: dict[str, Any] = {}
        for category in CATEGORIES:
            mask = assembled["category"].eq(category).to_numpy()
            threshold = float(thresholds["thresholds"][str(fold)][category]["threshold"])
            baseline_pred[mask] = (assembled.loc[mask, "baseline_score"] >= threshold).astype(np.int8)
            candidate_pred[mask] = (assembled.loc[mask, "candidate_score"] >= threshold).astype(np.int8)
            labels = assembled.loc[mask, "label"].to_numpy(np.int8)
            base_f1 = parent_screen._f1(labels, baseline_pred[mask])
            cand_f1 = parent_screen._f1(labels, candidate_pred[mask])
            categories[category] = {
                "baseline_f1": base_f1,
                "candidate_f1": cand_f1,
                "delta": cand_f1 - base_f1,
                "threshold": threshold,
            }
            grounded = candidate.loc[mask, "evidence"].astype(str).ne("NO_EVIDENCE")
            coverage_categories[category] = {
                "rows": int(mask.sum()),
                "grounded_rows": int(grounded.sum()),
                "grounded_coverage": float(grounded.mean()) if mask.any() else None,
            }
        assembled["baseline_prediction"] = baseline_pred
        assembled["candidate_prediction"] = candidate_pred
        fold_metrics[str(fold)] = {
            "rows": len(assembled),
            "baseline_macro_f1": float(np.mean([categories[c]["baseline_f1"] for c in CATEGORIES])),
            "candidate_macro_f1": float(np.mean([categories[c]["candidate_f1"] for c in CATEGORIES])),
            "categories": categories,
        }
        fold_metrics[str(fold)]["delta"] = (
            fold_metrics[str(fold)]["candidate_macro_f1"]
            - fold_metrics[str(fold)]["baseline_macro_f1"]
        )
        fold_grounded = candidate["evidence"].astype(str).ne("NO_EVIDENCE")
        coverage_folds[str(fold)] = {
            "rows": len(candidate),
            "grounded_rows": int(fold_grounded.sum()),
            "grounded_coverage": float(fold_grounded.mean()),
            "categories": coverage_categories,
        }
        assembled["fold"] = fold
        rows.append(assembled)

    combined = pd.concat(rows, ignore_index=True)
    labels = combined["label"].to_numpy(np.int8)
    baseline_pred = combined["baseline_prediction"].to_numpy(np.int8)
    candidate_pred = combined["candidate_prediction"].to_numpy(np.int8)
    corrected = int(((baseline_pred != labels) & (candidate_pred == labels)).sum())
    regressed = int(((baseline_pred == labels) & (candidate_pred != labels)).sum())
    category_metrics: dict[str, Any] = {}
    for category in CATEGORIES:
        mask = combined["category"].eq(category).to_numpy()
        base_f1 = parent_screen._f1(labels[mask], baseline_pred[mask])
        cand_f1 = parent_screen._f1(labels[mask], candidate_pred[mask])
        category_metrics[category] = {
            "baseline_f1": base_f1,
            "candidate_f1": cand_f1,
            "delta": cand_f1 - base_f1,
        }
    positive = labels == 1
    flammable = combined["category"].eq(FLAMMABLE).to_numpy()
    false_negatives = {
        "flammable": {
            "baseline": int((flammable & positive & (baseline_pred == 0)).sum()),
            "candidate": int((flammable & positive & (candidate_pred == 0)).sum()),
        },
        "all_positive": {
            "baseline": int((positive & (baseline_pred == 0)).sum()),
            "candidate": int((positive & (candidate_pred == 0)).sum()),
        },
    }
    for value in false_negatives.values():
        value["delta"] = value["candidate"] - value["baseline"]
    mean_fold_delta = float(np.mean([fold_metrics[str(fold)]["delta"] for fold in SCREEN_FOLDS]))
    gates = {
        "each_fold_delta_gt_0": all(fold_metrics[str(fold)]["delta"] > 0 for fold in SCREEN_FOLDS),
        "mean_fold_delta_at_least_0_0015": mean_fold_delta >= 0.0015,
        "corrected_to_regressed_at_least_1_5": corrected > 0 if regressed == 0 else corrected / regressed >= 1.5,
        "no_category_drop_below_minus_0_002": all(category_metrics[c]["delta"] >= -0.002 for c in CATEGORIES),
        "flammable_false_negatives_do_not_increase": false_negatives["flammable"]["delta"] <= 0,
        "all_positive_false_negatives_do_not_increase": false_negatives["all_positive"]["delta"] <= 0,
    }
    passed = all(gates.values())
    all_candidates = pd.concat([candidate_frames[fold] for fold in SCREEN_FOLDS], ignore_index=True)
    coverage_categories = {}
    for category in CATEGORIES:
        selected = all_candidates.loc[all_candidates["category"].eq(category)]
        grounded = selected["evidence"].astype(str).ne("NO_EVIDENCE")
        coverage_categories[category] = {
            "rows": len(selected),
            "grounded_rows": int(grounded.sum()),
            "grounded_coverage": float(grounded.mean()),
        }
    grounded = all_candidates["evidence"].astype(str).ne("NO_EVIDENCE")
    result: dict[str, Any] = {
        "protocol": f"{CANDIDATE_EXPERIMENT_ID}_frozen_donor_screen_v1",
        "screen_folds": list(SCREEN_FOLDS),
        "registry_sha256": sha256_file(registry_path),
        "threshold_contract_sha256": sha256_file(threshold_contract_path),
        "thresholds_tuned_after_candidate": False,
        "candidate_predictions_contained_labels": False,
        "sealed_rows_loaded": 0,
        "folds": fold_metrics,
        "mean_fold_delta": mean_fold_delta,
        "categories": category_metrics,
        "corrected": corrected,
        "regressed": regressed,
        "corrected_to_regressed": None if regressed == 0 else corrected / regressed,
        "corrected_to_regressed_infinite": regressed == 0 and corrected > 0,
        "false_negatives": false_negatives,
        "grounding": {
            "metric": "structural_exact_substring_coverage",
            "overall": {
                "rows": len(all_candidates),
                "grounded_rows": int(grounded.sum()),
                "grounded_coverage": float(grounded.mean()),
            },
            "categories": coverage_categories,
            "folds": coverage_folds,
            "human_quality_evaluated": False,
            "human_quality": None,
        },
        "gates": gates,
        "passed": passed,
        "decision": (
            "GO_LAUNCH_FOLDS_1_2_4"
            if passed
            else f"NO_GO_REJECT_{CANDIDATE_EXPERIMENT_ID}"
        ),
        "candidate_provenance": candidate_provenance,
        "prediction_audits": prediction_audits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _fold_paths(values: Sequence[str], *, required: tuple[int, ...]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        raw_fold, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError("fold path must use FOLD=PATH")
        fold = int(raw_fold)
        if fold in parsed:
            raise ValueError(f"duplicate fold path: {fold}")
        parsed[fold] = Path(raw_path)
    if set(parsed) != set(required):
        raise ValueError(f"required folds are {required}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen experiment-623 screen.")
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--baseline-fold", action="append", required=True)
    parser.add_argument("--runtime-fold", action="append", required=True)
    parser.add_argument("--artifact-fold", action="append", required=True)
    parser.add_argument("--threshold-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_screen(
        registry_path=args.registry,
        baseline_dirs=_fold_paths(args.baseline_fold, required=FOLDS),
        runtime_dirs=_fold_paths(args.runtime_fold, required=SCREEN_FOLDS),
        artifact_dirs=_fold_paths(args.artifact_fold, required=SCREEN_FOLDS),
        threshold_contract_path=args.threshold_contract,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

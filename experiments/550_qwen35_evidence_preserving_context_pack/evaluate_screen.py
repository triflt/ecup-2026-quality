from __future__ import annotations

"""Evaluate the immutable two-fold context-packing screen against route 400."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from contract import (
    AUDIT_ROWS_SHA256,
    AUDIT_SHA256,
    AUDIT_VERSION,
    DESCRIPTION_BUDGET,
    EXTRACTOR_SHA256,
    PARENT_ENVIRONMENT,
    SCREEN_FOLDS,
    VOCABULARY_SHA256,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
BASE_EVALUATOR_PATH = ROOT / "experiments/440_qwen3vl_rdrop/evaluate_screen.py"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "results"


def _load_base_evaluator():
    spec = importlib.util.spec_from_file_location(
        "_locked_route400_evaluator_for_exp550", BASE_EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError("locked route-400 evaluator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_evaluator()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_context_report(path: Path, *, fold: int) -> dict[str, object]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"fold {fold} context-pack report is missing") from error
    expected = {
        "experiment_id": "550",
        "status": "screen_fold_complete",
        "holdout_fold": fold,
        "single_changed_factor": "description packing only",
        "description_budget": DESCRIPTION_BUDGET,
        "packing_audit_version": AUDIT_VERSION,
        "packing_audit_sha256": AUDIT_SHA256,
        "packing_rows_sha256": AUDIT_ROWS_SHA256,
        "extractor_sha256": EXTRACTOR_SHA256,
        "vocabulary_sha256": VOCABULARY_SHA256,
        "selection_uses_labels": False,
        "selection_uses_folds": False,
        "prediction_source": "constant union of frozen_prediction=0 and 1",
        "parent_environment": PARENT_ENVIRONMENT,
        "parent_train_records": 5390,
        "parent_optimizer_updates": 337,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(
                f"fold {fold} context-pack report has {key}={report.get(key)!r}; expected {value!r}"
            )
    if int(report.get("unique_runtime_rows_packed", 0)) <= 0:
        raise ValueError(f"fold {fold} context-pack report has no runtime rows")
    parent_report_path = path.with_name("lora_holdout_report.json")
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    if (
        report.get("parent_holdout_report_sha256") != sha256(parent_report_path)
        or parent_report.get("holdout_fold") != fold
        or parent_report.get("train_records") != 5390
        or parent_report.get("download_failures") != 0
    ):
        raise ValueError(f"fold {fold} parent holdout report contract failed")
    adapter_config_path = path.with_name("adapter") / "adapter_config.json"
    adapter = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if not (
        adapter.get("r") == 16
        and adapter.get("lora_alpha") == 32
        and adapter.get("lora_dropout") == 0.05
        and adapter.get("use_rslora") is True
        and set(adapter.get("target_modules", [])) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    ):
        raise ValueError(f"fold {fold} adapter differs from the parent LoRA recipe")
    return expected


def load_candidate_logits(
    *,
    paths: dict[int, Path],
    ids: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    parent_logits: np.ndarray,
) -> tuple[np.ndarray, dict[str, str]]:
    hybrid = parent_logits.copy()
    checksums: dict[str, str] = {}
    for fold, path in paths.items():
        frame = pd.read_csv(path, dtype={"id": str})
        required = {"id", "category", "label", "fold", "lora_score"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"fold {fold} predictions lack columns: {missing}")
        expected = np.flatnonzero(folds == fold)
        if not np.array_equal(frame.id.astype(str).to_numpy(), ids[expected]):
            raise ValueError(f"fold {fold} prediction id/order mismatch")
        if not np.array_equal(frame.label.to_numpy(np.int8), labels[expected]):
            raise ValueError(f"fold {fold} prediction label mismatch")
        if not np.array_equal(frame.category.astype(str).to_numpy(), categories[expected]):
            raise ValueError(f"fold {fold} prediction category mismatch")
        if not (frame.fold.to_numpy(np.int8) == fold).all():
            raise ValueError(f"fold {fold} prediction file contains another fold")
        scores = frame.lora_score.to_numpy(np.float32)
        if not np.isfinite(scores).all():
            raise ValueError(f"fold {fold} prediction scores are not finite")
        flammable = categories[expected] == base.FLAMMABLE
        hybrid[expected[flammable]] = scores[flammable]
        checksums[f"fold_{fold}_predictions"] = sha256(path)
    return hybrid, checksums


def output_paths(output_dir: Path, *, null_control: bool) -> tuple[Path, Path]:
    stem = "null_screen_control" if null_control else "screen"
    return output_dir / f"{stem}_audit.json", output_dir / f"{stem}_predictions.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-0", type=Path)
    parser.add_argument("--fold-0-report", type=Path)
    parser.add_argument("--fold-3", type=Path)
    parser.add_argument("--fold-3-report", type=Path)
    parser.add_argument("--null-control", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    supplied = {
        0: (args.fold_0, args.fold_0_report),
        3: (args.fold_3, args.fold_3_report),
    }
    if args.null_control:
        if any(value is not None for pair in supplied.values() for value in pair):
            raise ValueError("null control does not accept candidate files")
    elif any(prediction is None for prediction, _ in supplied.values()):
        raise ValueError("candidate screen requires both folds 0 and 3")
    targets = output_paths(args.output_dir, null_control=args.null_control)
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite existing screen outputs")

    source = np.load(base.BASE, allow_pickle=True)
    qwen3vl = np.load(base.QWEN3VL, allow_pickle=True)
    original = np.load(base.QWEN35_SEED_A, allow_pickle=True)
    ids = source["ids"].astype(str)
    labels = source["labels"].astype(np.int8)
    categories = source["categories"].astype(str)
    folds = source["fold_ids"].astype(np.int8)
    for name, component in (("qwen3vl", qwen3vl), ("qwen35", original)):
        if not np.array_equal(component["ids"].astype(str), ids):
            raise ValueError(f"{name} id mismatch")
        if not np.array_equal(component["folds"].astype(np.int8), folds):
            raise ValueError(f"{name} fold mismatch")

    parent_paths = [
        base.PARENT_QWEN35 / f"fold_{fold}/lora_holdout_predictions.csv" for fold in range(5)
    ]
    parent_logits = base.load_seed_predictions(parent_paths, source)
    context_contracts: dict[str, dict[str, object]] = {}
    input_checksums: dict[str, str] = {}
    if args.null_control:
        hybrid_logits = parent_logits.copy()
    else:
        candidate_paths: dict[int, Path] = {}
        for fold, (prediction_path, explicit_report_path) in supplied.items():
            assert prediction_path is not None
            report_path = explicit_report_path or prediction_path.with_name(
                "context_pack_runtime_report.json"
            )
            context_contracts[str(fold)] = validate_context_report(report_path, fold=fold)
            candidate_paths[fold] = prediction_path
            input_checksums[f"fold_{fold}_context_report"] = sha256(report_path)
        hybrid_logits, prediction_checksums = load_candidate_logits(
            paths=candidate_paths,
            ids=ids,
            labels=labels,
            categories=categories,
            folds=folds,
            parent_logits=parent_logits,
        )
        input_checksums.update(prediction_checksums)

    parent_rank = base.fold_category_ranks(parent_logits, folds, categories)
    hybrid_rank = base.fold_category_ranks(hybrid_logits, folds, categories)
    original_rank = original["lora_rank"].astype(np.float32)
    baseline_qwen35 = np.where(categories == base.FLAMMABLE, parent_rank, original_rank).astype(
        np.float32
    )
    candidate_qwen35 = np.where(categories == base.FLAMMABLE, hybrid_rank, original_rank).astype(
        np.float32
    )
    base_rank = original["base_rank"].astype(np.float32)
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)
    baseline, _, baseline_detail = base.evaluate_locked(
        np.column_stack([base_rank, qwen3vl_rank, baseline_qwen35]),
        labels,
        categories,
        folds,
    )
    candidate, _, candidate_detail = base.evaluate_locked(
        np.column_stack([base_rank, qwen3vl_rank, candidate_qwen35]),
        labels,
        categories,
        folds,
    )
    frozen = np.load(base.ROUTE400, allow_pickle=False)
    if not np.array_equal(frozen["ids"].astype(str), ids):
        raise ValueError("route-400 id mismatch")
    if not np.array_equal(frozen["category_routed_nested_predictions"].astype(np.int8), baseline):
        raise ValueError("reconstructed baseline differs from route 400")

    guard = pd.read_csv(base.GUARD, dtype={"id": str})
    data = pd.read_csv(base.DATA, dtype={"id": str})
    if not np.array_equal(guard.id.astype(str).to_numpy(), ids):
        raise ValueError("connected guard id mismatch")
    if not np.array_equal(data.id.astype(str).to_numpy(), ids):
        raise ValueError("data id mismatch")
    safe = guard.safe_for_selection.astype(bool).to_numpy()
    text = data.name.fillna("").astype(str) + "\n" + data.description.fillna("").astype(str)
    safety_union = (
        (categories == base.FLAMMABLE)
        & (labels == 1)
        & text.str.contains(base.SAFETY_PATTERN, na=False).to_numpy()
    )
    fold_reports = []
    for fold in SCREEN_FOLDS:
        report = base.fold_audit(
            fold=fold,
            labels=labels,
            categories=categories,
            folds=folds,
            baseline=baseline,
            candidate=candidate,
            safe=safe,
            safety_union=safety_union,
        )
        report["gates"] = {
            "macro_delta_positive": report["delta_macro_f1"] > 0,
            "bad_predictions_unchanged": report["category_delta"][base.BAD] == 0,
            "flammable_false_negatives_not_increased": (
                report["flammable_false_negatives"]["delta"] <= 0
            ),
            "safety_union_false_negatives_not_increased": (
                report["safety_union_false_negatives"]["delta"] <= 0
            ),
            "connected_safe_delta_positive": report["connected_safe_delta_macro_f1"] > 0,
            "corrected_exceeds_regressed": report["corrected"] > report["regressed"],
        }
        report["passed"] = all(report["gates"].values())
        fold_reports.append(report)
    mean_delta = float(np.mean([report["delta_macro_f1"] for report in fold_reports]))
    null_passed = bool(np.array_equal(candidate, baseline))
    screen_passed = (
        not args.null_control
        and all(report["passed"] for report in fold_reports)
        and mean_delta >= 0.001
    )
    result = {
        "experiment_id": "550",
        "evaluation_version": "evidence_preserving_context_pack_screen_v1",
        "mode": "null_control" if args.null_control else "candidate",
        "purpose": "Reject-only screen; passing cannot authorize submission",
        "screen_folds": list(SCREEN_FOLDS),
        "single_changed_factor": "Qwen3.5 description packing only",
        "context_report_contracts": context_contracts,
        "packing_selection_uses_labels": False,
        "packing_selection_uses_folds": False,
        "folds": fold_reports,
        "mean_screen_delta_macro_f1": mean_delta,
        "required_mean_delta_macro_f1": 0.001,
        "bad_changed_predictions": int(((candidate != baseline) & (categories == base.BAD)).sum()),
        "screen_passed": screen_passed,
        "full_five_fold_cycle_allowed": screen_passed,
        "null_control_passed": null_passed if args.null_control else None,
        "baseline_locked_detail": baseline_detail,
        "candidate_locked_detail": candidate_detail,
        "input_sha256": {
            **input_checksums,
            "context_pack_audit": sha256(EXPERIMENT_DIR / "analysis/context_pack_audit.json"),
            "context_pack_rows": sha256(EXPERIMENT_DIR / "analysis/context_pack_rows.csv.gz"),
            "route400_predictions": sha256(base.ROUTE400),
            "connected_guard": sha256(base.GUARD),
            "data": sha256(base.DATA),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        targets[1],
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        baseline_predictions=baseline,
        candidate_predictions=candidate,
        baseline_qwen35_rank=baseline_qwen35,
        candidate_qwen35_rank=candidate_qwen35,
    )
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "mean_delta": mean_delta,
                "changed": int((candidate != baseline).sum()),
                "screen_passed": screen_passed,
                "null_control_passed": result["null_control_passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

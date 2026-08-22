from __future__ import annotations

"""Evaluate the locked two-fold InternVL rank-replacement screen."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from contract import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    EXPECTED_BASELINE_MACRO,
    FROZEN_THRESHOLDS,
    IMAGE_SIZE,
    LORA_RANK,
    OPTIMIZER_UPDATES,
    RANK_REPLACEMENT_WEIGHT,
    SCREEN_FOLDS,
    SEED,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
BASE_EVALUATOR_PATH = ROOT / "experiments/440_qwen3vl_rdrop/evaluate_screen.py"
MEMBERSHIP = EXPERIMENT_DIR / "analysis/selector_membership.csv"
PAIR_AUDIT = EXPERIMENT_DIR / "analysis/pair_manifest_audit.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "results"


def _load_base_evaluator():
    spec = importlib.util.spec_from_file_location(
        "_locked_route400_evaluator_for_exp530", BASE_EVALUATOR_PATH
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


def donor_cdf_rank(donor_scores: np.ndarray, outer_scores: np.ndarray) -> np.ndarray:
    donor = np.asarray(donor_scores, dtype=np.float64)
    outer = np.asarray(outer_scores, dtype=np.float64)
    if donor.ndim != 1 or outer.ndim != 1 or not len(donor):
        raise ValueError("donor and outer scores must be non-empty one-dimensional arrays")
    if not np.isfinite(donor).all() or not np.isfinite(outer).all():
        raise ValueError("InternVL scores must be finite")
    ordered = np.sort(donor, kind="mergesort")
    return (np.searchsorted(ordered, outer, side="right") / float(len(ordered))).astype(np.float32)


def strict_boolean(values: pd.Series, *, name: str) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized) - {"true", "false"})
    if unexpected:
        raise ValueError(f"invalid boolean values in {name}: {unexpected}")
    return normalized == "true"


def validate_training_report(
    path: Path, *, fold: int, pair_audit: dict[str, object]
) -> dict[str, object]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"fold {fold} training report is missing") from error
    expected = {
        "experiment_id": "530",
        "status": "screen_fold_complete",
        "holdout_fold": fold,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "image_tiles": 1,
        "image_source": "manifest",
        "lora_rank": LORA_RANK,
        "language_attention_only": True,
        "vision_frozen": True,
        "atomic_digit_tokens": True,
        "loss": "softplus(-(score_positive-score_negative))",
        "pair_manifest_sha256": pair_audit["folds"][str(fold)]["file_sha256"],
        "selector_membership_sha256": pair_audit["output_sha256"]["selector_membership"],
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(
                f"fold {fold} training report has {key}={report.get(key)!r}; expected {value!r}"
            )
    image_report_path = path.with_name("image_download_report.json")
    try:
        image_report = json.loads(image_report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"fold {fold} image download report is missing") from error
    if report.get("image_download_report_sha256") != sha256(image_report_path):
        raise ValueError(f"fold {fold} image download report checksum mismatch")
    if report.get("image_download_report") != image_report:
        raise ValueError(f"fold {fold} embedded image download report mismatch")
    expected_images = int(image_report.get("expected_images", -1))
    image_gates = {
        "source": image_report.get("image_source") == "manifest",
        "ready": image_report.get("ready") is True,
        "no_failures": image_report.get("failed_images") == 0,
        "no_white_fallbacks": image_report.get("white_fallbacks") == 0,
        "decoded_all": image_report.get("decoded_rgb_images") == expected_images,
        "id_hash_match": (
            isinstance(image_report.get("expected_ids_sha256"), str)
            and len(image_report["expected_ids_sha256"]) == 64
            and image_report.get("prepared_ids_sha256") == image_report.get("expected_ids_sha256")
        ),
        "covered_all": (
            int(image_report.get("cache_hits", -1)) + int(image_report.get("downloaded_images", -1))
            == expected_images
        ),
        "one_per_id": image_report.get("exactly_one_image_per_expected_id") is True,
    }
    if expected_images <= 0 or not all(image_gates.values()):
        raise ValueError(f"fold {fold} image preparation gates failed")
    return expected


def load_fold_rank(
    *,
    fold: int,
    predictions_path: Path,
    membership: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(predictions_path, dtype={"id": str})
    required = {"id", "fold", "row_role", "internvl_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fold {fold} score file lacks columns: {missing}")
    if frame.id.duplicated().any():
        raise ValueError(f"fold {fold} score file contains duplicate ids")
    expected_donor = set(
        membership.loc[
            (membership.fold.astype(int) != fold)
            & strict_boolean(membership.safe_for_selection, name="safe_for_selection"),
            "id",
        ].astype(str)
    )
    expected_outer = set(membership.loc[membership.fold.astype(int) == fold, "id"].astype(str))
    donor = frame.loc[frame.row_role == "donor"]
    outer = frame.loc[frame.row_role == "outer"]
    unexpected_roles = sorted(set(frame.row_role.astype(str)) - {"donor", "outer"})
    if unexpected_roles:
        raise ValueError(f"fold {fold} score file has invalid roles: {unexpected_roles}")
    if set(donor.id) != expected_donor or set(outer.id) != expected_outer:
        raise ValueError(f"fold {fold} donor/outer membership mismatch")
    membership_fold = membership.set_index("id").fold.astype(int)
    if not np.array_equal(
        frame.fold.to_numpy(np.int8), membership_fold.loc[frame.id].to_numpy(np.int8)
    ):
        raise ValueError(f"fold {fold} score file contains incorrect source folds")
    donor_scores = donor.internvl_score.to_numpy(np.float64)
    outer_scores = outer.internvl_score.to_numpy(np.float64)
    ranks = donor_cdf_rank(donor_scores, outer_scores)
    return outer.id.astype(str).to_numpy(), ranks


def component_bootstrap_probability(
    *,
    labels: np.ndarray,
    categories: np.ndarray,
    components: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    grouped: dict[str, list[np.ndarray]] = {}
    for category in sorted(np.unique(categories[mask])):
        positions_by_component: dict[str, list[int]] = {}
        for position in np.flatnonzero(mask & (categories == category)):
            positions_by_component.setdefault(str(components[position]), []).append(position)
        grouped[category] = [
            np.asarray(positions, dtype=np.int64) for positions in positions_by_component.values()
        ]
    if not grouped or any(not groups for groups in grouped.values()):
        raise ValueError("component bootstrap has an empty category")
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        old_values: list[float] = []
        new_values: list[float] = []
        for category in sorted(grouped):
            groups = grouped[category]
            sampled = rng.integers(0, len(groups), size=len(groups))
            positions = np.concatenate([groups[index] for index in sampled])
            old_values.append(base.f1(labels[positions], baseline[positions]))
            new_values.append(base.f1(labels[positions], candidate[positions]))
        deltas[iteration] = float(np.mean(new_values) - np.mean(old_values))
    return {
        "iterations": iterations,
        "seed": seed,
        "probability_delta_positive": float((deltas > 0).mean()),
        "delta_ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


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
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    supplied = {
        0: (args.fold_0, args.fold_0_report),
        3: (args.fold_3, args.fold_3_report),
    }
    if args.bootstrap <= 0:
        raise ValueError("bootstrap count must be positive")
    if args.null_control:
        if any(value is not None for pair in supplied.values() for value in pair):
            raise ValueError("null control does not accept candidate files")
    elif any(prediction is None for prediction, _ in supplied.values()):
        raise ValueError("candidate screen requires both frozen folds 0 and 3")
    if any(prediction is None and report is not None for prediction, report in supplied.values()):
        raise ValueError("training report supplied without scores")
    targets = output_paths(args.output_dir, null_control=args.null_control)
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing screen outputs")

    source = np.load(base.BASE, allow_pickle=True)
    qwen3vl = np.load(base.QWEN3VL, allow_pickle=True)
    original_qwen35 = np.load(base.QWEN35_SEED_A, allow_pickle=True)
    ids = source["ids"].astype(str)
    labels = source["labels"].astype(np.int8)
    categories = source["categories"].astype(str)
    folds = source["fold_ids"].astype(np.int8)
    for name, candidate_source in (("qwen3vl", qwen3vl), ("qwen35", original_qwen35)):
        if not np.array_equal(ids, candidate_source["ids"].astype(str)):
            raise ValueError(f"{name} id mismatch")
        if not np.array_equal(folds, candidate_source["folds"].astype(np.int8)):
            raise ValueError(f"{name} fold mismatch")

    parent_paths = [
        base.PARENT_QWEN35 / f"fold_{fold}/lora_holdout_predictions.csv" for fold in range(5)
    ]
    parent_logits = base.load_seed_predictions(parent_paths, source)
    parent_qwen35_rank = base.fold_category_ranks(parent_logits, folds, categories)
    original_qwen35_rank = original_qwen35["lora_rank"].astype(np.float32)
    routed_qwen35_rank = np.where(
        categories == base.FLAMMABLE, parent_qwen35_rank, original_qwen35_rank
    ).astype(np.float32)
    parent_qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)
    route_score = (
        np.float32(0.15) * original_qwen35["base_rank"].astype(np.float32)
        + np.float32(0.10) * parent_qwen3vl_rank
        + np.float32(0.75) * routed_qwen35_rank
    )
    frozen = np.load(base.ROUTE400, allow_pickle=False)
    if not np.array_equal(frozen["ids"].astype(str), ids):
        raise ValueError("frozen route-400 id mismatch")
    baseline = frozen["category_routed_nested_predictions"].astype(np.int8)
    candidate = baseline.copy()
    for fold in SCREEN_FOLDS:
        mask = (folds == fold) & (categories == base.FLAMMABLE)
        reconstructed = (route_score[mask] >= FROZEN_THRESHOLDS[fold]).astype(np.int8)
        if not np.array_equal(reconstructed, baseline[mask]):
            raise ValueError(f"fold {fold} fixed threshold does not reconstruct route-400")

    membership = pd.read_csv(MEMBERSHIP, dtype={"id": str})
    if membership.id.duplicated().any() or not set(membership.id) <= set(ids):
        raise ValueError("selector membership identity invariant failed")
    pair_audit = json.loads(PAIR_AUDIT.read_text(encoding="utf-8"))
    if sha256(MEMBERSHIP) != pair_audit["output_sha256"]["selector_membership"]:
        raise ValueError("selector membership checksum mismatch")
    id_position = {item_id: position for position, item_id in enumerate(ids)}
    report_contracts: dict[str, dict[str, object]] = {}
    input_checksums: dict[str, str] = {}
    candidate_rank = parent_qwen3vl_rank.copy()
    if not args.null_control:
        for fold, (predictions_path, explicit_report_path) in supplied.items():
            assert predictions_path is not None
            report_path = explicit_report_path or predictions_path.with_name("training_report.json")
            report_contracts[str(fold)] = validate_training_report(
                report_path, fold=fold, pair_audit=pair_audit
            )
            outer_ids, ranks = load_fold_rank(
                fold=fold,
                predictions_path=predictions_path,
                membership=membership,
            )
            positions = np.asarray([id_position[item_id] for item_id in outer_ids])
            if not ((folds[positions] == fold) & (categories[positions] == base.FLAMMABLE)).all():
                raise ValueError(f"fold {fold} selector outer rows do not align")
            candidate_rank[positions] = ranks
            input_checksums[f"fold_{fold}_scores"] = sha256(predictions_path)
            input_checksums[f"fold_{fold}_report"] = sha256(report_path)

    candidate_score = route_score + np.float32(RANK_REPLACEMENT_WEIGHT) * (
        candidate_rank - parent_qwen3vl_rank
    )
    for fold in SCREEN_FOLDS:
        mask = (folds == fold) & (categories == base.FLAMMABLE)
        candidate[mask] = (candidate_score[mask] >= FROZEN_THRESHOLDS[fold]).astype(np.int8)
    if not np.array_equal(candidate[categories == base.BAD], baseline[categories == base.BAD]):
        raise RuntimeError("BAD predictions changed")

    guard = pd.read_csv(base.GUARD, dtype={"id": str, "connected_component": str})
    if not np.array_equal(guard.id.astype(str).to_numpy(), ids):
        raise ValueError("connected guard id mismatch")
    safe = guard.safe_for_selection.astype(bool).to_numpy()
    components = guard.connected_component.astype(str).to_numpy()
    data = pd.read_csv(base.DATA, dtype={"id": str})
    if not np.array_equal(data.id.astype(str).to_numpy(), ids):
        raise ValueError("data id mismatch")
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
        corrected = int(report["corrected"])
        regressed = int(report["regressed"])
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
            "corrected_regressed_ratio_at_least_1_5": (
                regressed == 0 or corrected / regressed >= 1.5
            ),
        }
        report["passed"] = all(report["gates"].values())
        if not np.isclose(report["baseline_macro_f1"], EXPECTED_BASELINE_MACRO[fold], atol=1e-12):
            raise ValueError(f"fold {fold} baseline Macro does not match frozen contract")
        fold_reports.append(report)

    screen_mask = safe & np.isin(folds, SCREEN_FOLDS)
    bootstrap = component_bootstrap_probability(
        labels=labels,
        categories=categories,
        components=components,
        mask=screen_mask,
        baseline=baseline,
        candidate=candidate,
        iterations=args.bootstrap,
        seed=BOOTSTRAP_SEED,
    )
    mean_delta = float(np.mean([report["delta_macro_f1"] for report in fold_reports]))
    corrected = sum(int(report["corrected"]) for report in fold_reports)
    regressed = sum(int(report["regressed"]) for report in fold_reports)
    null_passed = bool(np.array_equal(candidate, baseline))
    screen_passed = (
        not args.null_control
        and all(report["passed"] for report in fold_reports)
        and mean_delta >= 0.003
        and (regressed == 0 or corrected / regressed >= 1.5)
        and bootstrap["probability_delta_positive"] >= 0.80
    )
    result = {
        "experiment_id": "530",
        "evaluation_version": "internvl_pairwise_rank_replacement_screen_v1",
        "mode": "null_control" if args.null_control else "candidate",
        "purpose": "Reject-only two-fold screen; passing does not authorize submission",
        "required_screen_folds": list(SCREEN_FOLDS),
        "changed_factor": (
            "On the frozen flammable selector only, replace the Qwen3-VL rank with "
            "the donor-CDF rank from pairwise-trained InternVL at fixed weight 0.10"
        ),
        "selector_uses_labels": False,
        "pair_labels_source": "outer-donor train rows only",
        "fixed_thresholds": FROZEN_THRESHOLDS,
        "candidate_report_contracts": report_contracts,
        "folds": fold_reports,
        "mean_screen_delta_macro_f1": mean_delta,
        "required_mean_delta_macro_f1": 0.003,
        "corrected": corrected,
        "regressed": regressed,
        "component_bootstrap": bootstrap,
        "required_bootstrap_probability": 0.80,
        "bad_changed_predictions": int(((candidate != baseline) & (categories == base.BAD)).sum()),
        "screen_passed": screen_passed,
        "full_five_fold_cycle_allowed": screen_passed,
        "null_control_passed": null_passed if args.null_control else None,
        "input_sha256": {
            **input_checksums,
            "selector_membership": sha256(MEMBERSHIP),
            "pair_manifest_audit": sha256(PAIR_AUDIT),
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
        route400_score=route_score,
        parent_qwen3vl_rank=parent_qwen3vl_rank,
        candidate_internvl_rank=candidate_rank,
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

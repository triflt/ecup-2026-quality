from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATES = ("hardneg_candidate", "rank_candidate")
CONTROL = "gold_control"
EXPECTED_GUARD_SHA256 = "b2cd736226c05122f33aa0c898b31408ef469df0cfe7fe739a95c5b5b94a526a"
EXPECTED_SCORE_CALIBRATION = "category_and_outer_fold_percentile_average_ties"
EXPECTED_IMAGE_PREPROCESSING = "solution140_first_image_thumbnail_448_lanczos_v1"
MATCHED_CONTROL_VERIFICATION = {
    "schema_version": "exp698_matched_control_verification_v1",
    "fold_output_schema": "exp698_fold_output_v2",
    "common_training_factors_equal_per_fold": True,
    "candidate_teacher_bindings_equal_per_fold": True,
    "gold_teacher_binding_absent": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def category_scores(
    labels: np.ndarray,
    predictions: np.ndarray,
    categories: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    result = {
        category: f1(
            labels[mask & (categories == category)],
            predictions[mask & (categories == category)],
        )
        for category in sorted(np.unique(categories))
    }
    result["Macro"] = float(np.mean(list(result.values())))
    return result


def audit_candidate(
    *,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    components: np.ndarray,
    safe: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict:
    control_safe = category_scores(labels, control, categories, safe)
    candidate_safe = category_scores(labels, candidate, categories, safe)
    category_deltas = {
        category: candidate_safe[category] - control_safe[category]
        for category in sorted(np.unique(categories))
    }
    fold_rows = []
    for fold in range(5):
        mask = safe & (folds == fold)
        old = category_scores(labels, control, categories, mask)
        new = category_scores(labels, candidate, categories, mask)
        fold_rows.append(
            {
                "fold": fold,
                "safe_rows": int(mask.sum()),
                "control_macro_f1": old["Macro"],
                "candidate_macro_f1": new["Macro"],
                "delta": new["Macro"] - old["Macro"],
            }
        )

    grouped: dict[str, list[np.ndarray]] = {}
    for category in sorted(np.unique(categories)):
        by_component: dict[str, list[int]] = {}
        for position in np.flatnonzero(safe & (categories == category)):
            by_component.setdefault(str(components[position]), []).append(int(position))
        grouped[category] = [np.asarray(rows, dtype=np.int64) for rows in by_component.values()]
    rng = np.random.default_rng(seed)
    deltas = np.empty(bootstrap, dtype=np.float64)
    for iteration in range(bootstrap):
        control_values = []
        candidate_values = []
        for category in sorted(grouped):
            groups = grouped[category]
            selected = rng.integers(0, len(groups), size=len(groups))
            sample = np.concatenate([groups[index] for index in selected])
            control_values.append(f1(labels[sample], control[sample]))
            candidate_values.append(f1(labels[sample], candidate[sample]))
        deltas[iteration] = np.mean(candidate_values) - np.mean(control_values)

    macro_delta = candidate_safe["Macro"] - control_safe["Macro"]
    fold_wins = int(sum(row["delta"] > 0 for row in fold_rows))
    probability = float((deltas > 0).mean())
    changed = safe & (control != candidate)
    gate = (
        macro_delta >= 0.001
        and fold_wins >= 4
        and min(category_deltas.values()) >= -0.005
        and probability >= 0.90
    )
    return {
        "safe_rows": int(safe.sum()),
        "control": control_safe,
        "candidate": candidate_safe,
        "macro_delta": macro_delta,
        "category_deltas": category_deltas,
        "fold_wins": fold_wins,
        "folds": fold_rows,
        "changed_predictions": int(changed.sum()),
        "corrected": int((changed & (control != labels) & (candidate == labels)).sum()),
        "regressed": int((changed & (control == labels) & (candidate != labels)).sum()),
        "component_bootstrap": {
            "iterations": bootstrap,
            "seed": seed,
            "probability_delta_positive": probability,
            "delta_ci95": [
                float(np.quantile(deltas, 0.025)),
                float(np.quantile(deltas, 0.975)),
            ],
        },
        "objective_semantic_gate": gate,
        "strict_component_transfer_v2_delta_gate": macro_delta >= 0.003,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite connected guard")
    if sha256(args.guard) != EXPECTED_GUARD_SHA256:
        raise ValueError("connected guard checksum mismatch")
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if evaluation.get("schema_version") != "exp698_evaluation_v2":
        raise ValueError("evaluation schema mismatch")
    if evaluation.get("matched_control_verification") != MATCHED_CONTROL_VERIFICATION:
        raise ValueError("matched-control verification mismatch")
    if evaluation.get("image_preprocessing") != EXPECTED_IMAGE_PREPROCESSING:
        raise ValueError("evaluation image preprocessing mismatch")
    for mode in (CONTROL, *CANDIDATES):
        if (
            evaluation.get("modes", {}).get(mode, {}).get("score_calibration")
            != EXPECTED_SCORE_CALIBRATION
        ):
            raise ValueError(f"evaluation score calibration mismatch for {mode}")
    if evaluation.get("oof_artifact_sha256") != sha256(args.oof):
        raise ValueError("evaluation/OOF binding mismatch")
    oof = np.load(args.oof, allow_pickle=False)
    guard = pd.read_csv(
        args.guard,
        dtype={"id": str, "category": str, "connected_component": str},
    )
    ids = oof["ids"].astype(str)
    labels = oof["labels"].astype(np.int8)
    categories = oof["categories"].astype(str)
    folds = oof["folds"].astype(np.int8)
    if not np.array_equal(ids, guard["id"].astype(str).to_numpy()):
        raise ValueError("OOF/guard ID alignment mismatch")
    if not np.array_equal(labels, guard["label"].to_numpy(np.int8)):
        raise ValueError("OOF/guard label alignment mismatch")
    if not np.array_equal(categories, guard["category"].astype(str).to_numpy()):
        raise ValueError("OOF/guard category alignment mismatch")
    if not np.array_equal(folds, guard["fold"].to_numpy(np.int8)):
        raise ValueError("OOF/guard fold alignment mismatch")
    safe_column = guard["safe_for_selection"]
    if pd.api.types.is_bool_dtype(safe_column):
        safe = safe_column.to_numpy(bool)
    else:
        mapped = safe_column.astype(str).str.lower().map({"true": True, "false": False})
        if mapped.isna().any():
            raise ValueError("connected-safe mask contains non-boolean values")
        safe = mapped.to_numpy(bool)
    if int(safe.sum()) != 11207:
        raise ValueError("connected-safe row count mismatch")
    components = guard["connected_component"].astype(str).to_numpy()
    control = oof[f"{CONTROL}_predictions"].astype(np.int8)
    audits = {}
    for offset, candidate_name in enumerate(CANDIDATES):
        audits[candidate_name] = audit_candidate(
            labels=labels,
            categories=categories,
            folds=folds,
            components=components,
            safe=safe,
            control=control,
            candidate=oof[f"{candidate_name}_predictions"].astype(np.int8),
            bootstrap=args.bootstrap,
            seed=args.seed + offset,
        )
        audits[candidate_name]["primary_oof_gate"] = bool(
            evaluation.get("comparisons", {})
            .get(candidate_name, {})
            .get("science_gate", False)
        )
        audits[candidate_name]["promotion_gate"] = (
            audits[candidate_name]["primary_oof_gate"]
            and audits[candidate_name]["objective_semantic_gate"]
        )
    promoted = [name for name, value in audits.items() if value["promotion_gate"]]
    result = {
        "schema_version": "exp698_connected_family_guard_v2",
        "experiment_id": "698",
        "evaluation_version": "connected_family_guard_v2",
        "rows": len(guard),
        "safe_rows": int(safe.sum()),
        "unsafe_rows": int((~safe).sum()),
        "audits": audits,
        "promoted_candidates": promoted,
        "decision": (
            "DISTILLATION_PROMOTED" if promoted else "NO_DISTILLATION_PROMOTION"
        ),
        "input_sha256": {
            "evaluation": sha256(args.evaluation),
            "oof": sha256(args.oof),
            "guard": sha256(args.guard),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

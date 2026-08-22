from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
LOCKED = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
EXP230 = ROOT / "experiments/230_qwen35_second_seed/results/seed_ensemble_report.npz"
BASELINE230 = ROOT / "research/qwen3vl-qwen35-5fold-nested-fusion.npz"
EXP300 = ROOT / "experiments/300_internvl35_attribute_screen/results/attribute_gate_report.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def scores(
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


def audit(
    *,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    components: np.ndarray,
    safe: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    all_rows = np.ones(len(labels), dtype=bool)
    full_old = scores(labels, baseline, categories, all_rows)
    full_new = scores(labels, candidate, categories, all_rows)
    safe_old = scores(labels, baseline, categories, safe)
    safe_new = scores(labels, candidate, categories, safe)
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        mask = safe & (folds == fold)
        old = scores(labels, baseline, categories, mask)
        new = scores(labels, candidate, categories, mask)
        fold_rows.append(
            {
                "fold": int(fold),
                "safe_rows": int(mask.sum()),
                "baseline_macro_f1": old["Macro"],
                "candidate_macro_f1": new["Macro"],
                "delta": new["Macro"] - old["Macro"],
                "baseline_category_f1": {k: v for k, v in old.items() if k != "Macro"},
                "candidate_category_f1": {k: v for k, v in new.items() if k != "Macro"},
            }
        )
    grouped: dict[str, list[np.ndarray]] = {}
    for category in sorted(np.unique(categories)):
        by_component: dict[str, list[int]] = {}
        for position in np.flatnonzero(safe & (categories == category)):
            by_component.setdefault(components[position], []).append(position)
        grouped[category] = [np.asarray(rows) for rows in by_component.values()]
    rng = np.random.default_rng(seed)
    deltas = np.empty(bootstrap, dtype=np.float64)
    for iteration in range(bootstrap):
        old_values, new_values = [], []
        for category in sorted(grouped):
            groups = grouped[category]
            selected = rng.integers(0, len(groups), size=len(groups))
            sample = np.concatenate([groups[index] for index in selected])
            old_values.append(f1(labels[sample], baseline[sample]))
            new_values.append(f1(labels[sample], candidate[sample]))
        deltas[iteration] = np.mean(new_values) - np.mean(old_values)
    return {
        "full_rows": {
            "baseline": full_old,
            "candidate": full_new,
            "delta_macro_f1": full_new["Macro"] - full_old["Macro"],
        },
        "connected_safe_rows": {
            "rows": int(safe.sum()),
            "baseline": safe_old,
            "candidate": safe_new,
            "delta_macro_f1": safe_new["Macro"] - safe_old["Macro"],
            "folds_won": int(sum(row["delta"] > 0 for row in fold_rows)),
            "folds": fold_rows,
            "changed_predictions": int((safe & (baseline != candidate)).sum()),
            "corrected": int((safe & (baseline != labels) & (candidate == labels)).sum()),
            "regressed": int((safe & (baseline == labels) & (candidate != labels)).sum()),
            "component_bootstrap": {
                "iterations": bootstrap,
                "seed": seed,
                "probability_delta_positive": float((deltas > 0).mean()),
                "delta_ci95": [
                    float(np.quantile(deltas, 0.025)),
                    float(np.quantile(deltas, 0.975)),
                ],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    guard = pd.read_csv(
        GUARD, dtype={"id": str, "connected_component": str}
    )
    safe = guard.safe_for_selection.astype(bool).to_numpy()
    components = guard.connected_component.astype(str).to_numpy()
    sources: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    locked = np.load(LOCKED, allow_pickle=False)
    locked_ids = locked["ids"].astype(str)
    sources["exp241_locked_190"] = (
        locked_ids,
        locked["labels"].astype(np.int8),
        locked["categories"].astype(str),
        locked["folds"].astype(np.int8),
        np.column_stack(
            [locked["baseline_nested_predictions"], locked["exp241_nested_predictions"]]
        ),
    )
    sources["exp260_locked_190"] = (
        locked_ids,
        locked["labels"].astype(np.int8),
        locked["categories"].astype(str),
        locked["folds"].astype(np.int8),
        np.column_stack(
            [locked["baseline_nested_predictions"], locked["exp260_nested_predictions"]]
        ),
    )
    sources["exp270_locked_190"] = (
        locked_ids,
        locked["labels"].astype(np.int8),
        locked["categories"].astype(str),
        locked["folds"].astype(np.int8),
        np.column_stack(
            [locked["baseline_nested_predictions"], locked["exp270_nested_predictions"]]
        ),
    )
    exp230 = np.load(EXP230, allow_pickle=True)
    baseline230 = np.load(BASELINE230, allow_pickle=True)
    for field in ("ids", "labels", "categories", "folds"):
        if not np.array_equal(
            baseline230[field].astype(exp230[field].dtype), exp230[field]
        ):
            raise ValueError(f"experiment 230 baseline {field} mismatch")
    sources["exp230_two_seed_probability"] = (
        exp230["ids"].astype(str),
        exp230["labels"].astype(np.int8),
        exp230["categories"].astype(str),
        exp230["folds"].astype(np.int8),
        np.column_stack(
            [baseline230["nested_predictions"], exp230["probability_nested_predictions"]]
        ),
    )
    exp300 = np.load(EXP300, allow_pickle=False)
    sources["exp300_internvl_attribute_gate"] = (
        exp300["ids"].astype(str),
        exp300["labels"].astype(np.int8),
        exp300["categories"].astype(str),
        exp300["folds"].astype(np.int8),
        np.column_stack(
            [exp300["baseline_predictions"], exp300["candidate_predictions"]]
        ),
    )

    audits = {}
    for offset, (name, (ids, labels, categories, folds, predictions)) in enumerate(
        sources.items()
    ):
        if not np.array_equal(ids, guard.id.to_numpy()):
            raise ValueError(f"{name} ids do not match connected guard")
        if not np.array_equal(folds, guard.fold.to_numpy(np.int8)):
            raise ValueError(f"{name} folds do not match connected guard")
        audits[name] = audit(
            labels=labels,
            categories=categories,
            folds=folds,
            components=components,
            safe=safe,
            baseline=predictions[:, 0].astype(np.int8),
            candidate=predictions[:, 1].astype(np.int8),
            bootstrap=args.bootstrap,
            seed=args.seed + offset,
        )
    result = {
        "evaluation_version": "connected_family_guard_v2",
        "purpose": "Re-audit already selected candidates on historical OOF rows without exact-text or exact-first-image components crossing folds",
        "selection_warning": "This audit is retrospective and cannot by itself promote a candidate; it calibrates validation reliability and becomes a mandatory guard for future local methods.",
        "rows_total": int(len(guard)),
        "safe_rows": int(safe.sum()),
        "unsafe_rows": int((~safe).sum()),
        "audits": audits,
        "input_sha256": {
            "guard": sha256(GUARD),
            "locked": sha256(LOCKED),
            "exp230": sha256(EXP230),
            "exp230_baseline": sha256(BASELINE230),
            "exp300": sha256(EXP300),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
DEFAULT_LOCKED = Path(__file__).resolve().parent / "results/locked_replacement_report.npz"
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
DATA = ROOT / "research/data.csv"
OUT = Path(__file__).resolve().parent / "results"
FLAMMABLE = "Легковоспламеняющиеся"

# Frozen before reading experiment 410 predictions. The union is the safety
# endpoint; individual cohorts remain diagnostic because they overlap.
SAFETY_PATTERNS = {
    "ignition_source": r"(?iu)\b(?:зажигалк|спич|огнив|факел|свеч|горелк)\w*\b",
    "pyrotechnics": r"(?iu)\b(?:фейерверк|салют|петард|бенгал|пиротех|дымогенератор|свеч\w*\s*фонтан|фонтан\w*\s+для\s+торт)\w*\b",
    "charcoal_wood": r"(?iu)\b(?:угол|уголь|дров|брик|розжиг)\w*\b",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


connected = load_module(
    "connected_guard_audit_410",
    ROOT / "research/audit_connected_guard_existing.py",
)
priors = load_module(
    "decision_survival_audit_410",
    ROOT / "research/audit_component_decision_survival.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(labels: np.ndarray, predictions: np.ndarray, categories: np.ndarray) -> dict:
    category = {
        value: connected.f1(labels[categories == value], predictions[categories == value])
        for value in sorted(np.unique(categories))
    }
    return {"category_f1": category, "macro_f1": float(np.mean(list(category.values())))}


def cohort_audit(
    frame: pd.DataFrame,
    labels: np.ndarray,
    categories: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict:
    text = (
        frame["name"].fillna("").astype(str)
        + "\n"
        + frame["description"].fillna("").astype(str)
    )
    positive_flammable = (categories == FLAMMABLE) & (labels == 1)
    union = np.zeros(len(frame), dtype=bool)
    cohorts = {}
    for name, pattern in SAFETY_PATTERNS.items():
        mask = positive_flammable & text.str.contains(pattern, regex=True, na=False).to_numpy()
        union |= mask
        old_fn = int((baseline[mask] == 0).sum())
        new_fn = int((candidate[mask] == 0).sum())
        cohorts[name] = {
            "positive_rows": int(mask.sum()),
            "baseline_false_negatives": old_fn,
            "candidate_false_negatives": new_fn,
            "false_negative_delta": new_fn - old_fn,
            "baseline_recall": float(1 - old_fn / max(1, int(mask.sum()))),
            "candidate_recall": float(1 - new_fn / max(1, int(mask.sum()))),
        }
    old_union_fn = int((baseline[union] == 0).sum())
    new_union_fn = int((candidate[union] == 0).sum())
    return {
        "definitions": SAFETY_PATTERNS,
        "cohorts": cohorts,
        "union": {
            "positive_rows": int(union.sum()),
            "baseline_false_negatives": old_union_fn,
            "candidate_false_negatives": new_union_fn,
            "false_negative_delta": new_union_fn - old_union_fn,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked", type=Path, default=DEFAULT_LOCKED)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    archive = np.load(args.locked, allow_pickle=False)
    ids = archive["ids"].astype(str)
    labels = archive["labels"].astype(np.int8)
    categories = archive["categories"].astype(str)
    folds = archive["folds"].astype(np.int8)
    original_nested = archive["baseline_nested_predictions"].astype(np.int8)
    exp260_nested = archive["exp260_nested_predictions"].astype(np.int8)
    exp410_nested = archive["exp410_nested_predictions"].astype(np.int8)
    original_production = archive["baseline_production_predictions"].astype(np.int8)
    exp260_production = archive["exp260_production_predictions"].astype(np.int8)
    exp410_production = archive["exp410_production_predictions"].astype(np.int8)

    route400_nested = np.where(
        categories == FLAMMABLE, exp260_nested, original_nested
    ).astype(np.int8)
    route410_nested = np.where(
        categories == FLAMMABLE, exp410_nested, original_nested
    ).astype(np.int8)
    route400_production = np.where(
        categories == FLAMMABLE, exp260_production, original_production
    ).astype(np.int8)
    route410_production = np.where(
        categories == FLAMMABLE, exp410_production, original_production
    ).astype(np.int8)

    baseline_score = score(labels, route400_nested, categories)
    candidate_score = score(labels, route410_nested, categories)
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        old = score(labels[mask], route400_nested[mask], categories[mask])
        new = score(labels[mask], route410_nested[mask], categories[mask])
        fold_rows.append({
            "fold": int(fold),
            "baseline_macro_f1": old["macro_f1"],
            "candidate_macro_f1": new["macro_f1"],
            "delta": new["macro_f1"] - old["macro_f1"],
            "baseline_category_f1": old["category_f1"],
            "candidate_category_f1": new["category_f1"],
        })

    changed = route400_nested != route410_nested
    corrected = changed & (route410_nested == labels) & (route400_nested != labels)
    regressed = changed & (route410_nested != labels) & (route400_nested == labels)
    flammable_positive = (categories == FLAMMABLE) & (labels == 1)
    overall_fn = {
        "baseline": int((route400_nested[flammable_positive] == 0).sum()),
        "candidate": int((route410_nested[flammable_positive] == 0).sum()),
    }
    overall_fn["delta"] = overall_fn["candidate"] - overall_fn["baseline"]

    guard = pd.read_csv(GUARD, dtype={"id": str, "connected_component": str})
    if not np.array_equal(guard.id.to_numpy(), ids):
        raise ValueError("guard id mismatch")
    connected_audit = connected.audit(
        labels=labels,
        categories=categories,
        folds=folds,
        components=guard.connected_component.astype(str).to_numpy(),
        safe=guard.safe_for_selection.astype(bool).to_numpy(),
        baseline=route400_nested,
        candidate=route410_nested,
        bootstrap=args.bootstrap,
        seed=41042,
    )

    routed_archive = OUT / "routed_predictions.npz"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        routed_archive,
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        baseline_nested_predictions=route400_nested,
        category_routed_nested_predictions=route410_nested,
        baseline_production_predictions=route400_production,
        category_routed_production_predictions=route410_production,
    )

    frame = priors.prepare_frame(DATA, np.load(routed_archive, allow_pickle=False))
    neighbor_indices, neighbor_scores = priors.build_neighbor_graph(frame)
    baseline_after, baseline_prior_audit = priors.apply_downstream_priors(
        frame, route400_nested, neighbor_indices, neighbor_scores
    )
    candidate_after, candidate_prior_audit = priors.apply_downstream_priors(
        frame, route410_nested, neighbor_indices, neighbor_scores
    )
    change_survival = priors.changed_audit(
        labels,
        categories,
        route400_nested,
        route410_nested,
        baseline_after,
        candidate_after,
    )
    prior_report = {
        "baseline_before_priors": priors.scores(labels, route400_nested, categories),
        "candidate_before_priors": priors.scores(labels, route410_nested, categories),
        "baseline_after_priors": priors.scores(labels, baseline_after, categories),
        "candidate_after_priors": priors.scores(labels, candidate_after, categories),
        "change_survival": change_survival,
        "baseline_prior_hits": baseline_prior_audit,
        "candidate_prior_hits": candidate_prior_audit,
    }
    prior_report["delta_after_priors"] = {
        key: prior_report["candidate_after_priors"][key]
        - prior_report["baseline_after_priors"][key]
        for key in ("bad_f1", "flammable_f1", "macro_f1")
    }
    (OUT / "prior_replay.json").write_text(
        json.dumps(prior_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    safety = cohort_audit(
        frame, labels, categories, route400_nested, route410_nested
    )
    (OUT / "safety_cohort_audit.json").write_text(
        json.dumps(safety, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    macro_delta = candidate_score["macro_f1"] - baseline_score["macro_f1"]
    safe_rows = connected_audit["connected_safe_rows"]
    bootstrap_probability = safe_rows["component_bootstrap"][
        "probability_delta_positive"
    ]
    survival_rate = change_survival["survival_rate"]
    acceptance = {
        "macro_delta_at_least_0_003": macro_delta >= 0.003,
        "folds_won_at_least_4": sum(row["delta"] > 0 for row in fold_rows) >= 4,
        "bad_unchanged": candidate_score["category_f1"]["БАД"]
        == baseline_score["category_f1"]["БАД"],
        "flammable_f1_positive": candidate_score["category_f1"][FLAMMABLE]
        > baseline_score["category_f1"][FLAMMABLE],
        "flammable_false_negatives_not_increased": overall_fn["delta"] <= 0,
        "safety_union_false_negatives_not_increased": safety["union"][
            "false_negative_delta"
        ]
        <= 0,
        "connected_bootstrap_probability_at_least_0_90": bootstrap_probability
        >= 0.90,
        "prior_survival_at_least_0_80": survival_rate >= 0.80,
        "post_prior_delta_positive": prior_report["delta_after_priors"][
            "macro_f1"
        ]
        > 0,
    }
    result = {
        "experiment_id": "410",
        "evaluation_version": "locked_190_category_routing_v2",
        "single_changed_factor": "family-diverse flammable negatives in the exact experiment 260 training recipe",
        "baseline": baseline_score,
        "candidate": candidate_score,
        "delta_macro_f1": macro_delta,
        "folds": fold_rows,
        "folds_won": int(sum(row["delta"] > 0 for row in fold_rows)),
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "regressed": int(regressed.sum()),
        "flammable_false_negatives": overall_fn,
        "safety_cohorts": safety,
        "connected_safe": connected_audit,
        "prior_replay": prior_report,
        "acceptance": acceptance,
        "stage1_passed": all(acceptance.values()),
        "repeat_seed_required": True,
        "input_sha256": {
            str(args.locked.relative_to(ROOT)): sha256(args.locked),
            str(GUARD.relative_to(ROOT)): sha256(GUARD),
            str(DATA.relative_to(ROOT)): sha256(DATA),
        },
    }
    (OUT / "acceptance_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "stage1_passed": result["stage1_passed"],
        "delta_macro_f1": macro_delta,
        "folds_won": result["folds_won"],
        "corrected": result["corrected"],
        "regressed": result["regressed"],
        "flammable_false_negative_delta": overall_fn["delta"],
        "safety_false_negative_delta": safety["union"]["false_negative_delta"],
        "connected_bootstrap_probability": bootstrap_probability,
        "prior_survival_rate": survival_rate,
        "post_prior_delta": prior_report["delta_after_priors"]["macro_f1"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

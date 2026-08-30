from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
PREDICTIONS = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
DATA = ROOT / "research/data.csv"
OUT = Path(__file__).resolve().parent / "results"
FLAMMABLE = "Легковоспламеняющиеся"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


connected = load_module(
    "connected_guard_audit",
    ROOT / "research/audit_connected_guard_existing.py",
)
priors = load_module(
    "decision_survival_audit",
    ROOT / "research/audit_component_decision_survival.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(labels: np.ndarray, predictions: np.ndarray, categories: np.ndarray) -> dict[str, float]:
    values = {
        category: connected.f1(labels[categories == category], predictions[categories == category])
        for category in sorted(np.unique(categories))
    }
    return {"category_f1": values, "macro_f1": float(np.mean(list(values.values())))}


def main() -> None:
    archive = np.load(PREDICTIONS, allow_pickle=False)
    ids = archive["ids"].astype(str)
    labels = archive["labels"].astype(np.int8)
    categories = archive["categories"].astype(str)
    folds = archive["folds"].astype(np.int8)
    baseline_nested = archive["baseline_nested_predictions"].astype(np.int8)
    exp260_nested = archive["exp260_nested_predictions"].astype(np.int8)
    baseline_production = archive["baseline_production_predictions"].astype(np.int8)
    exp260_production = archive["exp260_production_predictions"].astype(np.int8)
    routed_nested = np.where(categories == FLAMMABLE, exp260_nested, baseline_nested).astype(np.int8)
    routed_production = np.where(
        categories == FLAMMABLE, exp260_production, baseline_production
    ).astype(np.int8)

    baseline_score = score(labels, baseline_nested, categories)
    routed_score = score(labels, routed_nested, categories)
    fold_rows: list[dict[str, float | int]] = []
    for fold in sorted(np.unique(folds)):
        old, new = [], []
        for category in sorted(np.unique(categories)):
            mask = (folds == fold) & (categories == category)
            old.append(connected.f1(labels[mask], baseline_nested[mask]))
            new.append(connected.f1(labels[mask], routed_nested[mask]))
        fold_rows.append({
            "fold": int(fold),
            "baseline_macro_f1": float(np.mean(old)),
            "candidate_macro_f1": float(np.mean(new)),
            "delta": float(np.mean(new) - np.mean(old)),
        })
    changed = baseline_nested != routed_nested
    corrected = changed & (routed_nested == labels) & (baseline_nested != labels)
    regressed = changed & (routed_nested != labels) & (baseline_nested == labels)

    guard = pd.read_csv(GUARD, dtype={"id": str, "connected_component": str})
    if not np.array_equal(guard.id.to_numpy(), ids):
        raise ValueError("guard id mismatch")
    connected_audit = connected.audit(
        labels=labels,
        categories=categories,
        folds=folds,
        components=guard.connected_component.astype(str).to_numpy(),
        safe=guard.safe_for_selection.astype(bool).to_numpy(),
        baseline=baseline_nested,
        candidate=routed_nested,
        bootstrap=10000,
        seed=40042,
    )

    routed_archive = OUT / "routed_predictions.npz"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        routed_archive,
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        baseline_nested_predictions=baseline_nested,
        category_routed_nested_predictions=routed_nested,
        baseline_production_predictions=baseline_production,
        category_routed_production_predictions=routed_production,
    )

    frame = priors.prepare_frame(DATA, np.load(routed_archive, allow_pickle=False))
    neighbor_indices, neighbor_scores = priors.build_neighbor_graph(frame)
    baseline_after, baseline_prior_audit = priors.apply_downstream_priors(
        frame, baseline_nested, neighbor_indices, neighbor_scores
    )
    routed_after, routed_prior_audit = priors.apply_downstream_priors(
        frame, routed_nested, neighbor_indices, neighbor_scores
    )
    prior_report = {
        "baseline_before_priors": priors.scores(labels, baseline_nested, categories),
        "candidate_before_priors": priors.scores(labels, routed_nested, categories),
        "baseline_after_priors": priors.scores(labels, baseline_after, categories),
        "candidate_after_priors": priors.scores(labels, routed_after, categories),
        "change_survival": priors.changed_audit(
            labels, categories, baseline_nested, routed_nested, baseline_after, routed_after
        ),
        "baseline_prior_hits": baseline_prior_audit,
        "candidate_prior_hits": routed_prior_audit,
    }
    prior_report["delta_after_priors"] = {
        key: prior_report["candidate_after_priors"][key] - prior_report["baseline_after_priors"][key]
        for key in ("bad_f1", "flammable_f1", "macro_f1")
    }
    (OUT / "prior_replay.json").write_text(
        json.dumps(prior_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    macro_delta = routed_score["macro_f1"] - baseline_score["macro_f1"]
    result = {
        "experiment_id": "400",
        "evaluation_version": "locked_190_category_routing_v1",
        "single_changed_factor": "original adapter for BAD; experiment 260 adapter for flammable",
        "baseline": baseline_score,
        "candidate": routed_score,
        "category_delta": {
            category: routed_score["category_f1"][category] - baseline_score["category_f1"][category]
            for category in baseline_score["category_f1"]
        },
        "delta_macro_f1": macro_delta,
        "folds": fold_rows,
        "folds_won": int(sum(row["delta"] > 0 for row in fold_rows)),
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "regressed": int(regressed.sum()),
        "connected_safe": connected_audit,
        "prior_replay": prior_report,
        "acceptance": {
            "macro_delta_at_least_0_005": macro_delta >= 0.005,
            "folds_won_at_least_4": sum(row["delta"] > 0 for row in fold_rows) >= 4,
            "bad_unchanged": routed_score["category_f1"]["БАД"] == baseline_score["category_f1"]["БАД"],
            "flammable_delta_positive": routed_score["category_f1"][FLAMMABLE] > baseline_score["category_f1"][FLAMMABLE],
            "corrected_to_regressed_at_least_1_5": int(corrected.sum()) >= 1.5 * max(1, int(regressed.sum())),
            "connected_safe_delta_positive": connected_audit["connected_safe_rows"]["delta_macro_f1"] > 0,
            "post_prior_delta_positive": prior_report["delta_after_priors"]["macro_f1"] > 0,
        },
        "input_sha256": {
            str(PREDICTIONS.relative_to(ROOT)): sha256(PREDICTIONS),
            str(GUARD.relative_to(ROOT)): sha256(GUARD),
            str(DATA.relative_to(ROOT)): sha256(DATA),
        },
    }
    result["accepted_for_submission_integration"] = all(result["acceptance"].values())
    (OUT / "acceptance_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "accepted_for_submission_integration": result["accepted_for_submission_integration"],
        "delta_macro_f1": macro_delta,
        "folds_won": result["folds_won"],
        "corrected": result["corrected"],
        "regressed": result["regressed"],
        "connected_safe_delta": connected_audit["connected_safe_rows"]["delta_macro_f1"],
        "post_prior_delta": prior_report["delta_after_priors"]["macro_f1"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

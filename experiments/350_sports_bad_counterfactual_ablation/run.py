from __future__ import annotations

import gc
import hashlib
import html
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
sys.path.insert(0, str(RESEARCH))

from audit_component_decision_survival import (  # noqa: E402
    apply_downstream_priors,
    build_neighbor_graph,
    prepare_frame,
)
from qwen35_locked_190_audit import LOCKED_CONFIG, best_threshold, f1  # noqa: E402


DATA = RESEARCH / "data.csv"
QWEN35 = RESEARCH / "qwen35-hard-5fold-robust-fusion-report.npz"
QWEN3VL = RESEARCH / "lora-hard-5fold-robust-fusion-report.npz"
LOCKED = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
REPEATS = ROOT / "validation/connected_family_repeated_v1/rows.csv"
OUT = Path(__file__).resolve().parent / "results"

BAD = "БАД"
REGULATORY = re.compile(
    r"(?iu)(?:\bбад\b|биологически\s+активн\w*\s+добавк|dietary\s+supplement|food\s+supplement|"
    r"не\s+является\s+лекарств\w*|не\s+лекарственн\w*|таблетк\w*|капсул\w*|саше|порошок|сироп|капл[ияи]\w*|спрей\w*)"
)
SPORTS = re.compile(
    r"(?iu)(?:спортпит\w*|спортивн\w*\s+питан\w*|\bbcaa\b|протеин\w*|гейнер\w*|креатин\w*|"
    r"карнитин\w*|аминокислот\w*|предтрен\w*)"
)
EXPLICIT_BAD = re.compile(
    r"(?iu)(?:\bбад\b|биологически\s+активн\w*\s+добавк|dietary\s+supplement|food\s+supplement)"
)
PLACEBO = re.compile(r"(?iu)(?:продукт\w*|комплекс\w*|средств\w*|вкус\w*|упаковк\w*)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: object) -> str:
    value = html.unescape(str(value or "")).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", value).strip()


def compose(name: object, description: object) -> str:
    name = normalize(name)
    description = normalize(description)
    return f"{name}\n{name}\n{description}"


def ablate(text: str, pattern: re.Pattern[str]) -> str:
    return re.sub(r"\s+", " ", pattern.sub(" ", text)).strip()


def make_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.997, sublinear_tf=True,
            max_features=180_000, dtype=np.float32,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
            max_features=220_000, dtype=np.float32,
        )),
    ])


def score_topology(
    *,
    name: str,
    frame: pd.DataFrame,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    safe: np.ndarray,
    baseline: np.ndarray,
    qwen3vl_rank: np.ndarray,
    qwen35_rank: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray | str]]:
    candidate = baseline.copy()
    score_original = np.full(len(frame), np.nan, dtype=np.float32)
    score_regulatory = np.full(len(frame), np.nan, dtype=np.float32)
    score_sports = np.full(len(frame), np.nan, dtype=np.float32)
    score_placebo = np.full(len(frame), np.nan, dtype=np.float32)
    direction = np.full(len(frame), "", dtype="<U2")
    sports_selector = safe & (categories == BAD) & frame.sports_nutrition.to_numpy(bool)
    fold_rows = []

    for outer_fold in sorted(np.unique(folds[safe])):
        donor_all = safe & (folds != outer_fold)
        donor_bad = donor_all & (categories == BAD)
        outer = sports_selector & (folds == outer_fold)
        if not outer.any():
            raise ValueError(f"{name} fold {outer_fold} has no sports rows")
        vectorizer = make_vectorizer()
        donor_matrix = vectorizer.fit_transform(frame.loc[donor_all, "text"])
        donor_categories = categories[donor_all]
        model = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=8000)
        model.fit(donor_matrix[donor_categories == BAD], labels[donor_bad])

        variants = [
            frame.loc[outer, "text"].tolist(),
            frame.loc[outer, "text_regulatory"].tolist(),
            frame.loc[outer, "text_sports"].tolist(),
            frame.loc[outer, "text_placebo"].tolist(),
        ]
        values = [model.decision_function(vectorizer.transform(texts)).astype(np.float32) for texts in variants]
        score_original[outer], score_regulatory[outer], score_sports[outer], score_placebo[outer] = values

        _, q3_threshold = best_threshold(labels[donor_bad], qwen3vl_rank[donor_bad])
        _, q35_threshold = best_threshold(labels[donor_bad], qwen35_rank[donor_bad])
        q3_positive = qwen3vl_rank[outer] >= q3_threshold
        q35_positive = qwen35_rank[outer] >= q35_threshold
        explicit = frame.loc[outer, "explicit_bad"].to_numpy(bool)
        delta_reg = values[0] - values[1]
        delta_sport = values[0] - values[2]
        local_baseline = baseline[outer]
        fn = (
            (local_baseline == 0) & q3_positive & q35_positive & explicit
            & (delta_reg > 0) & (delta_sport < 0)
        )
        fp = (
            (local_baseline == 1) & ~q3_positive & ~q35_positive & ~explicit
            & (delta_sport > 0)
        )
        outer_positions = np.flatnonzero(outer)
        candidate[outer_positions[fn]] = 1
        candidate[outer_positions[fp]] = 0
        direction[outer_positions[fn]] = "FN"
        direction[outer_positions[fp]] = "FP"
        fold_rows.append({
            "fold": int(outer_fold),
            "selector_rows": int(outer.sum()),
            "fn_changes": int(fn.sum()),
            "fp_changes": int(fp.sum()),
        })
        del donor_matrix, vectorizer, model, values
        gc.collect()

    return candidate, {
        "score_original": score_original,
        "score_regulatory": score_regulatory,
        "score_sports": score_sports,
        "score_placebo": score_placebo,
        "direction": direction,
        "fold_rows": fold_rows,
    }


def summarize(
    *, name: str, frame: pd.DataFrame, labels: np.ndarray, categories: np.ndarray,
    folds: np.ndarray, safe: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
    components: np.ndarray, details: dict[str, np.ndarray | str], seed: int,
) -> dict[str, object]:
    category_old, category_new = {}, {}
    for category in sorted(np.unique(categories)):
        mask = safe & (categories == category)
        category_old[category] = f1(labels[mask], baseline[mask])
        category_new[category] = f1(labels[mask], candidate[mask])
    old_macro = float(np.mean(list(category_old.values())))
    new_macro = float(np.mean(list(category_new.values())))
    fold_metrics = []
    for fold in sorted(np.unique(folds[safe])):
        old_values, new_values = [], []
        for category in sorted(np.unique(categories)):
            mask = safe & (categories == category) & (folds == fold)
            old_values.append(f1(labels[mask], baseline[mask]))
            new_values.append(f1(labels[mask], candidate[mask]))
        fold_metrics.append({
            "fold": int(fold), "baseline_macro_f1": float(np.mean(old_values)),
            "candidate_macro_f1": float(np.mean(new_values)),
            "delta": float(np.mean(new_values) - np.mean(old_values)),
        })
    changed = safe & (baseline != candidate)
    corrected = changed & (baseline != labels) & (candidate == labels)
    regressed = changed & (baseline == labels) & (candidate != labels)
    sports = safe & (categories == BAD) & frame.sports_nutrition.to_numpy(bool)
    singleton = sports & (frame.component_size.to_numpy(int) == 1)
    direction_metrics = {}
    for value in ("FN", "FP"):
        mask = details["direction"] == value
        direction_metrics[value] = {
            "changed": int(mask.sum()),
            "corrected": int((mask & corrected).sum()),
            "regressed": int((mask & regressed).sum()),
        }

    rng = np.random.default_rng(seed)
    grouped: dict[str, list[np.ndarray]] = {}
    for category in sorted(np.unique(categories)):
        local: dict[str, list[int]] = {}
        for position in np.flatnonzero(safe & (categories == category)):
            local.setdefault(components[position], []).append(position)
        grouped[category] = [np.asarray(rows) for rows in local.values()]
    deltas = np.empty(5000, dtype=np.float64)
    for iteration in range(len(deltas)):
        old_values, new_values = [], []
        for category in sorted(grouped):
            groups = grouped[category]
            sampled = rng.integers(0, len(groups), size=len(groups))
            positions = np.concatenate([groups[index] for index in sampled])
            old_values.append(f1(labels[positions], baseline[positions]))
            new_values.append(f1(labels[positions], candidate[positions]))
        deltas[iteration] = np.mean(new_values) - np.mean(old_values)

    placebo_delta = details["score_original"] - details["score_placebo"]
    changed_positions = np.flatnonzero(changed)
    return {
        "topology": name,
        "rows": int(safe.sum()),
        "baseline_category_f1": category_old,
        "candidate_category_f1": category_new,
        "baseline_macro_f1": old_macro,
        "candidate_macro_f1": new_macro,
        "delta_macro_f1": new_macro - old_macro,
        "category_delta": {key: category_new[key] - category_old[key] for key in category_old},
        "folds_won": int(sum(row["delta"] > 0 for row in fold_metrics)),
        "folds": fold_metrics,
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "regressed": int(regressed.sum()),
        "direction": direction_metrics,
        "sports": {
            "rows": int(sports.sum()),
            "baseline_f1": f1(labels[sports], baseline[sports]),
            "candidate_f1": f1(labels[sports], candidate[sports]),
            "delta_f1": f1(labels[sports], candidate[sports]) - f1(labels[sports], baseline[sports]),
        },
        "singleton_sports": {
            "rows": int(singleton.sum()),
            "baseline_f1": f1(labels[singleton], baseline[singleton]),
            "candidate_f1": f1(labels[singleton], candidate[singleton]),
            "delta_f1": f1(labels[singleton], candidate[singleton]) - f1(labels[singleton], baseline[singleton]),
        },
        "placebo_diagnostic": {
            "median_absolute_delta_on_changed": float(np.nanmedian(np.abs(placebo_delta[changed_positions]))) if len(changed_positions) else 0.0,
            "median_sports_delta_on_changed": float(np.nanmedian(np.abs((details["score_original"] - details["score_sports"])[changed_positions]))) if len(changed_positions) else 0.0,
        },
        "component_bootstrap": {
            "iterations": len(deltas), "seed": seed,
            "probability_delta_positive": float((deltas > 0).mean()),
            "delta_mean": float(deltas.mean()),
            "delta_ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        },
        "structural_invariants": {
            "changed_unsafe": int(((baseline != candidate) & ~safe).sum()),
            "changed_non_sports": int(((baseline != candidate) & ~sports).sum()),
            "changed_flammable": int(((baseline != candidate) & (categories != BAD)).sum()),
        },
        "training_folds": details["fold_rows"],
    }


def production_baseline(
    matrix: np.ndarray, categories: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(categories), dtype=np.int8)
    for category in sorted(np.unique(categories)):
        mask = categories == category
        weights = np.asarray(LOCKED_CONFIG[category]["weights"], dtype=np.float32)
        score = np.sum(matrix[mask] * weights[None, :], axis=1, dtype=np.float64)
        result[mask] = score >= LOCKED_CONFIG[category]["production_threshold"]
    return result


def main() -> None:
    started = time.monotonic()
    frame = pd.read_csv(DATA, dtype={"id": str})
    qwen35 = np.load(QWEN35, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    locked = np.load(LOCKED, allow_pickle=False)
    ids = qwen35["ids"].astype(str)
    labels = qwen35["labels"].astype(np.int8)
    categories = qwen35["categories"].astype(str)
    historical_folds = qwen35["folds"].astype(np.int8)
    for source in (qwen3vl, locked):
        for key, expected in (("ids", ids), ("labels", labels), ("categories", categories)):
            if not np.array_equal(source[key].astype(expected.dtype), expected):
                raise ValueError(f"unaligned {key}")
    if not np.array_equal(frame.id.to_numpy(str), ids):
        raise ValueError("data id order mismatch")

    guard = pd.read_csv(GUARD, dtype={"id": str}).set_index("id").loc[ids]
    repeats = pd.read_csv(REPEATS, dtype={"id": str}).set_index("id").loc[ids]
    safe = guard.safe_for_selection.to_numpy(bool)
    components = guard.connected_component.astype(str).to_numpy()
    frame["component_size"] = guard.component_size.to_numpy(int)
    frame["text"] = [compose(a, b) for a, b in zip(frame.name.fillna(""), frame.description.fillna(""))]
    frame["sports_nutrition"] = frame.text.str.contains(SPORTS, regex=True, na=False)
    frame["explicit_bad"] = frame.text.str.contains(EXPLICIT_BAD, regex=True, na=False)
    frame["text_regulatory"] = frame.text.map(lambda value: ablate(value, REGULATORY))
    frame["text_sports"] = frame.text.map(lambda value: ablate(value, SPORTS))
    frame["text_placebo"] = frame.text.map(lambda value: ablate(value, PLACEBO))

    base_rank = qwen35["base_rank"].astype(np.float32)
    qwen3vl_rank = qwen3vl["lora_rank"].astype(np.float32)
    qwen35_rank = qwen35["lora_rank"].astype(np.float32)
    matrix = np.column_stack([base_rank, qwen3vl_rank, qwen35_rank])
    production = production_baseline(matrix, categories)
    historical_baseline = locked["baseline_nested_predictions"].astype(np.int8)

    topology_specs = [
        ("historical_connected_safe", historical_folds, historical_baseline),
        ("repeat_0", repeats.repeat_0_fold.to_numpy(np.int8), production),
        ("repeat_1", repeats.repeat_1_fold.to_numpy(np.int8), production),
        ("repeat_2", repeats.repeat_2_fold.to_numpy(np.int8), production),
    ]
    topologies: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}
    for index, (name, folds, baseline) in enumerate(topology_specs):
        print(f"topology={name} start", flush=True)
        candidate, details = score_topology(
            name=name, frame=frame, labels=labels, categories=categories, folds=folds,
            safe=safe, baseline=baseline, qwen3vl_rank=qwen3vl_rank, qwen35_rank=qwen35_rank,
        )
        topologies[name] = summarize(
            name=name, frame=frame, labels=labels, categories=categories, folds=folds,
            safe=safe, baseline=baseline, candidate=candidate, components=components,
            details=details, seed=35042 + index,
        )
        arrays[f"{name}_folds"] = folds
        arrays[f"{name}_baseline"] = baseline
        arrays[f"{name}_candidate"] = candidate
        for key in ("score_original", "score_regulatory", "score_sports", "score_placebo", "direction"):
            arrays[f"{name}_{key}"] = details[key]
        print(f"topology={name} delta={topologies[name]['delta_macro_f1']:.9f}", flush=True)

    prior_frame = prepare_frame(DATA, locked)
    neighbor_indices, neighbor_scores = build_neighbor_graph(prior_frame)
    historical_candidate = arrays["historical_connected_safe_candidate"]
    baseline_after, baseline_prior = apply_downstream_priors(prior_frame, historical_baseline, neighbor_indices, neighbor_scores)
    candidate_after, candidate_prior = apply_downstream_priors(prior_frame, historical_candidate, neighbor_indices, neighbor_scores)
    prior_category = {}
    for category in sorted(np.unique(categories)):
        mask = categories == category
        prior_category[category] = {
            "baseline_f1": f1(labels[mask], baseline_after[mask]),
            "candidate_f1": f1(labels[mask], candidate_after[mask]),
        }
    prior_baseline_macro = float(np.mean([value["baseline_f1"] for value in prior_category.values()]))
    prior_candidate_macro = float(np.mean([value["candidate_f1"] for value in prior_category.values()]))
    priors = {
        "category": prior_category,
        "baseline_macro_f1": prior_baseline_macro,
        "candidate_macro_f1": prior_candidate_macro,
        "delta_macro_f1": prior_candidate_macro - prior_baseline_macro,
        "changed_before": int((historical_baseline != historical_candidate).sum()),
        "changed_after": int((baseline_after != candidate_after).sum()),
        "baseline_prior_hits": baseline_prior,
        "candidate_prior_hits": candidate_prior,
    }

    core = topologies["historical_connected_safe"]
    repeat_deltas = [topologies[f"repeat_{index}"]["delta_macro_f1"] for index in range(3)]
    direction_ok = all(
        value["corrected"] >= 2 * max(1, value["regressed"])
        for value in core["direction"].values() if value["changed"] > 0
    )
    gates = {
        "connected_delta_at_least_0_003": core["delta_macro_f1"] >= 0.003,
        "connected_wins_at_least_4_of_5": core["folds_won"] >= 4,
        "bad_delta_at_least_0_006": core["category_delta"][BAD] >= 0.006,
        "sports_delta_at_least_0_01": core["sports"]["delta_f1"] >= 0.01,
        "corrected_to_regressed_at_least_2": core["corrected"] >= 2 * max(1, core["regressed"]),
        "direction_corrected_to_regressed_at_least_2": direction_ok,
        "singleton_sports_delta_positive": core["singleton_sports"]["delta_f1"] > 0,
        "bootstrap_probability_at_least_0_90": core["component_bootstrap"]["probability_delta_positive"] >= 0.90,
        "placebo_median_effect_less_than_sports": core["placebo_diagnostic"]["median_absolute_delta_on_changed"] < core["placebo_diagnostic"]["median_sports_delta_on_changed"],
        "all_repeat_deltas_positive": all(value > 0 for value in repeat_deltas),
        "repeat_mean_delta_at_least_0_003": float(np.mean(repeat_deltas)) >= 0.003,
        "positive_after_priors": priors["delta_macro_f1"] > 0,
        "structural_invariants": all(all(value == 0 for value in report["structural_invariants"].values()) for report in topologies.values()),
    }
    result = {
        "experiment_id": "350",
        "evaluation_version": "component_transfer_gate_v4_counterfactual_extension",
        "recipe_frozen_before_scoring": True,
        "topologies": topologies,
        "downstream_priors": priors,
        "acceptance": gates,
        "accepted_for_full_refit": all(gates.values()),
        "runtime_seconds": time.monotonic() - started,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (DATA, QWEN35, QWEN3VL, LOCKED, GUARD, REPEATS)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "counterfactual_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "prior_replay.json").write_text(json.dumps(priors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(OUT / "counterfactual_scores.npz", ids=ids, labels=labels, categories=categories, safe=safe, **arrays)
    print(json.dumps({
        "accepted_for_full_refit": result["accepted_for_full_refit"],
        "connected_delta": core["delta_macro_f1"],
        "sports_delta": core["sports"]["delta_f1"],
        "repeat_deltas": repeat_deltas,
        "after_priors_delta": priors["delta_macro_f1"],
        "runtime_seconds": result["runtime_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

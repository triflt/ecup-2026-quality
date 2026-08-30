from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
RESEARCH = ROOT / "research"
sys.path.insert(0, str(RESEARCH))

from audit_component_decision_survival import (  # noqa: E402
    apply_downstream_priors,
    build_neighbor_graph,
    prepare_frame,
)


DATA = RESEARCH / "data.csv"
FOLDS = ROOT / "validation/grouped_text_v1/folds.csv"
IMAGES = RESEARCH / "multi_image_manifest.tsv.gz"
LOCKED = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
BASE = RESEARCH / "oof-cache-extracted/oof_scores.npz"
FOUR_HEAD = RESEARCH / "four-head-r2-extracted/four_head_oof.npz"
QWEN3VL = RESEARCH / "lora-hard-5fold-robust-fusion-report.npz"
QWEN35 = RESEARCH / "qwen35-hard-5fold-robust-fusion-report.npz"
GEMMA = RESEARCH / "gemma-hard-5fold-robust-fusion-report.npz"
EXP260_FOLDS = ROOT / "experiments/260_bad_family_diverse_positives/artifacts/seed_42_diverse_positives"
OUTPUT = Path(os.environ.get("HARD_ERROR_OUTPUT", Path(__file__).resolve().parent))

BAD = "БАД"
FLAMMABLE = "Легковоспламеняющиеся"


BAD_CUES = {
    "explicit_bad": r"(?iu)(?:\bбад\b|биологически\s+активн\w*\s+добавк|dietary\s+supplement|food\s+supplement)",
    "sports_nutrition": r"(?iu)(?:спортпит|спортивн\w*\s+питан|\bbcaa\b|протеин|гейнер|креатин|аминокислот|предтрен)",
    "vitamin_mineral": r"(?iu)(?:витамин|минерал|магни|цинк|желез|кальци|омега[-\s]?3|коэнзим)",
    "medicine_language": r"(?iu)(?:лекарств|препарат|лечени|терапи|дозиров|противопоказан|показани\w*\s+к\s+применению)",
    "dosage_form": r"(?iu)(?:таблетк|капсул|саше|порошок|сироп|капл[ия]|спрей)",
    "food_beverage": r"(?iu)(?:чай|кофе|напиток|конфет|батончик|желе|сироп|пищев\w*\s+продукт)",
    "pet_veterinary": r"(?iu)(?:для\s+(?:кош|собак|животн)|ветеринар|питомц)",
    "beauty_collagen": r"(?iu)(?:коллаген|гиалурон|кож[аи]|волос|ногт)",
    "probiotic_digestive": r"(?iu)(?:пробиотик|пребиотик|лакто(?:бактери|бифид)|пищевар|кишечник)",
    "not_a_drug": r"(?iu)(?:не\s+является\s+лекарств|не\s+лекарственн)",
}

FLAMMABLE_CUES = {
    "gas_or_cartridge": r"(?iu)\b(?:газ|пропан|бутан|изобутан|баллон|картридж|цангов)\w*\b",
    "liquid_fuel": r"(?iu)(?:\b(?:топлив|бензин|керосин|спирт)\w*\b|жидкост\w*\s+для\s+розжиг)",
    "equipment": r"(?iu)\b(?:горелк|плит|примус|редуктор|адаптер|переходник|шланг|клапан|резак)\w*\b",
    "empty_or_no_fuel": r"(?iu)(?:без\s+(?:газ|топлив|баллон|картридж)|не\s+заправлен|пуст\w*\s+баллон|баллон\w*\s+в\s+комплект\w*\s+не)",
    "fuel_included": r"(?iu)(?:(?:газ|топлив|баллон|картридж)\w*[^.!?]{0,50}(?:в\s+комплект|входит|заправлен)|комплект\w*[^.!?]{0,50}(?:газ|топлив|баллон|картридж))",
    "lighter_matches": r"(?iu)\b(?:зажигалк|спич|огнив|факел)\w*\b",
    "candle_wax": r"(?iu)\b(?:свеч|воск|парафин|фитил)\w*\b",
    "charcoal_wood": r"(?iu)\b(?:угол|уголь|брик|дров|щеп|растопк)\w*\b",
    "aerosol_solvent": r"(?iu)\b(?:аэрозол|растворител|ацетон|лак|краск|клей|очистител)\w*\b",
    "fire_negation": r"(?iu)(?:не\s+горюч|не\s+воспламен|пожаробезопас|не\s+пожароопас)",
    "kit_bundle": r"(?iu)\b(?:комплект|набор|входит|поставк)\w*\b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: object) -> str:
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.lower().replace("ё", "е")
    value = re.sub(r"\d+(?:[.,]\d+)?", " # ", value)
    return re.sub(r"[^a-zа-я#]+", " ", value).strip()


def f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int | float]:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "errors": fp + fn, "f1": f1(labels, predictions)}


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered = labels[order]
    tp = np.cumsum(ordered == 1)
    fp = np.cumsum(ordered == 0)
    fn = int((labels == 1).sum()) - tp
    values = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    best = int(np.argmax(values))
    if best + 1 == len(scores):
        return float(scores[order[best]] - 1e-7)
    return float((scores[order[best]] + scores[order[best + 1]]) / 2)


def crossfit_predictions(
    labels: np.ndarray,
    scores: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(labels), dtype=np.int8)
    for category in sorted(np.unique(categories)):
        for fold in sorted(np.unique(folds)):
            train = (categories == category) & (folds != fold)
            valid = (categories == category) & (folds == fold)
            threshold = best_threshold(labels[train], scores[train])
            result[valid] = (scores[valid] >= threshold).astype(np.int8)
    return result


def fold_category_ranks(values: np.ndarray, folds: np.ndarray, categories: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float32)
    for fold in sorted(np.unique(folds)):
        for category in sorted(np.unique(categories)):
            positions = np.flatnonzero((folds == fold) & (categories == category))
            order = np.argsort(values[positions], kind="mergesort")
            ranks = np.empty(len(positions), dtype=np.float32)
            ranks[order] = np.linspace(0.0, 1.0, len(positions), dtype=np.float32)
            result[positions] = ranks
    return result


def aligned(
    source: np.lib.npyio.NpzFile,
    ids: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    *,
    fold_key: str,
) -> None:
    if not np.array_equal(source["ids"].astype(str), ids):
        raise ValueError("component id mismatch")
    if not np.array_equal(source["labels"].astype(np.int8), labels):
        raise ValueError("component label mismatch")
    if not np.array_equal(source["categories"].astype(str), categories):
        raise ValueError("component category mismatch")
    if fold_key not in source.files:
        raise ValueError(f"missing fold key {fold_key}")
    if not np.array_equal(source[fold_key].astype(np.int8), folds):
        raise ValueError("component fold mismatch")


def load_exp260_rank(ids: np.ndarray, folds: np.ndarray, categories: np.ndarray) -> np.ndarray:
    pieces = []
    for fold in sorted(np.unique(folds)):
        path = EXP260_FOLDS / f"fold_{fold}/lora_holdout_predictions.csv"
        piece = pd.read_csv(path, dtype={"id": str})
        pieces.append(piece)
    frame = pd.concat(pieces, ignore_index=True)
    if frame.id.duplicated().any() or set(frame.id) != set(ids):
        raise ValueError("exp260 OOF ids are incomplete or duplicated")
    frame = frame.set_index("id").loc[ids]
    if not np.array_equal(frame.fold.to_numpy(np.int8), folds):
        raise ValueError("exp260 fold mismatch")
    logits = frame.lora_score.to_numpy(np.float32)
    return fold_category_ranks(logits, folds, categories)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["name"] = frame.name.fillna("").astype(str)
    frame["description"] = frame.description.fillna("").astype(str)
    frame["text"] = frame.name + "\n" + frame.description
    frame["normalized_name_local"] = frame.name.map(normalize)
    frame["normalized_text_local"] = frame.text.map(normalize)
    frame["description_chars"] = frame.description.str.len()
    frame["description_bin"] = pd.cut(
        frame.description_chars,
        bins=[-1, 200, 800, 2000, np.inf],
        labels=["00_0-200", "01_201-800", "02_801-2000", "03_2001+"],
    ).astype(str)
    family = frame.groupby(["category", "group_hash"], sort=False)
    frame["family_size"] = family.id.transform("size")
    frame["family_label_count"] = family.label.transform("nunique")
    frame["family_kind"] = np.select(
        [frame.family_label_count > 1, frame.family_size == 1],
        ["conflicting_repeat", "singleton"],
        default="consistent_repeat",
    )
    names = frame.groupby(["category", "normalized_name_local"], sort=False)
    frame["name_family_size"] = names.id.transform("size")
    frame["name_family_label_count"] = names.label.transform("nunique")

    for cue, pattern in {**BAD_CUES, **FLAMMABLE_CUES}.items():
        frame[cue] = frame.text.str.contains(pattern, regex=True, na=False)

    bad_type = np.select(
        [
            frame.pet_veterinary,
            frame.sports_nutrition,
            frame.probiotic_digestive,
            frame.beauty_collagen,
            frame.vitamin_mineral,
            frame.food_beverage,
            frame.medicine_language,
            frame.explicit_bad,
        ],
        [
            "pet_veterinary",
            "sports_nutrition",
            "probiotic_digestive",
            "beauty_collagen",
            "vitamin_mineral",
            "food_beverage",
            "medicine_language",
            "explicit_bad_other",
        ],
        default="other",
    )
    flammable_type = np.select(
        [
            frame.empty_or_no_fuel | (frame.equipment & ~frame.fuel_included),
            frame.fuel_included,
            frame.liquid_fuel,
            frame.gas_or_cartridge,
            frame.lighter_matches,
            frame.candle_wax,
            frame.charcoal_wood,
            frame.aerosol_solvent,
        ],
        [
            "equipment_or_empty_container",
            "fuel_explicitly_included",
            "liquid_fuel",
            "gas_or_cartridge",
            "lighter_matches",
            "candle_wax",
            "charcoal_wood",
            "aerosol_solvent",
        ],
        default="other",
    )
    frame["product_type"] = np.where(frame.category == BAD, bad_type, flammable_type)
    return frame


def cohort_row(frame: pd.DataFrame, name: str, mask: np.ndarray) -> dict[str, object]:
    local = frame.loc[mask]
    labels = local.label.to_numpy(np.int8)
    baseline = local.baseline_after_prior.to_numpy(np.int8)
    candidate = local.candidate_after_prior.to_numpy(np.int8)
    negatives = labels == 0
    positives = labels == 1
    shared = (baseline != labels) & (candidate != labels)
    return {
        "cohort": name,
        "category": str(local.category.iloc[0]) if len(local) and local.category.nunique() == 1 else "all",
        "rows": int(len(local)),
        "families": int(local.group_hash.nunique()),
        "positives": int(positives.sum()),
        "negatives": int(negatives.sum()),
        "baseline_f1": f1(labels, baseline) if len(local) else None,
        "candidate_f1": f1(labels, candidate) if len(local) else None,
        "baseline_errors": int((baseline != labels).sum()),
        "candidate_errors": int((candidate != labels).sum()),
        "shared_errors": int(shared.sum()),
        "shared_fp": int((shared & negatives).sum()),
        "shared_fn": int((shared & positives).sum()),
        "shared_error_rate": float(shared.mean()) if len(local) else None,
        "shared_fp_rate": float((shared & negatives).sum() / negatives.sum()) if negatives.any() else None,
        "shared_fn_rate": float((shared & positives).sum() / positives.sum()) if positives.any() else None,
    }


def build_cohorts(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category in (BAD, FLAMMABLE):
        category_mask = (frame.category == category).to_numpy()
        rows.append(cohort_row(frame, f"{category}:all", category_mask))
        for label in (0, 1):
            rows.append(cohort_row(frame, f"{category}:label:{label}", category_mask & (frame.label == label).to_numpy()))
        for fold in sorted(frame.fold.unique()):
            rows.append(cohort_row(frame, f"{category}:fold:{fold}", category_mask & (frame.fold == fold).to_numpy()))
        for value in ("singleton", "consistent_repeat", "conflicting_repeat"):
            rows.append(cohort_row(frame, f"{category}:family:{value}", category_mask & (frame.family_kind == value).to_numpy()))
        for value in sorted(frame.image_count.unique()):
            rows.append(cohort_row(frame, f"{category}:images:{value}", category_mask & (frame.image_count == value).to_numpy()))
        for value in ("00_0-200", "01_201-800", "02_801-2000", "03_2001+"):
            rows.append(cohort_row(frame, f"{category}:description:{value}", category_mask & (frame.description_bin == value).to_numpy()))
        for value in sorted(frame.loc[frame.category == category, "product_type"].unique()):
            rows.append(cohort_row(frame, f"{category}:product:{value}", category_mask & (frame.product_type == value).to_numpy()))
        cues = BAD_CUES if category == BAD else FLAMMABLE_CUES
        for cue in cues:
            rows.append(cohort_row(frame, f"{category}:cue:{cue}", category_mask & frame[cue].to_numpy()))
    return pd.DataFrame(rows)


def category_scores(frame: pd.DataFrame, prediction: str) -> dict[str, object]:
    by_category = {}
    for category in (BAD, FLAMMABLE):
        local = frame.loc[frame.category == category]
        by_category[category] = confusion(local.label.to_numpy(), local[prediction].to_numpy())
    return {
        "by_category": by_category,
        "macro_f1": float(np.mean([row["f1"] for row in by_category.values()])),
    }


def json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    headers = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if pd.isna(value):
                rendered.append("")
            elif isinstance(value, float):
                rendered.append(f"{value:.6f}")
            else:
                rendered.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *rows,
        ]
    )


def main() -> None:
    locked = np.load(LOCKED, allow_pickle=False)
    ids = locked["ids"].astype(str)
    labels = locked["labels"].astype(np.int8)
    categories = locked["categories"].astype(str)
    folds = locked["folds"].astype(np.int8)

    frame = pd.read_csv(DATA, dtype={"id": str})
    fold_frame = pd.read_csv(FOLDS, dtype={"id": str})[["id", "fold", "group_hash"]]
    frame = frame.merge(fold_frame, on="id", validate="one_to_one")
    if not np.array_equal(frame.id.to_numpy(), ids):
        raise ValueError("data and locked prediction id order mismatch")
    if not np.array_equal(frame.fold.to_numpy(np.int8), folds):
        raise ValueError("fold registry mismatch")
    if not np.array_equal(frame.label.to_numpy(np.int8), labels):
        raise ValueError("label mismatch")
    images = pd.read_csv(IMAGES, sep="\t", compression="gzip", dtype={"id": str})
    images["image_count"] = images.image_urls.map(lambda value: len(json.loads(value)))
    frame = frame.merge(images[["id", "image_count"]], on="id", validate="one_to_one")
    frame = add_features(frame)

    frame["baseline_before_prior"] = locked["baseline_nested_predictions"].astype(np.int8)
    frame["candidate_before_prior"] = locked["exp260_nested_predictions"].astype(np.int8)

    prior_frame = prepare_frame(DATA, locked)
    neighbor_indices, neighbor_scores = build_neighbor_graph(prior_frame)
    baseline_after, baseline_prior_audit = apply_downstream_priors(
        prior_frame,
        frame.baseline_before_prior.to_numpy(np.int8),
        neighbor_indices,
        neighbor_scores,
    )
    candidate_after, candidate_prior_audit = apply_downstream_priors(
        prior_frame,
        frame.candidate_before_prior.to_numpy(np.int8),
        neighbor_indices,
        neighbor_scores,
    )
    frame["baseline_after_prior"] = baseline_after
    frame["candidate_after_prior"] = candidate_after

    base = np.load(BASE, allow_pickle=True)
    four = np.load(FOUR_HEAD, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    qwen35 = np.load(QWEN35, allow_pickle=True)
    gemma = np.load(GEMMA, allow_pickle=True)
    aligned(base, ids, labels, categories, folds, fold_key="fold_ids")
    aligned(four, ids, labels, categories, folds, fold_key="fold_ids")
    for source in (qwen3vl, qwen35, gemma):
        aligned(source, ids, labels, categories, folds, fold_key="folds")
    component_scores = {
        "text_tfidf": base["text_scores"].astype(np.float32),
        "all_images_embedding": four["all_images"].astype(np.float32),
        "first_image_embedding": four["first_image"].astype(np.float32),
        "extra_trees": four["extra_trees"].astype(np.float32),
        "robust_base_fusion": qwen35["base_rank"].astype(np.float32),
        "qwen3vl": qwen3vl["lora_rank"].astype(np.float32),
        "qwen35_original": qwen35["lora_rank"].astype(np.float32),
        "qwen35_exp260": load_exp260_rank(ids, folds, categories),
        "gemma_e4b": gemma["lora_rank"].astype(np.float32),
    }
    component_predictions = {
        name: crossfit_predictions(labels, score, categories, folds)
        for name, score in component_scores.items()
    }
    for name, predictions in component_predictions.items():
        frame[f"pred_{name}"] = predictions

    shared_before = (
        (frame.baseline_before_prior.to_numpy() != labels)
        & (frame.candidate_before_prior.to_numpy() != labels)
    )
    shared_after = (baseline_after != labels) & (candidate_after != labels)
    component_matrix = np.column_stack(list(component_predictions.values()))
    component_wrong = component_matrix != labels[:, None]
    frame["components_wrong"] = component_wrong.sum(axis=1)
    frame["all_components_wrong"] = component_wrong.all(axis=1)
    frame["any_component_rescues"] = (~component_wrong).any(axis=1)
    frame["shared_before_prior"] = shared_before
    frame["shared_after_prior"] = shared_after
    frame["baseline_error_type"] = np.select(
        [(labels == 0) & (baseline_after == 1), (labels == 1) & (baseline_after == 0)],
        ["FP", "FN"],
        default="correct",
    )
    frame["candidate_error_type"] = np.select(
        [(labels == 0) & (candidate_after == 1), (labels == 1) & (candidate_after == 0)],
        ["FP", "FN"],
        default="correct",
    )

    cohorts = build_cohorts(frame)
    category_totals = cohorts.set_index("cohort")
    fp_lifts, fn_lifts = [], []
    for row in cohorts.itertuples(index=False):
        if row.category in (BAD, FLAMMABLE):
            total = category_totals.loc[f"{row.category}:all"]
            fp_lifts.append(row.shared_fp_rate / total.shared_fp_rate if row.shared_fp_rate is not None and total.shared_fp_rate else None)
            fn_lifts.append(row.shared_fn_rate / total.shared_fn_rate if row.shared_fn_rate is not None and total.shared_fn_rate else None)
        else:
            fp_lifts.append(None)
            fn_lifts.append(None)
    cohorts["shared_fp_lift"] = fp_lifts
    cohorts["shared_fn_lift"] = fn_lifts

    component_summary = []
    for name, predictions in component_predictions.items():
        for category in (BAD, FLAMMABLE):
            category_mask = categories == category
            hard = shared_after & category_mask
            component_summary.append(
                {
                    "component": name,
                    "category": category,
                    "crossfit_f1": f1(labels[category_mask], predictions[category_mask]),
                    "all_errors": int((category_mask & (predictions != labels)).sum()),
                    "shared_hard_rows": int(hard.sum()),
                    "shared_hard_rescued": int((hard & (predictions == labels)).sum()),
                    "shared_hard_rescue_rate": float((predictions[hard] == labels[hard]).mean()) if hard.any() else None,
                }
            )
    component_summary_frame = pd.DataFrame(component_summary)

    family_summary = (
        frame.groupby(["category", "group_hash"], sort=False)
        .agg(
            rows=("id", "size"),
            positives=("label", "sum"),
            label_count=("label", "nunique"),
            shared_errors=("shared_after_prior", "sum"),
            baseline_errors=("baseline_after_prior", lambda values: int((values != frame.loc[values.index, "label"]).sum())),
            candidate_errors=("candidate_after_prior", lambda values: int((values != frame.loc[values.index, "label"]).sum())),
            example_id=("id", "first"),
            example_name=("name", "first"),
        )
        .reset_index()
    )
    family_summary = family_summary.loc[family_summary.shared_errors > 0].sort_values(
        ["shared_errors", "rows"], ascending=[False, False]
    )

    hard_columns = [
        "id", "category", "label", "fold", "group_hash", "family_size", "family_kind",
        "name_family_size", "name_family_label_count", "image_count", "description_chars",
        "description_bin", "product_type", "baseline_before_prior", "candidate_before_prior",
        "baseline_after_prior", "candidate_after_prior", "baseline_error_type",
        "candidate_error_type", "components_wrong", "all_components_wrong",
        "any_component_rescues", *[f"pred_{name}" for name in component_predictions],
        *BAD_CUES.keys(), *FLAMMABLE_CUES.keys(), "name", "description",
    ]
    hard_cases = frame.loc[shared_after, hard_columns].copy()
    hard_cases["description"] = hard_cases.description.str.slice(0, 2000)
    hard_cases = hard_cases.sort_values(
        ["category", "all_components_wrong", "components_wrong", "label", "fold"],
        ascending=[True, False, False, True, True],
    )

    shared_summary = {}
    for category in (BAD, FLAMMABLE):
        category_mask = categories == category
        hard = shared_after & category_mask
        before = shared_before & category_mask
        shared_summary[category] = {
            "shared_errors_before_priors": int(before.sum()),
            "shared_errors_after_priors": int(hard.sum()),
            "shared_fp_after_priors": int((hard & (labels == 0)).sum()),
            "shared_fn_after_priors": int((hard & (labels == 1)).sum()),
            "singleton_shared_errors": int((hard & (frame.family_kind.to_numpy() == "singleton")).sum()),
            "consistent_repeat_shared_errors": int((hard & (frame.family_kind.to_numpy() == "consistent_repeat")).sum()),
            "conflicting_repeat_shared_errors": int((hard & (frame.family_kind.to_numpy() == "conflicting_repeat")).sum()),
            "all_nine_components_wrong": int((hard & frame.all_components_wrong.to_numpy()).sum()),
            "at_least_one_component_correct": int((hard & frame.any_component_rescues.to_numpy()).sum()),
        }
    major_flammable_cues = (
        frame.lighter_matches
        | frame.liquid_fuel
        | frame.candle_wax
        | frame.charcoal_wood
        | frame.equipment
    ).to_numpy()
    shared_summary[FLAMMABLE]["major_semantic_cue_union_errors"] = int(
        (shared_after & (categories == FLAMMABLE) & major_flammable_cues).sum()
    )

    top_cohorts = {}
    for category in (BAD, FLAMMABLE):
        selected = cohorts.loc[
            (cohorts.category == category)
            & ~cohorts.cohort.str.contains(":all$|:label:|:fold:", regex=True)
            & (cohorts.rows >= 20)
            & (cohorts.shared_errors >= 2)
        ].copy()
        top_cohorts[category] = {
            "false_positive_lift": json_records(selected.sort_values(["shared_fp_lift", "shared_fp"], ascending=False).head(12)),
            "false_negative_lift": json_records(selected.sort_values(["shared_fn_lift", "shared_fn"], ascending=False).head(12)),
            "shared_error_volume": json_records(selected.sort_values(["shared_errors", "shared_error_rate"], ascending=False).head(12)),
        }

    report = {
        "scope": "Hard-error audit for locked Public-190 fusion versus exp260 replacement",
        "evaluation": {
            "folds": "grouped_text_v1",
            "thresholds": "each diagnostic component threshold fitted on the other four outer folds",
            "priors": "donor-only outer-fold replay of exact/name/numeric-family/shingle production rules",
            "rows": int(len(frame)),
        },
        "input_sha256": {
            "data": sha256(DATA),
            "folds": sha256(FOLDS),
            "locked_predictions": sha256(LOCKED),
            "image_manifest": sha256(IMAGES),
        },
        "ensemble_metrics": {
            "baseline_before_priors": category_scores(frame, "baseline_before_prior"),
            "candidate_before_priors": category_scores(frame, "candidate_before_prior"),
            "baseline_after_priors": category_scores(frame, "baseline_after_prior"),
            "candidate_after_priors": category_scores(frame, "candidate_after_prior"),
        },
        "prior_audit": {
            "baseline": baseline_prior_audit,
            "candidate": candidate_prior_audit,
        },
        "shared_hard_errors": shared_summary,
        "component_oracle": {
            "components": list(component_predictions),
            "important_warning": "Rescue counts are diagnostic oracle counts, not an honest routing gain.",
            "summary": json_records(component_summary_frame),
        },
        "top_cohorts": top_cohorts,
        "largest_error_families": json_records(family_summary.head(30)),
        "limitations": [
            "The locked OOF reconstruction estimates experiment 190 on train folds; it cannot identify hidden Public rows.",
            "Regex product types are deterministic diagnostic slices, not ground-truth semantic annotations.",
            "Component rescue counts use labels after the fact and must not be treated as a deployable router score.",
            "Equal Public Macro F1 for experiments 190 and 280 does not imply identical row predictions.",
        ],
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cohorts.to_csv(OUTPUT / "cohort_metrics.csv", index=False)
    component_summary_frame.to_csv(OUTPUT / "component_summary.csv", index=False)
    family_summary.to_csv(OUTPUT / "error_families.csv", index=False)
    hard_cases.to_csv(OUTPUT / "hard_error_cases.csv", index=False)
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bad = shared_summary[BAD]
    fire = shared_summary[FLAMMABLE]
    bad_product = cohorts.loc[cohorts.cohort.str.startswith(f"{BAD}:product:")].sort_values("shared_errors", ascending=False).head(6)
    fire_product = cohorts.loc[cohorts.cohort.str.startswith(f"{FLAMMABLE}:product:")].sort_values("shared_errors", ascending=False).head(8)
    component_best = component_summary_frame.sort_values(
        ["category", "shared_hard_rescued"], ascending=[True, False]
    ).groupby("category", sort=False).head(3)
    markdown = f"""# Глубокий разбор общих ошибок 190 и 260

## Что именно измерено

Проверка использует все 12 971 внеобучающих прогноза `locked_190_nested_v1`.
Семейные правила `190` воспроизведены отдельно: для каждого outer fold донорами
служат только остальные четыре fold. Пороги отдельных компонентов также подбираются
только на остальных четырёх folds. Поэтому цифры ниже — честная диагностика train OOF,
но не реконструкция строк скрытого Public.

## Главный результат

- До семейных правил оба ансамбля одновременно ошибались на {bad['shared_errors_before_priors']} БАД и {fire['shared_errors_before_priors']} flammable-карточках.
- После правил осталось {bad['shared_errors_after_priors']} общих ошибок БАД ({bad['shared_fp_after_priors']} FP, {bad['shared_fn_after_priors']} FN) и {fire['shared_errors_after_priors']} flammable ({fire['shared_fp_after_priors']} FP, {fire['shared_fn_after_priors']} FN).
- В БАД {bad['singleton_shared_errors']} общих ошибок приходятся на singleton-семейства, {bad['conflicting_repeat_shared_errors']} — на семейства с конфликтующими метками.
- Во flammable {fire['singleton_shared_errors']} общих ошибок приходятся на singleton-семейства; конфликтующих повторов среди общих ошибок — {fire['conflicting_repeat_shared_errors']}.
- Все девять диагностических компонентов одновременно ошибаются на {bad['all_nine_components_wrong']} БАД и {fire['all_nine_components_wrong']} flammable-карточках. Хотя бы один компонент знает правильный ответ для {bad['at_least_one_component_correct']} и {fire['at_least_one_component_correct']} карточек соответственно. Это только oracle-диагностика: правила маршрутизации здесь ещё нет.

## Продуктовые когорты БАД

{markdown_table(bad_product[['cohort','rows','positives','shared_errors','shared_fp','shared_fn','shared_fp_lift','shared_fn_lift']])}

## Продуктовые когорты flammable

{markdown_table(fire_product[['cohort','rows','positives','shared_errors','shared_fp','shared_fn','shared_fp_lift','shared_fn_lift']])}

## Какие независимые компоненты чаще спасают общий промах

{markdown_table(component_best[['category','component','crossfit_f1','shared_hard_rows','shared_hard_rescued','shared_hard_rescue_rate']])}

## Проверяемые гипотезы

1. **БАД: отделить регуляторный статус от общего “полезного” содержания.** Спортивное
   питание, витамины, коллаген, обычная еда и лекарственная лексика образуют близкие
   тексты, но метка зависит от статуса товара. Следующий компонент должен извлекать
   явные свидетельства `БАД / не лекарство / форма выпуска / пищевая ценность`, а
   обучение — использовать пары одного семейства с противоположными метками. Проверка:
   заранее зафиксированные product cohorts и отдельно FP/FN, без изменения flammable.

2. **Flammable: решать наличие горючего содержимого, а не узнавать объект.** Баллон,
   горелка, плита и зажигалка визуально и лексически похожи независимо от того, входит
   ли газ или топливо в комплект. MiniCPM/InternVL следует обучать на промежуточных
   атрибутах `вещество / пустая ёмкость / оборудование / топливо включено / отрицание`,
   затем применять фиксированное правило. Проверка: equipment-vs-fuel cohorts,
   положительная полнота и ложные срабатывания считаются раздельно.

3. **Не усреднять независимое правильное меньшинство вслепую.** Для значительной части
   общих ошибок хотя бы один компонент уже прав, но позднее среднее его подавляет.
   Возможный следующий шаг — label-blind uncertainty gate: согласие text и независимого
   vision attribute-head, расстояние до порога и устойчивость к повторному seed. Gate
   фиксируется на внутренних folds и проверяется один раз на outer folds; oracle-выбор
   правильной модели по метке запрещён.

## Артефакты

- `report.json` — полная сводка и контрольные суммы;
- `cohort_metrics.csv` — категории, типы товаров, сигналы, folds, длина и число изображений;
- `component_summary.csv` — честные cross-fit метрики и oracle rescue-анализ;
- `error_families.csv` — концентрация ошибок по семействам;
- `hard_error_cases.csv` — все общие ошибки после production-priors с прогнозами компонентов.
"""
    (OUTPUT / "REPORT.generated.md").write_text(markdown, encoding="utf-8")
    diagnostic_cohorts = cohorts.loc[
        cohorts.cohort.str.contains(":product:|:family:|:description:|:images:|:cue:", regex=True)
        & (cohorts.shared_errors >= 2),
        [
            "cohort", "rows", "positives", "shared_errors", "shared_fp", "shared_fn",
            "baseline_f1", "candidate_f1", "shared_fp_lift", "shared_fn_lift",
        ],
    ]
    output_files = sorted(path for path in OUTPUT.iterdir() if path.is_file())
    print(
        json.dumps(
            {
                "shared_hard_errors": shared_summary,
                "diagnostic_cohorts": json_records(diagnostic_cohorts),
                "component_summary": json_records(component_summary_frame),
                "largest_error_families": json_records(family_summary.head(10)),
                "outputs": [path.name for path in output_files],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

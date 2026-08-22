from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS = ROOT / "experiments/400_qwen35_category_routed_adapters/results/routed_predictions.npz"
DATA = ROOT / "research/data.csv"
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
IMAGE_MANIFEST = ROOT / "research/lora_image_manifest_complete.tsv.gz"
OUT = Path(__file__).resolve().parent

COHORT_PATTERNS = {
    "gas_fuel": r"\b(?:газ|газов|пропан|бутан|баллон|бензин|топлив|керосин|солярк|дизел)\w*",
    "ignition_source": r"\b(?:зажигал|спич|огнив|горелк|пьезо|розжиг)\w*",
    "paint_solvent": r"\b(?:краск|лак|растворител|ацетон|эмал|грунтовк)\w*",
    "alcohol_perfume": r"\b(?:спирт|алкогол|этанол|парфюм|духи|одеколон)\w*",
    "pyrotechnics": r"\b(?:пиротех|фейерверк|бенгаль|хлопуш|петард)\w*",
    "empty_equipment": r"\b(?:пуст|без\s+(?:газа|баллон|топлив)|оборудован|плит|печ|ламп)\w*",
    "battery_electronics": r"\b(?:аккумулятор|батаре|литий|зарядн|электр)\w*",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: object) -> str:
    value = re.sub(r"<[^>]+>", " ", str(value or "")).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    arrays = np.load(PREDICTIONS, allow_pickle=False)
    frame = pd.read_csv(DATA, dtype={"id": str})
    guard = pd.read_csv(GUARD, dtype={"id": str, "connected_component": str})
    ids = arrays["ids"].astype(str)
    labels = arrays["labels"].astype(np.int8)
    categories = arrays["categories"].astype(str)
    folds = arrays["folds"].astype(np.int8)
    if not np.array_equal(frame.id.to_numpy(), ids):
        raise ValueError("data id mismatch")
    if not np.array_equal(guard.id.to_numpy(), ids):
        raise ValueError("guard id mismatch")
    baseline = arrays["baseline_nested_predictions"].astype(np.int8)
    candidate = arrays["category_routed_nested_predictions"].astype(np.int8)
    changed = baseline != candidate

    image_counts = pd.Series(0, index=ids, dtype=np.int32)
    if IMAGE_MANIFEST.exists():
        manifest = pd.read_csv(IMAGE_MANIFEST, sep="\t", compression="gzip", dtype={"id": str})
        if "id" in manifest.columns:
            if "image_path" in manifest.columns:
                image_counts = manifest.groupby("id").image_path.count().reindex(ids, fill_value=0)
            elif "image_url" in manifest.columns:
                image_counts = manifest.groupby("id").image_url.count().reindex(ids, fill_value=0)
            elif "images" in manifest.columns:
                image_counts = manifest.set_index("id").images.fillna("").map(
                    lambda value: len([part for part in str(value).split("|") if part])
                ).reindex(ids, fill_value=0)

    output = frame.loc[changed, ["id", "name", "description", "category"]].copy()
    positions = np.flatnonzero(changed)
    output["label"] = labels[changed]
    output["fold"] = folds[changed]
    output["baseline_prediction"] = baseline[changed]
    output["candidate_prediction"] = candidate[changed]
    output["transition"] = [f"{old}->{new}" for old, new in zip(baseline[changed], candidate[changed])]
    output["outcome"] = np.where(candidate[changed] == labels[changed], "corrected", "regressed")
    output["connected_safe"] = guard.safe_for_selection.to_numpy(bool)[changed]
    output["connected_component"] = guard.connected_component.to_numpy()[changed]
    component_sizes = guard.groupby("connected_component").size()
    output["component_size"] = [int(component_sizes[value]) for value in output.connected_component]
    output["image_count"] = image_counts.iloc[positions].to_numpy(np.int32)
    output["description_chars"] = output.description.fillna("").astype(str).str.len()
    text = [normalize_text(f"{name} {description}") for name, description in zip(output.name, output.description)]
    for cohort, pattern in COHORT_PATTERNS.items():
        output[cohort] = [bool(re.search(pattern, value)) for value in text]
    cohort_columns = list(COHORT_PATTERNS)
    output["primary_cohort"] = [
        next((name for name in cohort_columns if bool(row[name])), "other")
        for _, row in output.iterrows()
    ]
    output.to_csv(OUT / "changed_decisions.csv", index=False)

    cohort_rows = []
    for cohort in [*cohort_columns, "other"]:
        mask = output.primary_cohort == cohort
        local = output.loc[mask]
        cohort_rows.append({
            "cohort": cohort,
            "rows": int(len(local)),
            "corrected": int((local.outcome == "corrected").sum()),
            "regressed": int((local.outcome == "regressed").sum()),
            "net": int((local.outcome == "corrected").sum() - (local.outcome == "regressed").sum()),
            "fold3_rows": int((local.fold == 3).sum()),
        })
    pd.DataFrame(cohort_rows).to_csv(OUT / "cohort_summary.csv", index=False)

    fold_rows = []
    for fold in sorted(output.fold.unique()):
        local = output[output.fold == fold]
        fold_rows.append({
            "fold": int(fold),
            "changed": int(len(local)),
            "corrected": int((local.outcome == "corrected").sum()),
            "regressed": int((local.outcome == "regressed").sum()),
            "false_positive_to_negative": int(((local.label == 0) & (local.transition == "1->0")).sum()),
            "false_negative_to_positive": int(((local.label == 1) & (local.transition == "0->1")).sum()),
        })
    report = {
        "experiment_id": "400",
        "scope": "all 26 locked nested decisions changed by category routing",
        "rows": int(len(output)),
        "corrected": int((output.outcome == "corrected").sum()),
        "regressed": int((output.outcome == "regressed").sum()),
        "transitions": output.transition.value_counts().to_dict(),
        "label_distribution": {str(key): int(value) for key, value in output.label.value_counts().items()},
        "connected_safe_rows": int(output.connected_safe.sum()),
        "singleton_components": int((output.component_size == 1).sum()),
        "folds": fold_rows,
        "cohorts": cohort_rows,
        "fold3_regressions": output.loc[
            (output.fold == 3) & (output.outcome == "regressed"),
            ["id", "name", "label", "transition", "primary_cohort", "component_size", "image_count"],
        ].to_dict(orient="records"),
        "input_sha256": {
            str(PREDICTIONS.relative_to(ROOT)): sha256(PREDICTIONS),
            str(DATA.relative_to(ROOT)): sha256(DATA),
            str(GUARD.relative_to(ROOT)): sha256(GUARD),
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

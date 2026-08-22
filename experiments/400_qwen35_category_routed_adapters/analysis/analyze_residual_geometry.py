from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREDICTIONS = (
    ROOT
    / "experiments/400_qwen35_category_routed_adapters/results/routed_predictions.npz"
)
DEFAULT_DATA = ROOT / "research/data.csv"
DEFAULT_GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent

COHORTS = {
    "gas_or_liquid_fuel": r"(?iu)\b(?:газ\w*|баллон\w*|бензин\w*|керосин\w*|топлив\w*|парафин\w*|спирт\w*)\b|жидк\w*\s+для\s+розжиг",
    "ignition_or_pyrotechnics": r"(?iu)\b(?:спич\w*|зажигал\w*|огнив\w*|горелк\w*|свеч\w*|бенгал\w*|фейерверк\w*|салют\w*|петард\w*|пиротех\w*|дымогенератор\w*)\b",
    "solid_fuel": r"(?iu)\b(?:угол\w*|уголь\w*|дров\w*|брик\w*|растопк\w*|розжиг\w*|ролл\w*)\b",
    "included_or_kit_scope": r"(?iu)\b(?:набор\w*|комплект\w*|входит|входят|в\s+комплекте|вместе\s+с)\b",
    "empty_equipment_scope": r"(?iu)\b(?:без\s+(?:газа|баллон\w*|топлив\w*)|пуст\w*|насадк\w*|резак\w*|оборудован\w*)\b",
    "explicit_bad_marker": r"(?iu)(?:\bбад\b|биологически\s+активн\w*\s+добавк\w*|dietary\s+supplement)",
    "sports_nutrition": r"(?iu)\b(?:протеин\w*|креатин\w*|bcaa|бцаа|аминокислот\w*|гейнер\w*|спорт\w*\s+питан\w*)\b",
    "raw_herb_or_ingredient": r"(?iu)\b(?:корень|трава|молот\w*|семена|сырь[её]|пищев\w*\s+продукт)\b",
    "dosage_form": r"(?iu)\b(?:капсул\w*|таблет\w*|порош\w*|саше|стик\w*|драже)\b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--guard", type=Path, default=DEFAULT_GUARD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    archive = np.load(args.predictions, allow_pickle=False)
    data = pd.read_csv(args.data, dtype={"id": str}).fillna("")
    guard = pd.read_csv(args.guard, dtype={"id": str, "connected_component": str})
    ids = archive["ids"].astype(str)
    if not np.array_equal(data.id.astype(str).to_numpy(), ids):
        raise ValueError("data ids are not aligned with experiment-400 predictions")
    if not np.array_equal(guard.id.astype(str).to_numpy(), ids):
        raise ValueError("connected guard ids are not aligned with predictions")

    labels = archive["labels"].astype(np.int8)
    predictions = archive["category_routed_nested_predictions"].astype(np.int8)
    categories = archive["categories"].astype(str)
    folds = archive["folds"].astype(np.int8)
    text = data["name"].astype(str) + "\n" + data["description"].astype(str)
    output = data[["id", "name", "description"]].copy()
    output["category"] = categories
    output["label"] = labels
    output["prediction"] = predictions
    output["fold"] = folds
    output["error_type"] = np.where(labels == 1, "false_negative", "false_positive")
    output["connected_safe"] = guard.safe_for_selection.astype(bool).to_numpy()
    output["connected_component"] = guard.connected_component.astype(str).to_numpy()
    output["component_size"] = guard.component_size.astype(int).to_numpy()
    mixed = guard.assign(label=labels).groupby("connected_component").label.transform("nunique") > 1
    output["mixed_label_component"] = mixed.to_numpy()
    for cohort, pattern in COHORTS.items():
        output[cohort] = text.str.contains(re.compile(pattern), na=False).to_numpy()
    errors = output[labels != predictions].copy()

    category_report: dict[str, object] = {}
    for category in sorted(np.unique(categories)):
        local = errors[errors.category == category]
        cohort_summary = {}
        for cohort in COHORTS:
            selected = local[local[cohort]]
            cohort_summary[cohort] = {
                "rows": len(selected),
                "false_negatives": int((selected.error_type == "false_negative").sum()),
                "false_positives": int((selected.error_type == "false_positive").sum()),
            }
        category_report[category] = {
            "errors": len(local),
            "false_negatives": int((local.error_type == "false_negative").sum()),
            "false_positives": int((local.error_type == "false_positive").sum()),
            "connected_safe_errors": int(local.connected_safe.sum()),
            "connected_unsafe_errors": int((~local.connected_safe).sum()),
            "mixed_label_component_errors": int(local.mixed_label_component.sum()),
            "cohorts": cohort_summary,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = args.output_dir / "residual_errors.csv"
    report_path = args.output_dir / "residual_error_report.json"
    errors.to_csv(errors_path, index=False)
    report = {
        "experiment_id": "400",
        "analysis": "posthoc_descriptive_residual_error_geometry_v1",
        "descriptive_only": True,
        "selection_authority": False,
        "warning": "Cohorts were inspected after labels and cannot authorize a rule or threshold without a separately frozen validation experiment.",
        "rows": len(data),
        "errors": len(errors),
        "categories": category_report,
        "key_observations": {
            "all_flammable_residual_errors_connected_safe": bool(
                errors.loc[errors.category == "Легковоспламеняющиеся", "connected_safe"].all()
            ),
            "bad_error_share_in_mixed_label_components": float(
                errors.loc[errors.category == "БАД", "mixed_label_component"].mean()
            ),
        },
        "input_sha256": {
            str(args.predictions.relative_to(ROOT)): sha256(args.predictions),
            str(args.data.relative_to(ROOT)): sha256(args.data),
            str(args.guard.relative_to(ROOT)): sha256(args.guard),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report["output_sha256"] = {"errors": sha256(errors_path)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

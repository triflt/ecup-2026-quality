from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/data.csv"
FOLDS = ROOT / "validation/grouped_text_v1/folds.csv"
QWEN3VL = ROOT / "research/lora-hard-5fold-robust-fusion-report.npz"
QWEN35 = ROOT / "research/qwen35-hard-5fold-robust-fusion-report.npz"
LOCKED = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
MANIFEST = (
    ROOT
    / "experiments/260_bad_family_diverse_positives/.local/compute/inputs/lora_image_manifest.tsv.gz"
)
FLAMMABLE = "Легковоспламеняющиеся"
PRODUCTION_THRESHOLD = 0.953912615776062
WEIGHTS = np.asarray([0.15, 0.10, 0.75], dtype=np.float32)


CUES = {
    "fuel_or_gas": r"(?iu)\b(?:газ|пропан|бутан|изобутан|баллон|картридж|цангов|топлив|бензин|керосин|спирт)\w*\b",
    "empty_equipment": r"(?iu)(?:\b(?:горелк|плит|примус|редуктор|адаптер|переходник|шланг|клапан|резак)\w*\b|без\s+(?:газ|топлив|баллон|картридж)|не\s+заправлен|пуст\w*\s+баллон)",
    "ignition_source": r"(?iu)\b(?:зажигалк|спич|огнив|факел|свеч|фитил)\w*\b",
    "fuel_included": r"(?iu)(?:(?:газ|топлив|баллон|картридж)\w*[^.!?]{0,50}(?:в\s+комплект|входит|заправлен)|комплект\w*[^.!?]{0,50}(?:газ|топлив|баллон|картридж))",
    "combustible_material": r"(?iu)\b(?:угол|уголь|брик|дров|щеп|растопк|воск|парафин)\w*\b",
    "absence_or_negation": r"(?iu)(?:без\s+(?:газ|топлив|баллон|картридж)|не\s+заправлен|не\s+входит|в\s+комплект\w*\s+не|не\s+горюч|не\s+воспламен)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.fraction < 1:
        raise ValueError("fraction must be in (0, 1)")

    qwen35 = np.load(QWEN35, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    locked = np.load(LOCKED, allow_pickle=False)
    ids = qwen35["ids"].astype(str)
    categories = qwen35["categories"].astype(str)
    labels = qwen35["labels"].astype(np.int8)
    folds = qwen35["folds"].astype(np.int8)
    for name, source in (("qwen3vl", qwen3vl), ("locked", locked)):
        for field, expected in (
            ("ids", ids),
            ("categories", categories),
            ("labels", labels),
            ("folds", folds),
        ):
            if not np.array_equal(source[field].astype(expected.dtype), expected):
                raise ValueError(f"{name} {field} mismatch")

    matrix = np.column_stack(
        [
            qwen35["base_rank"].astype(np.float32),
            qwen3vl["lora_rank"].astype(np.float32),
            qwen35["lora_rank"].astype(np.float32),
        ]
    )
    scores = np.sum(matrix * WEIGHTS[None, :], axis=1, dtype=np.float32)
    uncertainty = np.abs(scores - PRODUCTION_THRESHOLD)
    flammable = categories == FLAMMABLE
    cutoff = float(np.quantile(uncertainty[flammable], args.fraction))

    frame = pd.read_csv(DATA, dtype={"id": str}).set_index("id").loc[ids].reset_index()
    fold_frame = pd.read_csv(FOLDS, dtype={"id": str}).set_index("id").loc[ids]
    if not np.array_equal(fold_frame.fold.to_numpy(np.int8), folds):
        raise ValueError("fold registry mismatch")
    images = pd.read_csv(MANIFEST, sep="\t", compression="gzip", dtype={"id": str})
    frame = frame.merge(images[["id", "image_url"]], on="id", validate="one_to_one")
    frame["fold"] = folds
    frame["group_hash"] = fold_frame.group_hash.astype(str).to_numpy()
    frame["locked_score"] = scores
    frame["locked_uncertainty"] = uncertainty
    text = frame.name.fillna("").astype(str) + "\n" + frame.description.fillna("").astype(str)
    cue_columns = []
    for name, pattern in CUES.items():
        frame[f"cue_{name}"] = text.str.contains(pattern, regex=True, na=False)
        cue_columns.append(f"cue_{name}")
    relevant = frame[cue_columns].any(axis=1).to_numpy()
    selected = flammable & relevant & (uncertainty <= cutoff)
    frame["selected"] = selected

    output_columns = [
        "id",
        "fold",
        "group_hash",
        "name",
        "description",
        "image_url",
        "locked_score",
        "locked_uncertainty",
        *cue_columns,
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[selected, output_columns].to_csv(
        args.output, index=False, compression="gzip"
    )

    baseline = locked["baseline_nested_predictions"].astype(np.int8)
    baseline_error = baseline != labels
    flammable_errors = flammable & baseline_error
    report = {
        "selector_version": "internvl_attribute_probe_selector_v1",
        "selection_uses_labels": False,
        "category": FLAMMABLE,
        "uncertainty_fraction": args.fraction,
        "uncertainty_cutoff": cutoff,
        "required_text_cue_union": list(CUES),
        "category_rows": int(flammable.sum()),
        "uncertainty_rows_before_cues": int((flammable & (uncertainty <= cutoff)).sum()),
        "selected_rows": int(selected.sum()),
        "selected_fraction_of_category": float(selected.sum() / flammable.sum()),
        "selected_by_fold": {
            str(fold): int((selected & (folds == fold)).sum())
            for fold in sorted(np.unique(folds))
        },
        "post_selection_diagnostic_not_used_by_selector": {
            "baseline_flammable_errors": int(flammable_errors.sum()),
            "selected_baseline_errors": int((selected & baseline_error).sum()),
            "error_coverage": float(
                (selected & baseline_error).sum() / max(1, flammable_errors.sum())
            ),
            "selected_positives": int((selected & (labels == 1)).sum()),
        },
        "input_sha256": {
            "data": sha256(DATA),
            "folds": sha256(FOLDS),
            "qwen35": sha256(QWEN35),
            "qwen3vl": sha256(QWEN3VL),
            "locked": sha256(LOCKED),
            "image_manifest": sha256(MANIFEST),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

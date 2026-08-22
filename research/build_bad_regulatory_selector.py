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
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
QWEN3VL = ROOT / "research/lora-hard-5fold-robust-fusion-report.npz"
QWEN35 = ROOT / "research/qwen35-hard-5fold-robust-fusion-report.npz"
LOCKED = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
BAD = "БАД"
PRODUCTION_THRESHOLD = 0.27193570137023926
WEIGHTS = np.asarray([0.50, 0.25, 0.25], dtype=np.float32)

CUES = {
    "sports_nutrition": r"(?iu)(?:спортпит|спортивн\w*\s+питан|\bbcaa\b|протеин|гейнер|креатин|аминокислот|предтрен)",
    "explicit_bad": r"(?iu)(?:\bбад\b|биологически\s+активн\w*\s+добавк|dietary\s+supplement|food\s+supplement)",
    "medicine_language": r"(?iu)(?:лекарств|препарат|лечени|терапи|дозиров|противопоказан|показани\w*\s+к\s+применению)",
    "dosage_form": r"(?iu)(?:таблетк|капсул|саше|порошок|сироп|капл[ия]|спрей)",
    "food_beverage": r"(?iu)(?:чай|кофе|напиток|конфет|батончик|желе|сироп|пищев\w*\s+продукт)",
    "pet_veterinary": r"(?iu)(?:для\s+(?:кош|собак|животн)|ветеринар|питомц)",
    "not_a_drug": r"(?iu)(?:не\s+является\s+лекарств|не\s+лекарственн)",
}
SELECTOR_CUES = [
    "sports_nutrition",
    "medicine_language",
    "food_beverage",
    "pet_veterinary",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_matches(values: pd.Series, pattern: str) -> pd.Series:
    compiled = re.compile(pattern)
    return values.map(lambda value: len(compiled.findall(str(value or "")))).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(DATA, dtype={"id": str})
    folds = pd.read_csv(FOLDS, dtype={"id": str, "group_hash": str}).set_index("id")
    guard = pd.read_csv(
        GUARD, dtype={"id": str, "group_hash": str, "connected_component": str}
    ).set_index("id")
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    qwen35 = np.load(QWEN35, allow_pickle=True)
    locked = np.load(LOCKED, allow_pickle=False)
    ids = locked["ids"].astype(str)
    labels = locked["labels"].astype(np.int8)
    categories = locked["categories"].astype(str)
    fold_values = locked["folds"].astype(np.int8)
    if not np.array_equal(frame.id.to_numpy(), ids):
        raise ValueError("data ids do not match locked evaluation")
    for name, source in (("qwen3vl", qwen3vl), ("qwen35", qwen35)):
        for field, expected in (
            ("ids", ids),
            ("labels", labels),
            ("categories", categories),
            ("folds", fold_values),
        ):
            if not np.array_equal(source[field].astype(expected.dtype), expected):
                raise ValueError(f"{name} {field} mismatch")
    folds = folds.loc[ids]
    guard = guard.loc[ids]
    if not np.array_equal(folds.fold.to_numpy(np.int8), fold_values):
        raise ValueError("fold registry mismatch")
    if not np.array_equal(guard.fold.to_numpy(np.int8), fold_values):
        raise ValueError("connected guard fold mismatch")
    scores = np.sum(
        np.column_stack(
            [
                qwen35["base_rank"].astype(np.float32),
                qwen3vl["lora_rank"].astype(np.float32),
                qwen35["lora_rank"].astype(np.float32),
            ]
        )
        * WEIGHTS[None, :],
        axis=1,
        dtype=np.float32,
    )
    name = frame.name.fillna("").astype(str)
    description = frame.description.fillna("").astype(str)
    text = name + "\n" + description
    features: dict[str, pd.Series] = {}
    for cue, pattern in CUES.items():
        features[f"name_{cue}"] = name.str.contains(pattern, regex=True, na=False)
        features[f"description_{cue}"] = description.str.contains(
            pattern, regex=True, na=False
        )
        features[f"count_{cue}"] = count_matches(text, pattern)
        features[cue] = features[f"name_{cue}"] | features[f"description_{cue}"]
    selector = (categories == BAD) & np.column_stack(
        [features[cue].to_numpy(bool) for cue in SELECTOR_CUES]
    ).any(axis=1)
    manifest = pd.DataFrame(
        {
            "id": ids,
            "fold": fold_values,
            "group_hash": folds.group_hash.astype(str).to_numpy(),
            "connected_component": guard.connected_component.astype(str).to_numpy(),
            "safe_for_selection": guard.safe_for_selection.astype(bool).to_numpy(),
            "locked_score": scores,
            "locked_uncertainty": np.abs(scores - PRODUCTION_THRESHOLD),
            **features,
        }
    )
    # These interactions are frozen before head training and are the only dense
    # regulatory interactions allowed in experiment 320.
    manifest["sports_x_explicit_bad"] = (
        manifest.sports_nutrition & manifest.explicit_bad
    )
    manifest["sports_x_not_a_drug"] = (
        manifest.sports_nutrition & manifest.not_a_drug
    )
    manifest["sports_x_dosage_form"] = (
        manifest.sports_nutrition & manifest.dosage_form
    )
    manifest["sports_x_medicine"] = (
        manifest.sports_nutrition & manifest.medicine_language
    )
    manifest["sports_x_food"] = manifest.sports_nutrition & manifest.food_beverage
    selected = manifest.loc[selector].copy()
    if selected.id.duplicated().any() or not set(selected.id).issubset(set(ids)):
        raise ValueError("invalid regulatory selector ids")
    # Labels are deliberately absent from the frozen selector artifact.
    if "label" in selected.columns or "prediction" in selected.columns:
        raise ValueError("selector artifact must be label-blind")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(
        args.output,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )

    baseline = locked["baseline_nested_predictions"].astype(np.int8)
    baseline_error = baseline != labels
    safe = guard.safe_for_selection.astype(bool).to_numpy()
    sports = features["sports_nutrition"].to_numpy(bool) & (categories == BAD)
    report = {
        "selector_version": "bad_regulatory_selector_v1",
        "selection_uses_labels": False,
        "category": BAD,
        "selector_cues": SELECTOR_CUES,
        "frozen_feature_cues": list(CUES),
        "frozen_interactions": [
            "sports_x_explicit_bad",
            "sports_x_not_a_drug",
            "sports_x_dosage_form",
            "sports_x_medicine",
            "sports_x_food",
        ],
        "bad_rows": int((categories == BAD).sum()),
        "selected_rows": int(selector.sum()),
        "selected_safe_rows": int((selector & safe).sum()),
        "selected_unsafe_rows": int((selector & ~safe).sum()),
        "selected_by_fold": {
            str(fold): int((selector & (fold_values == fold)).sum())
            for fold in sorted(np.unique(fold_values))
        },
        "selected_by_cue": {
            cue: int((selector & features[cue].to_numpy(bool)).sum())
            for cue in SELECTOR_CUES
        },
        "post_selection_diagnostic_not_used_by_selector": {
            "selected_positives": int((selector & (labels == 1)).sum()),
            "selected_baseline_errors": int((selector & baseline_error).sum()),
            "sports_rows": int(sports.sum()),
            "sports_baseline_errors": int((sports & baseline_error).sum()),
            "sports_safe_rows": int((sports & safe).sum()),
            "sports_safe_baseline_errors": int((sports & safe & baseline_error).sum()),
        },
        "input_sha256": {
            "data": sha256(DATA),
            "folds": sha256(FOLDS),
            "guard": sha256(GUARD),
            "qwen3vl": sha256(QWEN3VL),
            "qwen35": sha256(QWEN35),
            "locked": sha256(LOCKED),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

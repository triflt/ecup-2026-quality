from __future__ import annotations

import gzip
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("research")
DATA = ROOT / "data.csv"
BASE = ROOT / "oof-cache-extracted/oof_scores.npz"
LORA = ROOT / "lora-hard-5fold-robust-fusion-report.npz"
MANIFEST = ROOT / "multi_image_manifest.tsv.gz"
REPORT = ROOT / "lora_error_cohort_report.json"
ERRORS = ROOT / "lora_error_cases.csv"

PATTERNS = {
    "bad_explicit": re.compile(
        r"(?iu)(?:\bбад\b|биологически\s+активн\w*\s+добавк|dietary\s+supplement|food\s+supplement)"
    ),
    "sports": re.compile(
        r"(?iu)(?:спортивн\w*\s+питан|спортпит|\bbcaa\b|\bпротеин|гейнер|предтрен|"
        r"креатин|л[-\s]?карнитин|l[-\s]?carnitine|аминокислот)"
    ),
    "flame_source": re.compile(
        r"(?iu)(?:зажигалк|спич(?:к|еч)|сух\w*\s+горюч|жидкост\w*\s+для\s+розжиг|"
        r"газов\w*\s+(?:баллон|картридж)|пропан|бутан|керосин|бензин|топлив)"
    ),
    "equipment": re.compile(
        r"(?iu)(?:мангал|грил|газов\w*\s+плит|горелк|печ(?:ь|ка)|насадк\w*\s+(?:на|для)\s+баллон)"
    ),
    "kit": re.compile(r"(?iu)(?:в\s+комплект|комплект\w*\s+с|входит\s+в\s+комплект)"),
}


def normalize(value: str) -> str:
    value = html.unescape(str(value or "")).lower()
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\W+", " ", value, flags=re.U).strip()


def f1(labels, predictions) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def metrics(frame: pd.DataFrame, mask: np.ndarray) -> dict:
    local = frame.loc[mask]
    if local.empty:
        return {"rows": 0}
    return {
        "rows": len(local),
        "positive": int(local.label.sum()),
        "positive_rate": float(local.label.mean()),
        "base_f1": f1(local.label, local.base_prediction),
        "lora_f1": f1(local.label, local.lora_prediction),
        "base_errors": int((local.label != local.base_prediction).sum()),
        "lora_errors": int((local.label != local.lora_prediction).sum()),
    }


def main() -> None:
    frame = pd.read_csv(DATA)
    frame["id"] = frame["id"].astype(str)
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    base = np.load(BASE, allow_pickle=True)
    lora = np.load(LORA, allow_pickle=True)
    if not np.array_equal(frame.id.to_numpy(), base["ids"].astype(str)):
        raise ValueError("base id mismatch")
    if not np.array_equal(frame.id.to_numpy(), lora["ids"].astype(str)):
        raise ValueError("LoRA id mismatch")
    frame["base_prediction"] = base["predictions"].astype(np.int8)
    frame["base_rank"] = lora["base_rank"]
    frame["lora_rank"] = lora["lora_rank"]
    frame["fold"] = lora["folds"]
    final_score = np.zeros(len(frame), dtype=np.float32)
    final_prediction = np.zeros(len(frame), dtype=np.int8)
    configs = {
        "БАД": (0.40, 0.60, 0.25814030990600584),
        "Легковоспламеняющиеся": (0.75, 0.25, 0.9654546632766724),
    }
    for category, (base_weight, lora_weight, threshold) in configs.items():
        mask = frame.category.to_numpy() == category
        final_score[mask] = base_weight * frame.loc[mask, "base_rank"] + lora_weight * frame.loc[mask, "lora_rank"]
        final_prediction[mask] = (final_score[mask] >= threshold).astype(np.int8)
    frame["lora_fused_score"] = final_score
    frame["lora_prediction"] = final_prediction

    manifest = pd.read_csv(MANIFEST, sep="\t", compression="gzip")
    manifest["id"] = manifest["id"].astype(str)
    manifest["image_count"] = manifest.image_urls.map(lambda value: len(json.loads(value)))
    frame = frame.merge(manifest[["id", "image_count"]], on="id", how="left", validate="one_to_one")
    if frame.image_count.isna().any():
        raise ValueError("image count missing")
    frame["text"] = frame.name + "\n" + frame.description
    frame["normalized_text"] = frame.text.map(normalize)
    sizes = frame.groupby(["category", "normalized_text"]).id.transform("size")
    uniques = frame.groupby(["category", "normalized_text"]).label.transform("nunique")
    frame["duplicate_group"] = np.where(
        sizes == 1, "singleton", np.where(uniques == 1, "consistent_duplicate", "conflicting_duplicate")
    )
    frame["description_chars"] = frame.description.str.len()
    frame["description_bin"] = pd.qcut(
        frame.description_chars.rank(method="first"), 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"]
    ).astype(str)
    for name, pattern in PATTERNS.items():
        frame[name] = frame.text.map(lambda value, p=pattern: bool(p.search(value)))

    report: dict[str, object] = {
        "overall": {},
        "by_fold": {},
        "by_image_count": {},
        "by_duplicate_group": {},
        "by_description_length": {},
        "by_rule_signal": {},
        "transitions": {},
    }
    for category in configs:
        category_mask = frame.category.to_numpy() == category
        report["overall"][category] = metrics(frame, category_mask)
        report["by_fold"][category] = {
            str(value): metrics(frame, category_mask & (frame.fold.to_numpy() == value))
            for value in sorted(frame.fold.unique())
        }
        report["by_image_count"][category] = {
            str(value): metrics(frame, category_mask & (frame.image_count.to_numpy() == value))
            for value in sorted(frame.image_count.unique())
        }
        report["by_duplicate_group"][category] = {
            value: metrics(frame, category_mask & (frame.duplicate_group.to_numpy() == value))
            for value in ["singleton", "consistent_duplicate", "conflicting_duplicate"]
        }
        report["by_description_length"][category] = {
            value: metrics(frame, category_mask & (frame.description_bin.to_numpy() == value))
            for value in ["Q1_short", "Q2", "Q3", "Q4_long"]
        }
        signals = ["bad_explicit", "sports"] if category == "БАД" else ["flame_source", "equipment", "kit"]
        report["by_rule_signal"][category] = {
            name: {
                "present": metrics(frame, category_mask & frame[name].to_numpy()),
                "absent": metrics(frame, category_mask & ~frame[name].to_numpy()),
            }
            for name in signals
        }
        base_ok = frame.base_prediction.to_numpy() == frame.label.to_numpy()
        lora_ok = frame.lora_prediction.to_numpy() == frame.label.to_numpy()
        report["transitions"][category] = {
            "corrected_by_lora": int((category_mask & ~base_ok & lora_ok).sum()),
            "regressed_by_lora": int((category_mask & base_ok & ~lora_ok).sum()),
            "wrong_in_both": int((category_mask & ~base_ok & ~lora_ok).sum()),
            "right_in_both": int((category_mask & base_ok & lora_ok).sum()),
        }

    error_columns = [
        "id", "category", "label", "base_prediction", "lora_prediction", "fold",
        "image_count", "duplicate_group", "base_rank", "lora_rank", "lora_fused_score",
        "name", "description",
    ]
    frame.loc[frame.label != frame.lora_prediction, error_columns].sort_values(
        ["category", "label", "lora_fused_score"]
    ).to_csv(ERRORS, index=False)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"error_rows={(frame.label != frame.lora_prediction).sum()} path={ERRORS}", flush=True)


if __name__ == "__main__":
    main()

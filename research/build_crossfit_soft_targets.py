from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


WEIGHTS = {
    "БАД": np.asarray([0.50, 0.25, 0.25], dtype=np.float32),
    "Легковоспламеняющиеся": np.asarray([0.15, 0.10, 0.75], dtype=np.float32),
}
ALPHA = {
    ("БАД", 0): 0.25,
    ("БАД", 1): 0.25,
    ("Легковоспламеняющиеся", 0): 0.25,
    ("Легковоспламеняющиеся", 1): 0.10,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_aligned(paths: list[Path]) -> tuple[np.lib.npyio.NpzFile, ...]:
    arrays = tuple(np.load(path, allow_pickle=True) for path in paths)
    base = arrays[0]
    ids = base["ids"].astype(str)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    folds = base["fold_ids"].astype(np.int8)
    for other in arrays[1:]:
        other_folds = other["folds"].astype(np.int8)
        for key, expected, actual in (
            ("ids", ids, other["ids"].astype(str)),
            ("labels", labels, other["labels"].astype(np.int8)),
            ("categories", categories, other["categories"].astype(str)),
            ("folds", folds, other_folds),
        ):
            if not np.array_equal(expected, actual):
                raise ValueError(f"{key} mismatch")
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--qwen3vl", type=Path, required=True)
    parser.add_argument("--qwen35", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    base, qwen3vl, qwen35 = load_aligned([args.base, args.qwen3vl, args.qwen35])
    ids = base["ids"].astype(str)
    labels = base["labels"].astype(np.int8)
    categories = base["categories"].astype(str)
    folds = base["fold_ids"].astype(np.int8)
    score_matrix = np.column_stack(
        [
            qwen35["base_rank"].astype(np.float32),
            qwen3vl["lora_rank"].astype(np.float32),
            qwen35["lora_rank"].astype(np.float32),
        ]
    )
    teacher_probability = np.full(len(ids), np.nan, dtype=np.float32)
    calibration_rows = []
    for category in sorted(np.unique(categories)):
        raw_score = score_matrix @ WEIGHTS[category]
        for fold in sorted(np.unique(folds)):
            train = (categories == category) & (folds != fold)
            valid = (categories == category) & (folds == fold)
            model = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )
            model.fit(raw_score[train, None], labels[train])
            probability = model.predict_proba(raw_score[valid, None])[:, 1]
            teacher_probability[valid] = probability.astype(np.float32)
            calibration_rows.append(
                {
                    "category": category,
                    "validation_fold": int(fold),
                    "train_rows": int(train.sum()),
                    "validation_rows": int(valid.sum()),
                    "coefficient": float(model.coef_[0, 0]),
                    "intercept": float(model.intercept_[0]),
                }
            )
    if np.isnan(teacher_probability).any():
        raise ValueError("teacher probability contains missing values")

    alpha = np.asarray(
        [ALPHA[(category, int(label))] for category, label in zip(categories, labels)],
        dtype=np.float32,
    )
    soft_target = (1.0 - alpha) * labels.astype(np.float32) + alpha * teacher_probability
    output = pd.DataFrame(
        {
            "id": ids,
            "category": categories,
            "label": labels,
            "fold": folds,
            "teacher_probability": teacher_probability,
            "alpha": alpha,
            "soft_target": soft_target,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    report = {
        "protocol": "outer-fold cross-fitted calibration of the fixed Public-190 three-head score",
        "weights": {key: value.tolist() for key, value in WEIGHTS.items()},
        "alpha": {f"{category}:{label}": value for (category, label), value in ALPHA.items()},
        "rows": len(output),
        "calibration": calibration_rows,
        "metrics": {},
        "integrity": {
            "base_sha256": sha256(args.base),
            "qwen3vl_sha256": sha256(args.qwen3vl),
            "qwen35_sha256": sha256(args.qwen35),
        },
        "limitations": [
            "Teacher scores come only from models that did not train on the scored outer fold.",
            "The teacher is the current architecture and can preserve shared systematic errors.",
            "Fixed mild smoothing protects rare positive flammable examples but is not a substitute for an independent model family.",
        ],
    }
    for category in sorted(np.unique(categories)):
        mask = categories == category
        report["metrics"][category] = {
            "rows": int(mask.sum()),
            "positives": int(labels[mask].sum()),
            "teacher_brier": float(brier_score_loss(labels[mask], teacher_probability[mask])),
            "teacher_log_loss": float(log_loss(labels[mask], teacher_probability[mask])),
            "teacher_roc_auc": float(roc_auc_score(labels[mask], teacher_probability[mask])),
            "soft_target_mean": float(soft_target[mask].mean()),
            "soft_target_positive_min": float(soft_target[mask & (labels == 1)].min()),
            "soft_target_negative_max": float(soft_target[mask & (labels == 0)].max()),
        }
    report["integrity"]["soft_targets_sha256"] = sha256(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

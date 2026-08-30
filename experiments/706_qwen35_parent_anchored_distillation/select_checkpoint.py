from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = load(args.eval_root / "parent" / "metrics.json")
    parent_metrics = parent["metrics"]
    candidates = []
    for update in (10, 20, 40, 80):
        report = load(args.eval_root / f"update_{update:04d}" / "metrics.json")
        metrics = report["metrics"]
        row = {
            "update": update,
            "adapter_sha256": report["adapter_sha256"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "best_f1_diagnostic": metrics["best_f1_diagnostic"]["f1"],
            "delta_roc_auc": metrics["roc_auc"] - parent_metrics["roc_auc"],
            "delta_average_precision": metrics["average_precision"]
            - parent_metrics["average_precision"],
            "delta_best_f1_diagnostic": metrics["best_f1_diagnostic"]["f1"]
            - parent_metrics["best_f1_diagnostic"]["f1"],
        }
        row["eligible"] = (
            row["delta_average_precision"] > 0.0
            and row["delta_roc_auc"] >= -0.002
            and row["delta_best_f1_diagnostic"] >= 0.0
        )
        candidates.append(row)
    eligible = [row for row in candidates if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["delta_average_precision"],
            row["delta_roc_auc"],
            row["delta_best_f1_diagnostic"],
            -row["update"],
        ),
        default=None,
    )
    report = {
        "schema_version": "exp706_strict_holdout_selection_v1",
        "selection_data": "fold4_gold_never_seen_by_student",
        "teacher_target_training_folds": [0, 1, 2, 3],
        "category": "Легковоспламеняющиеся",
        "parent": {
            "adapter_sha256": parent["adapter_sha256"],
            "roc_auc": parent_metrics["roc_auc"],
            "average_precision": parent_metrics["average_precision"],
            "best_f1_diagnostic": parent_metrics["best_f1_diagnostic"]["f1"],
        },
        "candidates": candidates,
        "selected_update": selected["update"] if selected else None,
        "decision": "PROMOTE_TO_FULL_OOF_REFIT" if selected else "RUN_RESCUE_OR_KEEP_EXP140",
        "public_probe_required": False,
    }
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable selection report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

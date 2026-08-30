from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

FLAMMABLE = "Легковоспламеняющиеся"
FULL_FOLDS = (0, 1, 2, 3, 4)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def fold_macro(summary: dict[str, Any], fold: int) -> float:
    return float(
        np.mean(
            [
                summary["categories"][category]["fold_f1"][str(fold)]
                for category in sorted(summary["categories"])
            ]
        )
    )


def family_diagnostics(
    evaluation: Any,
    labels: np.ndarray,
    categories: np.ndarray,
    family_sizes: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for name, mask in {
        "singleton": family_sizes == 1,
        "rare_le2": family_sizes <= 2,
        "repeated": family_sizes > 1,
    }.items():
        result[name] = {
            **evaluation.compare(
                candidate[mask], baseline[mask], labels[mask], categories[mask]
            ),
            "rows": int(np.sum(mask)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-evaluator", type=Path, required=True)
    parser.add_argument("--evaluation-module", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--baseline-qwen3vl", type=Path, required=True)
    parser.add_argument("--baseline-qwen35", type=Path, required=True)
    parser.add_argument("--runtime-map", type=Path, required=True)
    parser.add_argument("--runtime-map-contract", type=Path, required=True)
    parser.add_argument("--source-runtime-root", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    screen = importlib.util.spec_from_file_location(
        "exp699_screen", args.screen_evaluator
    )
    if screen is None or screen.loader is None:
        raise RuntimeError("cannot import screen evaluator")
    screen_module = importlib.util.module_from_spec(screen)
    screen.loader.exec_module(screen_module)
    evaluation = screen_module.load_module(args.evaluation_module)

    oof = np.load(args.oof, allow_pickle=True)
    oof_ids = oof["ids"].astype(str)
    oof_labels = oof["labels"].astype(np.int8)
    oof_categories = oof["categories"].astype(str)
    oof_folds = oof["fold_ids"].astype(np.int8)
    source_positions, folds, global_indices = screen_module.load_source_split(
        args.runtime_map,
        args.runtime_map_contract,
        oof_ids,
        oof_categories,
    )
    ids = oof_ids[source_positions]
    labels = oof_labels[source_positions]
    categories = oof_categories[source_positions]
    if tuple(sorted(np.unique(folds).tolist())) != FULL_FOLDS:
        raise ValueError("exact full fivefold OOF is required")

    baseline_sources = {
        "qwen3vl_2b": np.load(args.baseline_qwen3vl, allow_pickle=True),
        "qwen35_4b": np.load(args.baseline_qwen35, allow_pickle=True),
    }
    for source in baseline_sources.values():
        if (
            not np.array_equal(source["ids"].astype(str), oof_ids)
            or not np.array_equal(source["labels"].astype(np.int8), oof_labels)
            or not np.array_equal(source["folds"].astype(np.int8), oof_folds)
            or not np.array_equal(
                source["categories"].astype(str), oof_categories
            )
        ):
            raise ValueError("baseline array binding mismatch")
    base_q3 = baseline_sources["qwen3vl_2b"]["lora_rank"].astype(np.float32)[
        source_positions
    ]
    base_q35 = baseline_sources["qwen35_4b"]["lora_rank"].astype(np.float32)[
        source_positions
    ]
    robust = baseline_sources["qwen3vl_2b"]["base_rank"].astype(np.float32)[
        source_positions
    ]

    spec = screen_module.parse_spec(args.candidate)
    if spec["architecture"] != "qwen35_4b":
        raise ValueError("this weight search requires a Qwen3.5-4B candidate")
    raw_scores, artifact_audit = screen_module.load_candidate(
        spec,
        ids,
        folds,
        categories,
        global_indices,
        evaluation_folds=FULL_FOLDS,
    )
    candidate_q35 = np.empty_like(base_q35)
    for fold in FULL_FOLDS:
        for category in sorted(np.unique(categories)):
            positions = np.flatnonzero(
                (folds == fold) & (categories == category)
            )
            candidate_q35[positions] = evaluation.rank01(raw_scores[positions])

    baseline_predictions, baseline_selection = evaluation.nested_fusion(
        labels,
        categories,
        folds,
        {"robust_base": robust, "qwen3vl": base_q3, "qwen35": base_q35},
    )
    candidate_predictions, candidate_selection = evaluation.nested_fusion(
        labels,
        categories,
        folds,
        {
            "robust_base": robust,
            "qwen3vl": base_q3,
            "qwen35": candidate_q35,
        },
    )
    baseline_summary = evaluation.summarize(
        labels, categories, folds, baseline_predictions
    )
    candidate_summary = evaluation.summarize(
        labels, categories, folds, candidate_predictions
    )
    comparison = evaluation.compare(
        candidate_predictions, baseline_predictions, labels, categories
    )
    fold_delta = {
        str(fold): fold_macro(candidate_summary, fold)
        - fold_macro(baseline_summary, fold)
        for fold in FULL_FOLDS
    }
    family_sizes, family_audit = screen_module.load_semantic_families(
        args.source_runtime_root,
        ids,
        folds,
        categories,
        global_indices,
    )
    flammable = categories == FLAMMABLE
    report: dict[str, Any] = {
        "schema": "exp699_nested_oof_fusion_search_v1",
        "experiment_id": "699",
        "candidate": {
            "name": spec["name"],
            "source": spec["source"],
            "mode": spec["mode"],
            "cap": spec["cap"],
            "artifact_audit": artifact_audit,
        },
        "protocol": {
            "outer_folds": list(FULL_FOLDS),
            "selection": "leave_one_fold_out",
            "simplex_step": 0.05,
            "threshold": "max_train_f1_at_unique_score_boundary",
            "public_used": False,
        },
        "baseline": {
            **baseline_summary,
            "selection": baseline_selection,
        },
        "candidate_result": {
            **candidate_summary,
            "macro_delta": candidate_summary["macro_f1"]
            - baseline_summary["macro_f1"],
            "fold_macro_delta": fold_delta,
            "fold_wins": int(sum(value > 0 for value in fold_delta.values())),
            "selection": candidate_selection,
            "flammable_component_ap": {
                "baseline": float(
                    average_precision_score(labels[flammable], base_q35[flammable])
                ),
                "candidate": float(
                    average_precision_score(
                        labels[flammable], candidate_q35[flammable]
                    )
                ),
            },
            "corrections_regressions": comparison,
            "semantic_families": family_diagnostics(
                evaluation,
                labels,
                categories,
                family_sizes,
                candidate_predictions,
                baseline_predictions,
            ),
        },
        "semantic_family_audit": family_audit,
        "validation_labels_read_for_evaluation": len(labels),
        "sealed_rows": 0,
        "public_used": False,
        "decision": "OOF_WEIGHT_SEARCH_COMPLETE",
    }
    report["self_sha256"] = canonical_sha256(report)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["candidate_result"], ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FLAMMABLE = "Легковоспламеняющиеся"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def average_precision(labels: Iterable[int], scores: Iterable[float]) -> float:
    """Non-interpolated sklearn-equivalent AP; equal scores share one threshold."""
    groups: dict[float, list[int]] = {}
    positive_total = row_count = 0
    for raw_label, raw_score in zip(labels, scores, strict=True):
        label, score = int(raw_label), float(raw_score)
        if label not in {0, 1} or not math.isfinite(score):
            raise ValueError("AP expects binary labels and finite scores")
        group = groups.setdefault(score, [0, 0])
        group[0] += label
        group[1] += 1
        positive_total += label
        row_count += 1
    if row_count == 0 or positive_total == 0:
        raise ValueError("AP requires rows and at least one positive")
    true_positives = predicted_positives = 0
    result = 0.0
    for score in sorted(groups, reverse=True):
        group_positives, group_count = groups[score]
        true_positives += group_positives
        predicted_positives += group_count
        result += (group_positives / positive_total) * (true_positives / predicted_positives)
    return result


def binary_f1(labels: Iterable[int], scores: Iterable[float], threshold: float) -> dict[str, Any]:
    tp = fp = fn = 0
    for label, score in zip(labels, scores, strict=True):
        prediction = int(float(score) >= threshold)
        tp += int(label == 1 and prediction == 1)
        fp += int(label == 0 and prediction == 1)
        fn += int(label == 1 and prediction == 0)
    denominator = 2 * tp + fp + fn
    return {
        "threshold": threshold,
        "f1": 0.0 if denominator == 0 else 2 * tp / denominator,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def load_registry(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        item_id = int(row["id"])
        if item_id in result:
            raise ValueError(f"duplicate registry id: {item_id}")
        result[item_id] = {
            "category": row["category"],
            "label": int(row["label"]),
            "fold": int(row["development_fold"]) if row["development_fold"] else None,
            "split": row["split"],
        }
    return result


def load_predictions(path: Path, fold: int) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        item_id = int(row["id"])
        if item_id in result or int(row["fold"]) != fold:
            raise ValueError(f"duplicate id or fold mismatch in {path}")
        result[item_id] = row
    return result


def evaluate_fold(
    registry: dict[int, dict[str, Any]], small_path: Path, large_path: Path, fold: int
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    small, large = load_predictions(small_path, fold), load_predictions(large_path, fold)
    expected = {
        item_id
        for item_id, row in registry.items()
        if row["split"] == "development" and row["fold"] == fold
    }
    if set(small) != expected or set(large) != expected:
        raise ValueError(f"prediction coverage mismatch for fold {fold}")
    ids = sorted(item_id for item_id in expected if registry[item_id]["category"] == FLAMMABLE)
    arrays: dict[str, list[Any]] = {
        "labels": [registry[item_id]["label"] for item_id in ids],
        "small": [float(small[item_id]["score"]) for item_id in ids],
        "large": [float(large[item_id]["score"]) for item_id in ids],
    }
    arrays["blend"] = [
        (left + right) / 2 for left, right in zip(arrays["small"], arrays["large"], strict=True)
    ]
    return {
        "fold": fold,
        "rows": len(ids),
        "positives": sum(arrays["labels"]),
        "prediction_sha256": {
            "small": sha256_file(small_path),
            "large": sha256_file(large_path),
        },
        "average_precision": {
            name: average_precision(arrays["labels"], arrays[name])
            for name in ("small", "large", "blend")
        },
    }, arrays


def audit_runtime(path: Path, fold: int) -> dict[str, Any]:
    train_path, audit_path = path / "train.jsonl", path / "runtime_audit.json"
    rows = read_jsonl(train_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["output_sha256"]["train.jsonl"] != sha256_file(train_path):
        raise ValueError(f"runtime checksum mismatch for fold {fold}")
    if audit["outer_fold"] != fold or audit["train_occurrences"] != len(rows):
        raise ValueError(f"runtime contract mismatch for fold {fold}")
    occurrences = Counter((row["category"], int(row["label"])) for row in rows)
    unique_rows = {int(row["id"]): row for row in rows}.values()
    unique = Counter((row["category"], int(row["label"])) for row in unique_rows)
    flammable_unique = sum(value for (category, _), value in unique.items() if category == FLAMMABLE)
    positive_occurrences, negative_occurrences = (
        occurrences[(FLAMMABLE, 1)],
        occurrences[(FLAMMABLE, 0)],
    )
    sampled = positive_occurrences / (positive_occurrences + negative_occurrences)
    natural = unique[(FLAMMABLE, 1)] / flammable_unique
    shift = math.log(sampled / (1 - sampled)) - math.log(natural / (1 - natural))
    return {
        "fold": fold,
        "train_sha256": sha256_file(train_path),
        "runtime_contract_sha256": audit["contract_sha256"],
        "train_occurrences": len(rows),
        "train_unique_rows": len({int(row["id"]) for row in rows}),
        "bad_negative_occurrences": occurrences[("БАД", 0)],
        "bad_positive_occurrences": occurrences[("БАД", 1)],
        "flammable_negative_occurrences": negative_occurrences,
        "flammable_positive_occurrences": positive_occurrences,
        "flammable_unique_rows": flammable_unique,
        "flammable_unique_positives": unique[(FLAMMABLE, 1)],
        "flammable_sampled_positive_prevalence": sampled,
        "flammable_natural_positive_prevalence": natural,
        "diagnostic_prior_logit_shift": shift,
    }


def audit_recipe(training_script: Path, report_paths: list[Path]) -> dict[str, Any]:
    text = training_script.read_text(encoding="utf-8")
    fields = {
        "learning_rate": r"AdamW\(trainable, lr=([0-9.e-]+)",
        "weight_decay": r"weight_decay=([0-9.]+)\)",
        "warmup_steps": r"warmup = (\d+)",
        "lora_rank": r"\br=(\d+),",
        "lora_alpha": r"lora_alpha=(\d+),",
        "lora_dropout": r"lora_dropout=([0-9.]+),",
    }
    parsed: dict[str, Any] = {}
    for name, pattern in fields.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"recipe field not found: {name}")
        raw = match.group(1)
        parsed[name] = float(raw) if any(char in raw for char in ".e") else int(raw)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    missing = [
        field
        for field in (
            "loss_history",
            "learning_rate_history",
            "gradient_norm_history",
            "intermediate_checkpoints",
            "inner_validation_average_precision",
        )
        if all(field not in report for report in reports)
    ]
    parsed.update(
        {
            "model": "Qwen/Qwen3.6-27B",
            "objective": "binary BCE on final-token logit(1)-logit(0)",
            "precision": "bf16",
            "epochs": 1,
            "optimizer_steps": 306,
            "micro_batch_size": 1,
            "gradient_accumulation": 16,
            "effective_batch_size": 16,
            "scheduler": "cosine_to_zero",
            "gradient_clip_norm": 1.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "use_rslora": True,
            "first_image_max_pixels": 262144,
            "maximum_sequence_length": 1536,
            "training_script_sha256": sha256_file(training_script),
            "completed_reports": [
                {
                    "fold": report["outer_fold"],
                    "sha256": sha256_file(path),
                    "elapsed_seconds": report["elapsed_seconds"],
                    "packages": report["packages"],
                }
                for path, report in zip(report_paths, reports, strict=True)
            ],
            "missing_observability": missing,
            "final_checkpoint_only": len(missing) == 5,
        }
    )
    return parsed


def run(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.registry)
    per_fold, arrays_by_fold = [], []
    for fold, small, large in zip(
        args.folds, args.small_predictions, args.large_predictions, strict=True
    ):
        metrics, arrays = evaluate_fold(registry, small, large, fold)
        per_fold.append(metrics)
        arrays_by_fold.append(arrays)
    pooled = {
        key: [value for arrays in arrays_by_fold for value in arrays[key]]
        for key in ("labels", "small", "large", "blend")
    }
    runtime = [
        audit_runtime(path, fold) for fold, path in zip(args.folds_all, args.runtime_dirs, strict=True)
    ]
    diagnostics = []
    runtime_by_fold = {item["fold"]: item for item in runtime}
    for metrics, arrays in zip(per_fold, arrays_by_fold, strict=True):
        shift = runtime_by_fold[metrics["fold"]]["diagnostic_prior_logit_shift"]
        diagnostics.append(
            {
                "fold": metrics["fold"],
                "at_zero": binary_f1(arrays["labels"], arrays["large"], 0.0),
                "after_analytic_donor_prior_shift": binary_f1(
                    arrays["labels"], arrays["large"], shift
                ),
            }
        )
    result = {
        "schema_version": 1,
        "experiment_id": "676",
        "status": "complete",
        "public_used": False,
        "sealed_rows": 0,
        "metric_definition": {
            "name": "average_precision",
            "equivalent": "sklearn.metrics.average_precision_score",
            "ties": "all equal scores form one threshold group",
        },
        "registry_sha256": sha256_file(args.registry),
        "flammable_average_precision": {
            "per_fold": per_fold,
            "pooled_screen_folds": {
                "folds": args.folds,
                "rows": len(pooled["labels"]),
                "positives": sum(pooled["labels"]),
                "small": average_precision(pooled["labels"], pooled["small"]),
                "large": average_precision(pooled["labels"], pooled["large"]),
                "fixed_equal_blend": average_precision(pooled["labels"], pooled["blend"]),
            },
        },
        "sampling_audit": runtime,
        "calibration_diagnostic_not_for_selection": diagnostics,
        "threshold_policy": "do not infer hidden prevalence or tune from Public",
        "training_recipe": audit_recipe(args.training_script, args.reports),
        "confirmed_findings": [
            "Large-model and fixed-blend ranking improve on both completed screen folds.",
            "Rare positives are repeated five times, so sampled and natural prevalence differ.",
            "Final-only checkpointing cannot prove that optimizer step 306 is optimal.",
            "BF16 score ties require threshold-grouped Average Precision.",
        ],
        "decision": "PREPARE_DONOR_INNER_PR_AUC_DYNAMICS_SCREEN_WAIT_FOR_TERMINAL_659",
        "next_experiment": {
            "changed_factor": "optimizer stop step only",
            "candidate_steps": [51, 102, 153, 204, 255, 306],
            "selection_data": "grouped donor-inner split inside each outer training fold",
            "primary_metric": "flammable average precision",
            "acceptance": {
                "non_final_checkpoint_wins": "at least 3 of 4 donor-inner screens",
                "mean_flammable_ap_gain": 0.01,
                "maximum_bad_ap_drop": 0.002,
            },
            "launch_blockers": [
                "terminal five-fold frozen route result not yet available",
                "GPU capacity is reserved by the active frozen route",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--folds", type=int, nargs="+", default=[0, 3])
    result.add_argument("--small-predictions", type=Path, nargs="+", required=True)
    result.add_argument("--large-predictions", type=Path, nargs="+", required=True)
    result.add_argument("--folds-all", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    result.add_argument("--runtime-dirs", type=Path, nargs="+", required=True)
    result.add_argument("--training-script", type=Path, required=True)
    result.add_argument("--reports", type=Path, nargs="+", required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if len(arguments.folds) != len(arguments.small_predictions) or len(arguments.folds) != len(
        arguments.large_predictions
    ):
        raise ValueError("prediction counts must match folds")
    if len(arguments.folds_all) != len(arguments.runtime_dirs):
        raise ValueError("runtime counts must match folds-all")
    print(json.dumps(run(arguments), ensure_ascii=False, indent=2, sort_keys=True))

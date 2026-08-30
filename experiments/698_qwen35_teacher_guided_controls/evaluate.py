from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

MODES = ("gold_control", "hardneg_candidate", "rank_candidate")
EXPECTED_DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"
INCUMBENT_MACRO = 0.9118425205786493
EXPECTED_IMAGE_PREPROCESSING = "solution140_first_image_thumbnail_448_lanczos_v1"
EXPECTED_ORDER_SHA256 = "0c915039e8535786282b007e0824abf4e8801c973532ca06f8d2d7bdf18b8da1"
EXPECTED_TRAINING_CONTRACT = {
    "seed": 42,
    "sampler": "exp697_label_only_v1",
    "epochs": 1,
    "max_length": 1536,
    "micro_batch": 2,
    "gradient_accumulation": 8,
    "effective_batch": 16,
    "occurrence_order": {
        "rule": "python_random_seeded_shuffle_of_occurrence_indices",
        "sha256": EXPECTED_ORDER_SHA256,
    },
    "optimizer": {
        "name": "AdamW",
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "clip_grad_norm": 1.0,
    },
    "scheduler": {"name": "cosine", "warmup_fraction": 0.05},
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none",
        "use_rslora": True,
    },
    "model_dtype": "bfloat16",
    "attention_implementation": "eager",
    "image_preprocessing": EXPECTED_IMAGE_PREPROCESSING,
}
EXPECTED_OBJECTIVES = {
    "gold_control": {"name": "hard_gold_bce", "teacher_target_consumed": False},
    "hardneg_candidate": {
        "name": "flammable_teacher_disagreement_weighted_hard_gold_bce",
        "teacher_target_consumed": True,
        "hard_weight_max": 2.0,
        "teacher_verdict_replaces_gold": False,
    },
    "rank_candidate": {
        "name": "flammable_same_label_within_batch_teacher_rank_plus_hard_gold_bce",
        "teacher_target_consumed": True,
        "rank_coefficient": 0.5,
        "opposite_label_pairs_allowed": False,
        "teacher_verdict_replaces_gold": False,
    },
}
MATCHED_CONTROL_VERIFICATION = {
    "schema_version": "exp698_matched_control_verification_v1",
    "fold_output_schema": "exp698_fold_output_v2",
    "common_training_factors_equal_per_fold": True,
    "candidate_teacher_bindings_equal_per_fold": True,
    "gold_teacher_binding_absent": True,
}


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) != 2:
        raise ValueError("threshold donor lacks both labels")
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


def fold_category_percentile(data: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(data) or not np.isfinite(scores).all():
        raise ValueError("score vector is misaligned or non-finite")
    calibrated = np.empty(len(scores), dtype=np.float32)
    folds = data["fold"].to_numpy(np.int8)
    categories = data["category"].astype(str).to_numpy()
    for fold in sorted(np.unique(folds)):
        for category in sorted(np.unique(categories)):
            positions = np.flatnonzero((folds == fold) & (categories == category))
            if not len(positions):
                raise ValueError(f"empty calibration stratum fold={fold} category={category}")
            calibrated[positions] = (
                pd.Series(scores[positions]).rank(method="average", pct=True).to_numpy(np.float32)
            )
    if not np.isfinite(calibrated).all() or not np.all((calibrated > 0) & (calibrated <= 1)):
        raise ValueError("invalid fold/category percentile calibration")
    return calibrated


def load_mode(root: Path, mode: str, data: pd.DataFrame) -> tuple[np.ndarray, dict]:
    score_by_id: dict[str, float] = {}
    bindings: dict[str, dict] = {}
    for fold in range(5):
        directory = root / f"{mode}-f{fold}"
        prediction_path = directory / "predictions.jsonl"
        contract_path = directory / "output_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        payload = dict(contract)
        digest = payload.pop("contract_sha256", None)
        canonical = hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        expected = {
            "schema_version": "exp698_fold_output_v2",
            "experiment_id": "698",
            "source_experiment_id": "697",
            "fold": fold,
            "mode": mode,
            "technical_smoke": False,
            "model": "/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3.5-4B",
            "submission_eligible_base_model": True,
            "teacher_model_required_at_inference": False,
            "image_preprocessing": EXPECTED_IMAGE_PREPROCESSING,
            "train_occurrences": 5390,
            "full_train_occurrences": 5390,
            "micro_batch": 2,
            "gradient_accumulation": 8,
            "effective_batch": 16,
            "optimizer_steps": 337,
            "training_contract": EXPECTED_TRAINING_CONTRACT,
            "objective_contract": EXPECTED_OBJECTIVES[mode],
        }
        mismatch = {
            key: {"expected": value, "actual": contract.get(key)}
            for key, value in expected.items()
            if contract.get(key) != value
        }
        if mismatch or digest != canonical:
            raise ValueError(f"{mode} fold {fold} contract mismatch: {mismatch}")
        if contract.get("predictions_sha256") != sha256(prediction_path):
            raise ValueError(f"{mode} fold {fold} prediction checksum mismatch")
        adapter = directory / "adapter"
        if (
            contract.get("adapter_model_sha256")
            != sha256(adapter / "adapter_model.safetensors")
            or contract.get("adapter_config_sha256")
            != sha256(adapter / "adapter_config.json")
        ):
            raise ValueError(f"{mode} fold {fold} adapter checksum mismatch")
        rows = read_jsonl(prediction_path)
        if len(rows) != int(contract["validation_rows"]):
            raise ValueError(f"{mode} fold {fold} prediction row count mismatch")
        expected_ids = set(data.loc[data["fold"] == fold, "id"])
        actual_ids = {str(row["id"]) for row in rows}
        if (
            len(rows) != len(expected_ids)
            or actual_ids != expected_ids
            or set(score_by_id) & actual_ids
        ):
            raise ValueError(f"{mode} fold {fold} prediction coverage mismatch")
        for row in rows:
            if set(row) != {"id", "fold", "mode", "score"}:
                raise ValueError("prediction schema mismatch")
            score = float(row["score"])
            if int(row["fold"]) != fold or row["mode"] != mode or not math.isfinite(score):
                raise ValueError("prediction value mismatch")
            score_by_id[str(row["id"])] = score
        teacher_binding = contract.get("teacher_binding")
        if mode == "gold_control":
            if teacher_binding is not None:
                raise ValueError("gold control unexpectedly consumed teacher targets")
        else:
            expected_teacher_keys = {
                "teacher_target_contract_sha256",
                "teacher_targets_sha256",
                "adapter_output_contract_sha256",
                "adapter_model_sha256",
                "outer_safe_validation_excluded",
                "targets_are_in_sample_within_outer_train",
            }
            if (
                not isinstance(teacher_binding, dict)
                or set(teacher_binding) != expected_teacher_keys
                or not all(
                    is_sha256(teacher_binding[key])
                    for key in expected_teacher_keys
                    if key.endswith("sha256")
                )
                or teacher_binding.get("outer_safe_validation_excluded") is not True
                or teacher_binding.get("targets_are_in_sample_within_outer_train") is not True
            ):
                raise ValueError(f"{mode} fold {fold} teacher binding mismatch")
        bindings[str(fold)] = {
            "output_contract_sha256": sha256(contract_path),
            "predictions_sha256": sha256(prediction_path),
            "adapter_model_sha256": contract["adapter_model_sha256"],
            "adapter_config_sha256": contract["adapter_config_sha256"],
            "runtime_audit_sha256": contract["runtime_audit_sha256"],
            "training_contract": contract["training_contract"],
            "teacher_binding": teacher_binding,
            "runtime_minutes": contract["runtime_minutes"],
            "peak_cuda_memory_bytes": contract["peak_cuda_memory_bytes"],
        }
    if set(score_by_id) != set(data["id"]):
        raise ValueError(f"{mode} five-fold coverage mismatch")
    return np.asarray([score_by_id[row_id] for row_id in data["id"]]), bindings


def nested_report(data: pd.DataFrame, scores: np.ndarray) -> tuple[dict, np.ndarray]:
    predictions = np.zeros(len(data), dtype=np.int8)
    categories: dict[str, dict] = {}
    for category in sorted(data["category"].astype(str).unique()):
        positions = np.flatnonzero(data["category"].astype(str).to_numpy() == category)
        labels = data.iloc[positions]["label"].to_numpy(np.int8)
        folds = data.iloc[positions]["fold"].to_numpy(np.int8)
        local_scores = scores[positions]
        local_predictions = np.zeros(len(positions), dtype=np.int8)
        fold_rows: dict[str, dict] = {}
        donor_thresholds: list[float] = []
        for fold in range(5):
            validation = folds == fold
            threshold = best_threshold(labels[~validation], local_scores[~validation])
            donor_thresholds.append(threshold)
            local_predictions[validation] = local_scores[validation] >= threshold
            fold_rows[str(fold)] = {
                "rows": int(validation.sum()),
                "threshold": threshold,
                "f1": float(f1_score(labels[validation], local_predictions[validation])),
            }
        predictions[positions] = local_predictions
        categories[category] = {
            "f1": float(f1_score(labels, local_predictions)),
            "average_precision": float(average_precision_score(labels, local_scores)),
            "folds": fold_rows,
            "deployment": {
                "score": "category_batch_percentile_rank",
                "threshold": float(np.median(donor_thresholds)),
                "threshold_rule": "median_of_five_outer_train_thresholds",
                "outer_train_thresholds": donor_thresholds,
            },
        }
    macro = float(np.mean([value["f1"] for value in categories.values()]))
    return {"nested_macro_f1": macro, "categories": categories}, predictions


def verify_matched_controls(report: dict) -> None:
    for fold in range(5):
        key = str(fold)
        common_runtime = {
            report["modes"][mode]["fold_artifacts"][key]["runtime_audit_sha256"]
            for mode in MODES
        }
        common_training = {
            json.dumps(
                report["modes"][mode]["fold_artifacts"][key]["training_contract"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for mode in MODES
        }
        if len(common_runtime) != 1 or len(common_training) != 1:
            raise ValueError(f"fold {fold} matched-control common-factor mismatch")
        if (
            report["modes"]["hardneg_candidate"]["fold_artifacts"][key]["teacher_binding"]
            != report["modes"]["rank_candidate"]["fold_artifacts"][key]["teacher_binding"]
        ):
            raise ValueError(f"fold {fold} candidate teacher-target mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("canonical data checksum mismatch")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite evaluation")
    oof_path = args.output.with_suffix(".npz")
    if oof_path.exists():
        raise FileExistsError("refusing to overwrite OOF artifact")
    data = pd.read_csv(args.data, dtype={"id": str})
    folds = pd.read_csv(args.folds, dtype={"id": str})
    if data["id"].duplicated().any() or folds["id"].duplicated().any():
        raise ValueError("duplicate source IDs")
    fold_map = dict(zip(folds["id"], folds["fold"].astype(int), strict=True))
    if set(data["id"]) != set(fold_map):
        raise ValueError("data/folds ID mismatch")
    data["fold"] = [fold_map[row_id] for row_id in data["id"]]

    report = {
        "schema_version": "exp698_evaluation_v2",
        "experiment_id": "698",
        "evaluation": "nested_grouped_v1",
        "image_preprocessing": EXPECTED_IMAGE_PREPROCESSING,
        "deployment_score": "category_batch_percentile_rank",
        "deployment_threshold_rule": "median_of_five_outer_train_thresholds",
        "rows": len(data),
        "data_sha256": sha256(args.data),
        "folds_sha256": sha256(args.folds),
        "incumbent_140_nested_macro_f1": INCUMBENT_MACRO,
        "matched_control_verification": MATCHED_CONTROL_VERIFICATION,
        "modes": {},
        "comparisons": {},
    }
    mode_predictions: dict[str, np.ndarray] = {}
    mode_scores: dict[str, np.ndarray] = {}
    mode_raw_scores: dict[str, np.ndarray] = {}
    for mode in MODES:
        raw_scores, bindings = load_mode(args.outputs, mode, data)
        scores = fold_category_percentile(data, raw_scores)
        metrics, predictions = nested_report(data, scores)
        metrics["fold_artifacts"] = bindings
        metrics["score_calibration"] = "category_and_outer_fold_percentile_average_ties"
        metrics["standalone_delta_vs_140"] = metrics["nested_macro_f1"] - INCUMBENT_MACRO
        report["modes"][mode] = metrics
        mode_predictions[mode] = predictions
        mode_scores[mode] = scores.astype(np.float32)
        mode_raw_scores[mode] = raw_scores.astype(np.float32)

    verify_matched_controls(report)

    control = report["modes"]["gold_control"]
    for mode in ("hardneg_candidate", "rank_candidate"):
        candidate = report["modes"][mode]
        fold_wins = 0
        category_deltas = {}
        for category in control["categories"]:
            category_deltas[category] = (
                candidate["categories"][category]["f1"]
                - control["categories"][category]["f1"]
            )
        for fold in range(5):
            control_fold = np.mean(
                [value["folds"][str(fold)]["f1"] for value in control["categories"].values()]
            )
            candidate_fold = np.mean(
                [
                    value["folds"][str(fold)]["f1"]
                    for value in candidate["categories"].values()
                ]
            )
            fold_wins += int(candidate_fold > control_fold)
        delta = candidate["nested_macro_f1"] - control["nested_macro_f1"]
        report["comparisons"][mode] = {
            "nested_macro_delta_vs_gold_control": delta,
            "category_f1_deltas": category_deltas,
            "fold_wins": fold_wins,
            "changed_predictions": int(
                np.sum(mode_predictions[mode] != mode_predictions["gold_control"])
            ),
            "science_gate": (
                delta >= 0.001
                and fold_wins >= 4
                and min(category_deltas.values()) >= -0.005
            ),
        }
    passing = [
        mode for mode, value in report["comparisons"].items() if value["science_gate"]
    ]
    report["decision"] = (
        "DISTILLATION_SIGNAL_CONFIRMED" if passing else "NO_DISTILLATION_SIGNAL"
    )
    report["passing_candidates"] = passing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        oof_path,
        ids=data["id"].astype(str).to_numpy(),
        labels=data["label"].to_numpy(np.int8),
        categories=data["category"].astype(str).to_numpy(),
        folds=data["fold"].to_numpy(np.int8),
        **{f"{mode}_scores": values for mode, values in mode_scores.items()},
        **{f"{mode}_raw_scores": values for mode, values in mode_raw_scores.items()},
        **{
            f"{mode}_predictions": values
            for mode, values in mode_predictions.items()
        },
    )
    report["oof_artifact"] = oof_path.name
    report["oof_artifact_sha256"] = sha256(oof_path)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

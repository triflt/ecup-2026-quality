from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "677"
ALLOWED_NONFINAL_FRACTIONS = {0.25, 0.5, 0.75}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_self_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    digest = payload.pop("contract_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError(f"self-hash mismatch: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def freeze(
    *,
    inner_selection_path: Path,
    step_mapping_path: Path,
    source_runtime_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite an outer0 confirmation contract")
    selection = load_self_hashed(inner_selection_path)
    expected_selection = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "outer_screen_fold": 0,
        "inner_folds": [1, 2, 3, 4],
        "sealed_rows": 0,
        "public_used": False,
        "threshold_tuned": False,
        "decision": "GO_CONFIRM_SELECTED_STOP_ON_OUTER_FOLD_0_ONLY",
    }
    if any(selection.get(key) != value for key, value in expected_selection.items()):
        raise ValueError("inner selection did not authorize outer0 confirmation")
    fraction = float(selection.get("selected_training_fraction"))
    if fraction not in ALLOWED_NONFINAL_FRACTIONS:
        raise ValueError("selected fraction is not an allowed non-final checkpoint")

    mapping = load_self_hashed(step_mapping_path)
    expected_mapping = {
        "experiment_id": EXPERIMENT_ID,
        "outer_fold": 0,
        "source_experiment_id": "641",
        "micro_batch_size": 2,
        "gradient_accumulation": 8,
        "optimizer_updates": 306,
        "rounding_policy": "python_round_ties_to_even",
        "comparison_same_training_trajectory_required": True,
        "sealed_rows": 0,
        "public_used": False,
    }
    if any(mapping.get(key) != value for key, value in expected_mapping.items()):
        raise ValueError("outer0 step mapping contract mismatch")
    step_by_fraction = {
        float(row["requested_training_fraction"]): int(row["optimizer_step"])
        for row in mapping.get("fraction_to_optimizer_step", [])
    }
    if step_by_fraction != {0.25: 76, 0.5: 153, 0.75: 230, 1.0: 306}:
        raise ValueError("outer0 fraction-to-step mapping drifted")

    audit_path = source_runtime_dir / "runtime_audit.json"
    train_path = source_runtime_dir / "train.jsonl"
    validation_path = source_runtime_dir / "validation.jsonl"
    runtime = load_self_hashed(audit_path)
    expected_runtime = {
        "experiment_id": "641",
        "outer_fold": 0,
        "train_occurrences": 4892,
        "validation_rows": 2224,
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "decision": "GO",
    }
    if any(runtime.get(key) != value for key, value in expected_runtime.items()):
        raise ValueError("source outer0 runtime contract mismatch")
    if runtime.get("output_sha256") != {
        "train.jsonl": sha256_file(train_path),
        "validation.jsonl": sha256_file(validation_path),
    }:
        raise ValueError("source outer0 runtime payload mismatch")
    train_rows, validation_rows = read_jsonl(train_path), read_jsonl(validation_path)
    if len(train_rows) != 4892 or len(validation_rows) != 2224:
        raise ValueError("source outer0 runtime row counts mismatch")
    if any("label" in row or "evidence_target" in row for row in validation_rows):
        raise ValueError("outer0 validation supervision is forbidden")
    if runtime["contract_sha256"] != mapping.get("source_runtime_contract_sha256"):
        raise ValueError("step mapping is not bound to the source outer0 runtime")

    selected_step = step_by_fraction[fraction]
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "outer_fold": 0,
        "changed_factor": "optimizer_stop_step_only",
        "selected_training_fraction": fraction,
        "selected_optimizer_step": selected_step,
        "final_optimizer_step": 306,
        "required_checkpoint_steps": [selected_step, 306],
        "comparison_same_training_trajectory_required": True,
        "selection_result_sha256": sha256_file(inner_selection_path),
        "step_mapping_sha256": sha256_file(step_mapping_path),
        "source_runtime_contract_sha256": runtime["contract_sha256"],
        "source_train_sha256": runtime["output_sha256"]["train.jsonl"],
        "source_validation_sha256": runtime["output_sha256"]["validation.jsonl"],
        "outer0_labels_read": False,
        "threshold_tuned": False,
        "sealed_rows": 0,
        "public_used": False,
        "decision": "READY_TO_BUILD_OUTER0_SAME_TRAJECTORY_CONFIRMATION",
    }
    result["contract_sha256"] = canonical_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--inner-selection", type=Path, required=True)
    result.add_argument("--step-mapping", type=Path, required=True)
    result.add_argument("--source-runtime-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            freeze(
                inner_selection_path=args.inner_selection,
                step_mapping_path=args.step_mapping,
                source_runtime_dir=args.source_runtime_dir,
                output_path=args.output,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

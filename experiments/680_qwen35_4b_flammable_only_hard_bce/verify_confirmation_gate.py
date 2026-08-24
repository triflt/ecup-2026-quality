from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(
    gate_path: Path,
    runtime_dir: Path,
    screen_report_path: Path,
    fold: int,
) -> dict[str, Any]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    expected = (
        gate.get("experiment_id") == "680"
        and gate.get("control_experiment_id") == "641"
        and gate.get("decision") == "OPEN_CONFIRMATION"
        and gate.get("training_lane_open") is True
        and gate.get("allowed_folds") == [1, 2, 4]
        and gate.get("screen_folds") == [0, 3]
        and gate.get("confirmation_folds") == [1, 2, 4]
        and gate.get("model_id") == "Qwen/Qwen3.5-4B"
        and gate.get("model_revision") == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        and gate.get("objective") == "class_only_binary_bce"
        and gate.get("changed_factor") == "remove_bad_training_occurrences"
        and gate.get("learning_rate") == 0.0002
        and gate.get("expected_optimizer_steps") == 143
        and gate.get("runtime_backend") == "legacy_eager"
        and gate.get("micro_batch_size") == 2
        and gate.get("gradient_accumulation") == 8
        and gate.get("effective_batch_size") == 16
        and gate.get("threshold") == 0.0
        and gate.get("threshold_tuned") is False
        and gate.get("public_used") is False
        and gate.get("sealed_rows") == 0
    )
    if not expected or fold not in gate.get("allowed_folds", []):
        raise ValueError("experiment-680 confirmation gate is closed or inconsistent")

    screen_sha256 = sha256_file(screen_report_path)
    if screen_sha256 != gate.get("screen_report_sha256"):
        raise ValueError("screen checksum differs from frozen confirmation gate")
    screen = json.loads(screen_report_path.read_text(encoding="utf-8"))
    if not (
        screen.get("experiment_id") == "680"
        and screen.get("stage") == "screen"
        and screen.get("changed_factor") == "remove_bad_training_occurrences"
        and screen.get("passed") is True
        and screen.get("decision") == "OPEN_CONFIRMATION"
        and screen.get("public_used") is False
        and screen.get("sealed_rows") == 0
        and set(screen.get("fold_flammable_average_precision", {})) == {"0", "3"}
        and all(screen.get("gates", {}).values())
    ):
        raise ValueError("screen report does not authorize confirmation")

    audit = json.loads((runtime_dir / "runtime_audit.json").read_text(encoding="utf-8"))
    if audit.get("contract_sha256") != gate["runtime_contract_sha256"][str(fold)]:
        raise ValueError("runtime contract differs from frozen confirmation runtime")
    if (
        audit.get("derived_experiment_id") != "680"
        or audit.get("changed_factor") != "remove_bad_training_occurrences"
        or audit.get("bad_train_occurrences") != 0
        or audit.get("train_occurrences") != 2280
        or audit.get("validation_labels_written") != 0
        or audit.get("sealed_rows_written") != 0
    ):
        raise ValueError("runtime semantic contract mismatch")
    for name in ("train.jsonl", "validation.jsonl"):
        actual = sha256_file(runtime_dir / name)
        if actual != gate["runtime_payload_sha256"][str(fold)][name]:
            raise ValueError(f"runtime payload mismatch: {name}")
    return gate


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--gate", type=Path, required=True)
    result.add_argument("--runtime-dir", type=Path, required=True)
    result.add_argument("--screen-report", type=Path, required=True)
    result.add_argument("--fold", type=int, choices=(1, 2, 4), required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            verify(args.gate, args.runtime_dir, args.screen_report, args.fold),
            indent=2,
            sort_keys=True,
        )
    )

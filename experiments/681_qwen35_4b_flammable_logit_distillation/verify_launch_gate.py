from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


FLAMMABLE = "Легковоспламеняющиеся"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(gate_path: Path, runtime_dir: Path, fold: int) -> dict:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    allowed = gate.get("allowed_folds")
    required = (
        gate.get("experiment_id") == 681
        and gate.get("ranking_control_experiment_id") == 641
        and gate.get("routed_production_baseline") == "frozen_semantic_v3_replay"
        and gate.get("routed_production_baseline_contract_sha256")
        == "1771dceda26996529b9f70f46a97653c2d29afa099983260c3b40a62fca518b0"
        and gate.get("causal_control_experiment_id") == 680
        and gate.get("teacher_scoring_experiment_id") == 662
        and set(gate.get("causal_control_archive_sha256", {})) == {"0", "3"}
        and set(gate.get("causal_control_prediction_sha256", {})) == {"0", "3"}
        and all(
            isinstance(value, str) and len(value) == 64
            for value in gate.get("causal_control_archive_sha256", {}).values()
        )
        and all(
            isinstance(value, str) and len(value) == 64
            for value in gate.get("causal_control_prediction_sha256", {}).values()
        )
        and gate.get("full_causal_control_status") == "BLOCKED_MISSING_FOLDS_1_2_4"
        and gate.get("decision") == "OPEN_SCREEN"
        and gate.get("training_lane_open") is True
        and isinstance(allowed, list)
        and set(allowed).issubset({0, 3})
        and allowed == sorted(allowed)
        and gate.get("changed_factor")
        == "hard_bce_to_fixed_hard_plus_teacher_soft_bce_on_680_specialist"
        and gate.get("temperature") == 2.0
        and gate.get("soft_loss_weight") == 0.5
        and gate.get("learning_rate") == 0.0002
        and gate.get("expected_optimizer_steps") == 143
        and gate.get("threshold") == 0.0
        and gate.get("threshold_tuned") is False
        and gate.get("ordinary_oof_merge_used") is False
        and gate.get("public_used") is False
        and gate.get("sealed_rows") == 0
    )
    if not required or fold not in allowed:
        raise ValueError("experiment-681 gate is closed or inconsistent")
    audit = json.loads((runtime_dir / "runtime_audit.json").read_text(encoding="utf-8"))
    if audit.get("contract_sha256") != gate["runtime_contract_sha256"][str(fold)]:
        raise ValueError("student runtime contract mismatch")
    if (
        audit.get("distillation_experiment_id") != "681"
        or audit.get("teacher_experiment_id") != "662"
        or int(audit.get("teacher_outer_fold", -1)) != fold
        or audit.get("teacher_target_scope")
        != "outer_train_in_sample_outer_validation_unread"
        or audit.get("changed_factor")
        != "hard_bce_to_fixed_hard_plus_teacher_soft_bce"
        or audit.get("temperature") != 2.0
        or audit.get("soft_loss_weight") != 0.5
        or audit.get("ordinary_oof_merge_used") is not False
        or int(audit.get("outer_validation_teacher_overlap", -1)) != 0
        or int(audit.get("bad_train_occurrences", -1)) != 0
        or int(audit.get("train_occurrences", -1)) != 2280
    ):
        raise ValueError("student runtime violates the frozen distillation contract")
    teacher = audit.get("teacher_manifest", {})
    if (
        teacher.get("teacher_scores_sha256")
        != gate["teacher_scores_sha256"][str(fold)]
        or teacher.get("archive_sha256")
        != gate["teacher_archive_sha256"][str(fold)]
    ):
        raise ValueError("teacher target artifact mismatch")
    for name in ("train.jsonl", "validation.jsonl"):
        if sha256_file(runtime_dir / name) != gate["runtime_payload_sha256"][str(fold)][name]:
            raise ValueError(f"student runtime payload mismatch: {name}")
    rows = [
        json.loads(line)
        for line in (runtime_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != 2280 or any(
        row.get("category") != FLAMMABLE
        or not math.isfinite(float(row.get("teacher_score", math.nan)))
        for row in rows
    ):
        raise ValueError("student train target schema mismatch")
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 3), required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.gate, args.runtime_dir, args.fold), indent=2, sort_keys=True))

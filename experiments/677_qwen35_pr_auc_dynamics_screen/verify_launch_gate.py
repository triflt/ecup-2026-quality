from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PARENT_RESULT_SHA256 = "e2999faf11f4bbc98c9adaea5a7e8514c50d03b6571b8c62f4cced0260b4c91a"
SMOKE_ACCEPTANCE_SHA256 = "185c72bdcd8ce2d349f94374049b6b6fe89571b0f353db3eaeb2462175c25032"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def verify(
    path: Path,
    parent_result_path: Path,
    smoke_acceptance_path: Path,
    inner_fold: int,
    technical_smoke: bool,
) -> dict:
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": "677",
        "parent_experiment_id": "659",
        "parent_passed": True,
        "parent_decision": "ACCEPT_FULL_COMPONENT_ROUTE",
        "smoke_lane_open": False,
        "full_wave_open": True,
        "technical_smoke_only": False,
        "technical_smoke_accepted": True,
        "decision": "OPEN_FOUR_INNER_DYNAMICS_SCREENS",
        "gpu_jobs": 4,
        "gpus_per_job": 1,
        "blind_confirmation_folds": [0],
        "fold3_is_blind": False,
        "sealed_rows": 0,
        "public_used": False,
    }
    if any(gate.get(key) != value for key, value in expected.items()):
        raise ValueError("experiment-677 launch gate is closed or malformed")
    if technical_smoke:
        raise ValueError("technical smoke lane is closed after artifact acceptance")
    if gate.get("inner_folds") != [1, 2, 3, 4]:
        raise ValueError("full inner-fold authorization is incomplete")
    if inner_fold not in gate.get("inner_folds", []):
        raise ValueError("requested inner fold is not authorized")
    parent_sha = str(gate.get("parent_result_sha256", ""))
    if parent_sha != PARENT_RESULT_SHA256 or sha256_file(parent_result_path) != PARENT_RESULT_SHA256:
        raise ValueError("terminal parent artifact SHA-256 mismatch")
    parent = json.loads(parent_result_path.read_text(encoding="utf-8"))
    parent_expected = {
        "schema_version": 1,
        "experiment_id": "659",
        "decision": "ACCEPT_FULL_COMPONENT_ROUTE",
        "passed": True,
        "fold_wins": 5,
        "public_used": False,
        "sealed_rows": 0,
        "threshold_tuned": False,
    }
    if any(parent.get(key) != value for key, value in parent_expected.items()):
        raise ValueError("terminal parent artifact contract mismatch")
    required_parent_gates = {
        "at_least_four_of_five_fold_wins",
        "corrected_to_regressed_at_least_1_5",
        "flammable_false_negatives_do_not_increase",
        "mean_delta_at_least_0_0015",
        "no_category_drop_below_minus_0_002",
        "screen_folds_reproduced_exactly",
    }
    parent_gates = parent.get("gates", {})
    if set(parent_gates) != required_parent_gates or not all(parent_gates.values()):
        raise ValueError("terminal parent acceptance gates are incomplete")
    if set(parent.get("folds", {})) != {"0", "1", "2", "3", "4"}:
        raise ValueError("terminal parent artifact does not cover five folds")
    if (
        gate.get("smoke_acceptance_sha256") != SMOKE_ACCEPTANCE_SHA256
        or sha256_file(smoke_acceptance_path) != SMOKE_ACCEPTANCE_SHA256
    ):
        raise ValueError("technical smoke acceptance file SHA-256 mismatch")
    smoke = json.loads(smoke_acceptance_path.read_text(encoding="utf-8"))
    smoke_contract_sha256 = str(smoke.get("contract_sha256", ""))
    if smoke_contract_sha256 != canonical_sha256(
        {key: value for key, value in smoke.items() if key != "contract_sha256"}
    ):
        raise ValueError("technical smoke acceptance self-hash mismatch")
    smoke_expected = {
        "schema_version": 1,
        "experiment_id": "677",
        "inner_validation_fold": 1,
        "decision": "ACCEPT_TECHNICAL_SMOKE_ONLY",
        "artifact_schema_passed": True,
        "ordered_runtime_binding_passed": True,
        "finite_scores": True,
        "labels_read": 0,
        "sealed_rows": 0,
        "public_used": False,
        "scientific_quality_evidence": False,
        "optimizer_steps": 2,
        "rows_per_checkpoint": 4,
    }
    if any(smoke.get(key) != value for key, value in smoke_expected.items()):
        raise ValueError("technical smoke acceptance contract mismatch")
    return gate


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--gate", type=Path, required=True)
    result.add_argument("--parent-result", type=Path, required=True)
    result.add_argument("--smoke-acceptance", type=Path, required=True)
    result.add_argument("--inner-fold", type=int, choices=(1, 2, 3, 4), required=True)
    result.add_argument("--technical-smoke", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(
        json.dumps(
            verify(
                arguments.gate,
                arguments.parent_result,
                arguments.smoke_acceptance,
                arguments.inner_fold,
                arguments.technical_smoke,
            ),
            indent=2,
            sort_keys=True,
        )
    )

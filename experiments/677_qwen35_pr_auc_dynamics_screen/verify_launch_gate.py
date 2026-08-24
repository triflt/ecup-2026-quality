from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PARENT_RESULT_SHA256 = "e2999faf11f4bbc98c9adaea5a7e8514c50d03b6571b8c62f4cced0260b4c91a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, parent_result_path: Path, inner_fold: int, technical_smoke: bool) -> dict:
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": "677",
        "parent_experiment_id": "659",
        "parent_passed": True,
        "parent_decision": "ACCEPT_FULL_COMPONENT_ROUTE",
        "smoke_lane_open": True,
        "full_wave_open": False,
        "technical_smoke_only": True,
        "decision": "OPEN_TECHNICAL_SMOKE_ONLY",
        "gpu_jobs": 1,
        "gpus_per_job": 1,
        "blind_confirmation_folds": [0],
        "fold3_is_blind": False,
        "sealed_rows": 0,
        "public_used": False,
    }
    if any(gate.get(key) != value for key, value in expected.items()):
        raise ValueError("experiment-677 launch gate is closed or malformed")
    if not technical_smoke:
        raise ValueError("full inner wave is closed until technical smoke artifact acceptance")
    if gate.get("inner_folds") != [1] or inner_fold != 1:
        raise ValueError("pre-smoke gate authorizes only one inner-fold-1 technical smoke")
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
    return gate


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--gate", type=Path, required=True)
    result.add_argument("--parent-result", type=Path, required=True)
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
                arguments.inner_fold,
                arguments.technical_smoke,
            ),
            indent=2,
            sort_keys=True,
        )
    )

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(path: Path, inner_fold: int) -> dict:
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": "677",
        "parent_experiment_id": "659",
        "parent_passed": True,
        "parent_decision": "ACCEPT_FULL_COMPONENT_ROUTE",
        "training_lane_open": True,
        "decision": "OPEN_FOUR_INNER_DYNAMICS_SCREENS",
        "gpu_jobs": 4,
        "gpus_per_job": 1,
        "sealed_rows": 0,
        "public_used": False,
    }
    if any(gate.get(key) != value for key, value in expected.items()):
        raise ValueError("experiment-677 launch gate is closed or malformed")
    if inner_fold not in gate.get("inner_folds", []):
        raise ValueError("requested inner fold is not authorized")
    parent_sha = str(gate.get("parent_result_sha256", ""))
    if len(parent_sha) != 64 or any(character not in "0123456789abcdef" for character in parent_sha):
        raise ValueError("parent result SHA-256 is missing")
    return gate


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--gate", type=Path, required=True)
    result.add_argument("--inner-fold", type=int, choices=(1, 2, 3, 4), required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(verify(arguments.gate, arguments.inner_fold), indent=2, sort_keys=True))

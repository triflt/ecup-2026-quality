from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def verify(path: Path, fold: int) -> dict[str, Any]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected = (
        gate.get("experiment_id") == "661"
        and gate.get("control_experiment_id") == "641"
        and gate.get("target_consumer_experiment_id") == "659"
        and gate.get("control_screen_passed") is True
        and gate.get("decision") == "OPEN_REMAINING_CONTROL_FOLDS"
        and gate.get("allowed_folds") == [1, 2, 4]
        and gate.get("runtime_backend") == "legacy_eager"
        and gate.get("micro_batch_size") == 2
        and gate.get("gradient_accumulation") == 8
        and gate.get("effective_batch_size") == 16
        and gate.get("threshold") == 0.0
        and gate.get("threshold_tuned") is False
        and gate.get("sealed_rows") == 0
        and gate.get("public_used") is False
    )
    if not expected or fold not in gate.get("allowed_folds", []):
        raise ValueError("experiment-661 continuation gate mismatch")
    return gate


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--gate", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(verify(args.gate, args.fold), indent=2, sort_keys=True))


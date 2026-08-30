from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import train_lora as control


EXPERIMENT_ID = "679"
CONTROL_EXPERIMENT_ID = "641"
CONTROL_LEARNING_RATE = 2e-4
CANDIDATE_LEARNING_RATE = 1e-4


def rewrite_contract(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("experiment_id") != CONTROL_EXPERIMENT_ID:
        raise ValueError("unexpected control report experiment")
    if report.get("objective") != "class_only":
        raise ValueError("unexpected control objective")
    if report.get("model_id") != "Qwen/Qwen3.5-4B":
        raise ValueError("unexpected control model")
    if not report.get("technical_smoke") and report.get("optimizer_steps_executed") != 306:
        raise ValueError("optimizer step count changed")
    source_contract_sha256 = report.get("contract_sha256")
    if not isinstance(source_contract_sha256, str):
        raise ValueError("control report lacks its self hash")
    result = dict(report)
    result.pop("contract_sha256")
    result.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "control_experiment_id": CONTROL_EXPERIMENT_ID,
            "changed_factor": "learning_rate_only",
            "control_learning_rate": CONTROL_LEARNING_RATE,
            "candidate_learning_rate": CANDIDATE_LEARNING_RATE,
            "source_control_contract_sha256": source_contract_sha256,
            "submission_base_model": "Qwen/Qwen3.5-4B",
            "uses_27b_at_training": False,
            "uses_27b_at_inference": False,
        }
    )
    result["contract_sha256"] = control.canonical_sha256(result)
    return result


def run(args: Any) -> dict[str, Any]:
    if control.LEARNING_RATE != CONTROL_LEARNING_RATE:
        raise ValueError("parent learning rate differs from the frozen control")
    control.LEARNING_RATE = CANDIDATE_LEARNING_RATE
    try:
        parent_report = control.run(CONTROL_EXPERIMENT_ID, args)
    finally:
        control.LEARNING_RATE = CONTROL_LEARNING_RATE
    result = rewrite_contract(parent_report)
    output_contract = args.output_dir / "output_contract.json"
    output_contract.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = control.parser_for(CONTROL_EXPERIMENT_ID).parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

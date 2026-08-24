from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import train_lora as control

EXPERIMENT_ID = "680"
CONTROL_EXPERIMENT_ID = "641"
FLAMMABLE = "Легковоспламеняющиеся"


def rewrite_contract(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("experiment_id") != CONTROL_EXPERIMENT_ID:
        raise ValueError("unexpected control report")
    if report.get("model_id") != "Qwen/Qwen3.5-4B" or report.get("objective") != "class_only":
        raise ValueError("unexpected model/objective")
    if not report.get("technical_smoke") and report.get("optimizer_steps_executed") != 143:
        raise ValueError("specialist update count drifted")
    source_hash = report.get("contract_sha256")
    result = dict(report)
    result.pop("contract_sha256")
    result.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "control_experiment_id": CONTROL_EXPERIMENT_ID,
            "changed_factor": "remove_bad_training_occurrences",
            "source_control_contract_sha256": source_hash,
            "category_scope": FLAMMABLE,
            "submission_base_model": "Qwen/Qwen3.5-4B",
            "uses_27b_at_training": False,
            "uses_27b_at_inference": False,
        }
    )
    result["contract_sha256"] = control.canonical_sha256(result)
    return result


def run(args: Any) -> dict[str, Any]:
    parent = control.run(CONTROL_EXPERIMENT_ID, args)
    result = rewrite_contract(parent)
    (args.output_dir / "output_contract.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    args = control.parser_for(CONTROL_EXPERIMENT_ID).parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


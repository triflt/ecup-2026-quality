from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from promotion_gate import (
    AUTHORITY,
    EVIDENCE_FIELDS,
    HEX64_FIELDS,
    canonical_sha256,
    sha256_file,
    verify_promotion_gate,
)

STAGE_RULES = {
    "blind3": ([3], "OPEN_CONFIRMATION_FOLDS124"),
    "confirmation": ([1, 2, 4], "OPEN_FULL_REPLAY"),
}


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite promotion gate")
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    body = dict(evaluation)
    evaluation_sha = body.pop("evaluation_sha256", None)
    if evaluation_sha != canonical_sha256(body):
        raise ValueError("scientific evaluation self-hash mismatch")
    if args.stage not in STAGE_RULES:
        raise ValueError("unsupported promotion stage")
    folds, decision = STAGE_RULES[args.stage]
    expected = {
        "experiment_id": "686",
        "stage": args.stage,
        "folds": folds,
        "passed": True,
        "decision": decision,
        "validation_labels_read_by_training": 0,
        "sealed_rows": 0,
        "public_used": False,
    }
    if any(evaluation.get(key) != value for key, value in expected.items()):
        raise ValueError("evaluation does not authorize promotion")
    gates = evaluation.get("gates")
    if not isinstance(gates, dict) or not gates or any(value is not True for value in gates.values()):
        raise ValueError("scientific evaluation has a failed gate")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.job_id):
        raise ValueError("invalid evaluator job identity")
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_output_metadata_sha256) or not re.fullmatch(
        r"[0-9a-f]{40}", args.main_revision
    ):
        raise ValueError("invalid immutable verifier provenance")
    parent = {
        "parent_promotion_gate_sha256": None,
        "parent_promotion_gate_file_sha256": None,
        "parent_promotion_evaluation_file_sha256": None,
        "parent_promotion_source": None,
    }
    if args.stage == "confirmation":
        if (
            args.parent_gate is None
            or args.parent_evaluation is None
            or args.parent_source is None
        ):
            raise ValueError("confirmation gate requires accepted blind3 lineage")
        parent_gate = verify_promotion_gate(
            args.parent_gate,
            args.parent_evaluation,
            expected_stage="blind3",
            expected_folds=[3],
            expected_decision="OPEN_CONFIRMATION_FOLDS124",
            expected_source=args.parent_source,
        )
        parent = {
            "parent_promotion_gate_sha256": parent_gate["promotion_gate_sha256"],
            "parent_promotion_gate_file_sha256": sha256_file(args.parent_gate),
            "parent_promotion_evaluation_file_sha256": sha256_file(
                args.parent_evaluation
            ),
            "parent_promotion_source": args.parent_source,
        }
    elif any(
        value is not None
        for value in (args.parent_gate, args.parent_evaluation, args.parent_source)
    ):
        raise ValueError("blind3 gate must not consume parent lineage")
    gate = {
        "schema_version": 1,
        **expected,
        "gates": gates,
        **{
            field: evaluation[field]
            for field in HEX64_FIELDS
            if field
            not in {"evaluation_file_sha256", "source_output_metadata_sha256"}
        },
        "evaluation_file_sha256": sha256_file(args.evaluation),
        **{field: evaluation[field] for field in EVIDENCE_FIELDS},
        "evaluation_code_revision": evaluation["evaluation_code_revision"],
        "evaluator_job_id": args.job_id,
        "evaluator_job_state": "SUCCEEDED",
        "evaluation_output_source": args.output_source,
        "source_output_metadata_sha256": args.source_output_metadata_sha256,
        "gate_authority": AUTHORITY,
        "gate_main_revision": args.main_revision,
        **parent,
    }
    gate["promotion_gate_sha256"] = canonical_sha256(gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verify_promotion_gate(
        args.output,
        args.evaluation,
        expected_stage=args.stage,
        expected_folds=folds,
        expected_decision=decision,
        expected_source=args.output_source,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGE_RULES), required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-source", required=True)
    parser.add_argument("--source-output-metadata-sha256", required=True)
    parser.add_argument("--main-revision", required=True)
    parser.add_argument("--parent-gate", type=Path)
    parser.add_argument("--parent-evaluation", type=Path)
    parser.add_argument("--parent-source")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments), ensure_ascii=False, indent=2, sort_keys=True))

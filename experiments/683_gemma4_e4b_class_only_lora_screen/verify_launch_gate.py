from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import EXPERIMENT_ID, SCREEN_FOLDS, verify_self_hash


def verify(gate_path: Path, runtime_dir: Path, fold: int, technical_smoke: bool) -> dict:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    verify_self_hash(gate)
    expected_decision = "OPEN_TECHNICAL_SMOKE_ONLY" if technical_smoke else "OPEN_SCREEN"
    if gate.get("experiment_id") != EXPERIMENT_ID or gate.get("decision") != expected_decision:
        raise ValueError("launch gate is closed")
    allowed = [0] if technical_smoke else list(SCREEN_FOLDS)
    if gate.get("allowed_folds") != allowed or fold not in allowed:
        raise ValueError("fold is not allowed")
    if gate.get("max_gpu_jobs") != (1 if technical_smoke else 2):
        raise ValueError("GPU budget drift")
    if gate.get("public_used") is not False or gate.get("sealed_rows") != 0:
        raise ValueError("forbidden scope opened")
    audit = json.loads((runtime_dir / "runtime_audit.json").read_text(encoding="utf-8"))
    verify_self_hash(audit)
    if audit.get("contract_sha256") != gate["runtime_contract_sha256"][str(fold)]:
        raise ValueError("runtime contract mismatch")
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=SCREEN_FOLDS, required=True)
    parser.add_argument("--technical-smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.gate, args.runtime_dir, args.fold, args.technical_smoke), indent=2, sort_keys=True))

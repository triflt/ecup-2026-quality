from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(gate_path: Path, runtime_dir: Path, fold: int) -> dict:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    required = (
        gate.get("experiment_id") == "680"
        and gate.get("control_experiment_id") == "641"
        and gate.get("decision") == "OPEN_SCREEN"
        and gate.get("training_lane_open") is True
        and gate.get("allowed_folds") == [0, 3]
        and gate.get("changed_factor") == "remove_bad_training_occurrences"
        and gate.get("learning_rate") == 0.0002
        and gate.get("expected_optimizer_steps") == 143
        and gate.get("threshold") == 0.0
        and gate.get("threshold_tuned") is False
        and gate.get("public_used") is False
        and gate.get("sealed_rows") == 0
    )
    if not required or fold not in gate["allowed_folds"]:
        raise ValueError("experiment-680 gate is closed or inconsistent")
    audit = json.loads((runtime_dir / "runtime_audit.json").read_text(encoding="utf-8"))
    if audit.get("contract_sha256") != gate["runtime_contract_sha256"][str(fold)]:
        raise ValueError("runtime contract mismatch")
    if audit.get("bad_train_occurrences") != 0 or audit.get("train_occurrences") != 2280:
        raise ValueError("runtime is not the frozen flammable-only multiset")
    for name in ("train.jsonl", "validation.jsonl"):
        if sha256_file(runtime_dir / name) != gate["runtime_payload_sha256"][str(fold)][name]:
            raise ValueError(f"runtime payload mismatch: {name}")
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 3), required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.gate, args.runtime_dir, args.fold), indent=2, sort_keys=True))


from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import (
    EXPERIMENT_ID,
    SCREEN_FOLDS,
    canonical_sha256,
    sha256_file,
    verify_self_hash,
)


def open_gate(smoke_gate_path: Path, acceptance_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite screen gate")
    smoke_gate = json.loads(smoke_gate_path.read_text(encoding="utf-8"))
    verify_self_hash(smoke_gate)
    if not (
        smoke_gate.get("experiment_id") == EXPERIMENT_ID
        and smoke_gate.get("decision") == "OPEN_TECHNICAL_SMOKE_ONLY"
        and smoke_gate.get("allowed_folds") == [0]
        and smoke_gate.get("max_gpu_jobs") == 1
        and smoke_gate.get("public_used") is False
    ):
        raise ValueError("technical-smoke gate is not frozen")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    digest = acceptance.pop("acceptance_sha256", None)
    if digest != canonical_sha256(acceptance):
        raise ValueError("acceptance audit self-hash mismatch")
    if not (
        acceptance.get("experiment_id") == EXPERIMENT_ID
        and acceptance.get("fold") == 0
        and acceptance.get("technical_smoke") is True
        and acceptance.get("rows") == 2
        and acceptance.get("exact_runtime_binding") is True
        and acceptance.get("adapter_reload_parity") is True
        and acceptance.get("target_module_count") == 132
        and 0 < int(acceptance.get("peak_cuda_memory_bytes", 0)) < 75 * 1024**3
        and acceptance.get("decision") == "ACCEPT_ARTIFACT"
    ):
        raise ValueError("technical smoke was not fully accepted")
    acceptance["acceptance_sha256"] = digest
    result = dict(smoke_gate)
    result.update(
        {
            "decision": "OPEN_SCREEN",
            "allowed_folds": list(SCREEN_FOLDS),
            "max_gpu_jobs": 2,
            "technical_smoke_acceptance_sha256": sha256_file(acceptance_path),
            "technical_smoke_archive_sha256": acceptance["archive_sha256"],
            "technical_smoke_peak_cuda_memory_bytes": acceptance[
                "peak_cuda_memory_bytes"
            ],
        }
    )
    result.pop("contract_sha256", None)
    result["contract_sha256"] = canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-gate", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            open_gate(args.smoke_gate, args.acceptance, args.output),
            indent=2,
            sort_keys=True,
        )
    )

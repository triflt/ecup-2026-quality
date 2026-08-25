from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def self_hashed(path: Path, field: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    digest = body.pop(field, None)
    if digest != canonical_sha256(body):
        raise ValueError(f"self-hash mismatch: {path.name}")
    return value


def sanitize(source_runtime: Path, pair_runtime: Path, transport: Path) -> dict[str, Any]:
    validation = source_runtime / "validation.jsonl"
    if not validation.is_file() or validation.is_symlink():
        raise FileNotFoundError("regular validation transport file is required")
    source_audit = self_hashed(source_runtime / "runtime_audit.json", "contract_sha256")
    pair_audit = self_hashed(pair_runtime / "runtime_audit.json", "contract_sha256")
    transport_audit = self_hashed(transport, "transport_acceptance_sha256")
    if source_audit.get("outer_fold") != 3 or pair_audit.get("outer_fold") != 3:
        raise ValueError("sanitizer is frozen to outer fold3")
    if (
        pair_audit.get("source_641_runtime_contract_sha256")
        != source_audit.get("contract_sha256")
    ):
        raise ValueError("pair runtime is not derived from the transported source runtime")
    derived_validation_sha = pair_audit.get("derived_680_output_sha256", {}).get(
        "validation.jsonl"
    )
    if not isinstance(derived_validation_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", derived_validation_sha
    ):
        raise ValueError("pair runtime lacks the filtered validation checksum")
    expected = [
        source_audit.get("output_sha256", {}).get("validation.jsonl"),
        transport_audit.get("accepted_files", {})
        .get("source_runtime/validation.jsonl", {})
        .get("sha256"),
    ]
    actual = sha256_file(validation)
    if any(value != actual for value in expected):
        raise ValueError("validation transport differs from frozen source/pair contracts")
    bytes_verified = validation.stat().st_size
    validation.unlink()
    if validation.exists() or validation.is_symlink():
        raise RuntimeError("validation transport file survived sanitization")
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "687",
        "outer_fold": 3,
        "source_runtime_contract_sha256": source_audit["contract_sha256"],
        "pair_runtime_contract_sha256": pair_audit["contract_sha256"],
        "transport_acceptance_sha256": transport_audit[
            "transport_acceptance_sha256"
        ],
        "validation_sha256": actual,
        "derived_filtered_validation_sha256": derived_validation_sha,
        "validation_bytes_verified": bytes_verified,
        "outer_validation_transport_checksum_verified": True,
        "outer_validation_rows_consumed_by_probe": 0,
        "outer_validation_labels_read": 0,
        "decision": "ACCEPT_SANITIZED_PROBE_SOURCE",
    }
    result["sanitizer_acceptance_sha256"] = canonical_sha256(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--pair-runtime", type=Path, required=True)
    parser.add_argument("--transport-acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite sanitizer acceptance")
    value = sanitize(args.source_runtime, args.pair_runtime, args.transport_acceptance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value), flush=True)

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_pair_runtime import canonical_sha256
from verify_pair_runtime import verify as verify_pair_runtime

EXPERIMENT_ID = "685"
REQUIRED_FILES = (
    "r0_acceptance.json",
    "runtime/pairs.jsonl",
    "runtime/runtime_audit.json",
    "runtime/train_targets.jsonl",
    "source_runtime/runtime_audit.json",
    "source_runtime/train.jsonl",
    "source_runtime/validation.jsonl",
)
REQUIRED_DIRECTORIES = {"runtime", "source_runtime"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_transport_metadata(relative: Path) -> bool:
    return any(
        part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
        for part in relative.parts
    )


def inventory(root: Path) -> tuple[dict[str, bytes], list[str]]:
    if not root.is_dir():
        raise FileNotFoundError("remote pair input is not a directory")
    expected = set(REQUIRED_FILES)
    payloads: dict[str, bytes] = {}
    ignored: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"remote pair input contains a symlink: {relative}")
        if path.is_dir():
            name = relative.as_posix()
            if name in REQUIRED_DIRECTORIES:
                continue
            if is_transport_metadata(relative):
                ignored.append(name)
                continue
            raise ValueError(f"unexpected remote pair input directory: {name}")
        name = relative.as_posix()
        if name in expected:
            payloads[name] = path.read_bytes()
        elif is_transport_metadata(relative):
            ignored.append(name)
        else:
            raise ValueError(f"unexpected remote pair input member: {name}")
    missing = expected - set(payloads)
    if missing:
        raise FileNotFoundError(f"remote pair input is incomplete: {sorted(missing)}")
    if any(not payload for payload in payloads.values()):
        raise ValueError("remote pair input contains an empty required payload")
    return payloads, ignored


def verify_self_hash(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    digest = body.pop(field, None)
    if digest != canonical_sha256(body):
        raise ValueError(f"{field} self-hash mismatch")
    return str(digest)


def stage(
    source: Path,
    output: Path,
    *,
    fold: int,
    expected_pair_acceptance_sha256: str,
    expected_pair_runtime_contract_sha256: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite staged training input")
    payloads, ignored = inventory(source)
    r0_acceptance = json.loads(payloads["r0_acceptance.json"])
    acceptance_sha = verify_self_hash(r0_acceptance, "acceptance_sha256")
    if acceptance_sha != expected_pair_acceptance_sha256:
        raise ValueError("R0 acceptance differs from the frozen consumer contract")
    if (
        r0_acceptance.get("experiment_id") != EXPERIMENT_ID
        or int(r0_acceptance.get("outer_fold", -1)) != fold
        or r0_acceptance.get("decision") != "ACCEPT_R0_OPEN_TECHNICAL_SMOKE"
        or r0_acceptance.get("runtime_contract_sha256")
        != expected_pair_runtime_contract_sha256
        or int(r0_acceptance.get("validation_labels_read", -1)) != 0
        or int(r0_acceptance.get("sealed_rows_used", -1)) != 0
        or r0_acceptance.get("public_used") is not False
    ):
        raise ValueError("R0 acceptance does not open this frozen training consumer")

    pair_acceptance = verify_pair_runtime(source / "runtime")
    if (
        pair_acceptance["acceptance_sha256"] != acceptance_sha
        or pair_acceptance["runtime_contract_sha256"]
        != expected_pair_runtime_contract_sha256
    ):
        raise ValueError("independent pair-runtime verification differs from R0 acceptance")

    output.mkdir(parents=True)
    for relative, payload in payloads.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    accepted_files = {
        relative: {"sha256": sha256_bytes(payload), "size": len(payload)}
        for relative, payload in sorted(payloads.items())
    }
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": "TRAINING_INPUT_EXACT_WHITELIST",
        "outer_fold": fold,
        "accepted_files": accepted_files,
        "ignored_transport_metadata": ignored,
        "pair_runtime_contract_sha256": expected_pair_runtime_contract_sha256,
        "pair_runtime_acceptance_sha256": expected_pair_acceptance_sha256,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "decision": "ACCEPT_TRAINING_INPUT",
    }
    result["transport_acceptance_sha256"] = canonical_sha256(result)
    (output / "transport_acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--expected-pair-acceptance-sha256", required=True)
    parser.add_argument("--expected-pair-runtime-contract-sha256", required=True)
    args = parser.parse_args()
    for value in (
        args.expected_pair_acceptance_sha256,
        args.expected_pair_runtime_contract_sha256,
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("frozen digest must be 64 lowercase hexadecimal characters")
    print(
        json.dumps(
            stage(
                args.source,
                args.output,
                fold=args.fold,
                expected_pair_acceptance_sha256=args.expected_pair_acceptance_sha256,
                expected_pair_runtime_contract_sha256=(
                    args.expected_pair_runtime_contract_sha256
                ),
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

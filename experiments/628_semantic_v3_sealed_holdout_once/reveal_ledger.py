"""Atomic one-way ledger for the single permitted sealed evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "frozen_recipe_sha256",
    "sealed_acceptance_policy_sha256",
    "evaluator_sha256",
    "sealed_input_identity_sha256",
    "revealed_at_utc",
}


def record_reveal_once(path: Path, record: dict[str, Any]) -> None:
    if set(record) != REQUIRED_FIELDS:
        raise ValueError("sealed reveal record schema mismatch")
    for field in REQUIRED_FIELDS - {"revealed_at_utc"}:
        value = record[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"invalid SHA-256 field: {field}")
    payload = {"schema_version": "sealed_reveal_ledger_v1", "evaluation_count": 1, **record}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def assert_single_reveal(path: Path, *, frozen_recipe_sha256: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "sealed_reveal_ledger_v1":
        raise ValueError("sealed reveal ledger schema mismatch")
    if value.get("evaluation_count") != 1:
        raise ValueError("sealed evaluation count must equal one")
    if value.get("frozen_recipe_sha256") != frozen_recipe_sha256:
        raise ValueError("sealed reveal used a different frozen recipe")
    return value

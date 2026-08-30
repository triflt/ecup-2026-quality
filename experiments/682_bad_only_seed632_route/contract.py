from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC_PATH = HERE / "frozen_spec.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def verify_self_hash(value: Mapping[str, Any], key: str) -> None:
    observed = value.get(key)
    if not isinstance(observed, str) or not SHA256_RE.fullmatch(observed):
        raise ValueError(f"missing or malformed {key}")
    payload = {name: item for name, item in value.items() if name != key}
    if canonical_sha256(payload) != observed:
        raise ValueError(f"{key} mismatch")


def require_sha(path: Path, expected: str, *, name: str | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        label = name or path.name
        raise ValueError(f"{label} checksum mismatch: expected={expected} actual={actual}")


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = load_json(path)
    verify_self_hash(spec, "spec_sha256")
    expected = {
        "schema_version": "exp682_bad_only_seed632_route_v1",
        "experiment_id": "682",
        "parent_full_system": "140",
        "upstream_component_experiment": "632",
        "upstream_seed": 31415,
        "validation": "semantic_family_v3",
        "folds": [0, 1, 2, 3, 4],
        "categories": ["БАД", "Легковоспламеняющиеся"],
        "uses_public_for_selection": False,
        "sealed_rows_allowed": 0,
        "auxiliary_head_used_for_verdict": False,
    }
    mismatches = {
        key: {"expected": value, "actual": spec.get(key)}
        for key, value in expected.items()
        if spec.get(key) != value
    }
    if mismatches:
        raise ValueError(f"frozen spec mismatch: {mismatches}")
    return spec


def add_self_hash(value: dict[str, Any], key: str) -> dict[str, Any]:
    if key in value:
        raise ValueError(f"refusing to replace existing {key}")
    result = dict(value)
    result[key] = canonical_sha256(result)
    return result

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFY = load(
    "verify_627",
    ROOT / "experiments/627_semantic_v3_final_recipe_freeze/verify_manifest.py",
)
LEDGER = load(
    "ledger_628",
    ROOT / "experiments/628_semantic_v3_sealed_holdout_once/reveal_ledger.py",
)


def test_recipe_manifest_is_fail_closed() -> None:
    schema = json.loads(
        (ROOT / "experiments/627_semantic_v3_final_recipe_freeze/manifest_schema_v1.json")
        .read_text(encoding="utf-8")
    )
    manifest = {field: "a" * 64 for field in schema["required_sha256_fields"]}
    manifest.update(
        schema_version="final_recipe_manifest_v1",
        selected_recipe_count=1,
        component_weights={"original": 1.0},
        route_weights={"BAD": 1.0, "flammable": 1.0},
        seed=42,
        sealed_holdout_used=False,
    )
    VERIFY.verify_manifest(manifest, schema)
    manifest["selected_recipe_count"] = 2
    with pytest.raises(ValueError, match="exactly one"):
        VERIFY.verify_manifest(manifest, schema)


def test_sealed_ledger_refuses_second_reveal(tmp_path: Path) -> None:
    path = tmp_path / "sealed_reveal.json"
    record = {
        "frozen_recipe_sha256": "a" * 64,
        "sealed_acceptance_policy_sha256": "b" * 64,
        "evaluator_sha256": "c" * 64,
        "sealed_input_identity_sha256": "d" * 64,
        "revealed_at_utc": "2026-08-23T00:00:00Z",
    }
    LEDGER.record_reveal_once(path, record)
    assert LEDGER.assert_single_reveal(path, frozen_recipe_sha256="a" * 64)[
        "evaluation_count"
    ] == 1
    with pytest.raises(FileExistsError):
        LEDGER.record_reveal_once(path, record)

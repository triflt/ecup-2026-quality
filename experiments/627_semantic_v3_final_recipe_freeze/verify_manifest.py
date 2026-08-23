"""Fail-closed verifier for the one recipe frozen before sealed evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def verify_manifest(manifest: dict[str, Any], schema: dict[str, Any]) -> None:
    if manifest.get("schema_version") != schema.get("schema_version"):
        raise ValueError("recipe manifest schema mismatch")
    for field in schema["required_sha256_fields"]:
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"missing or invalid SHA-256 field: {field}")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"non-hex SHA-256 field: {field}") from error
    missing = [field for field in schema["required_value_fields"] if field not in manifest]
    if missing:
        raise ValueError(f"missing required recipe fields: {missing}")
    if manifest["selected_recipe_count"] != 1:
        raise ValueError("exactly one final recipe must be selected")
    if manifest["sealed_holdout_used"] is not False:
        raise ValueError("recipe must be frozen before sealed evaluation")
    if not isinstance(manifest["component_weights"], dict) or not manifest["component_weights"]:
        raise ValueError("component weights must be a nonempty mapping")
    if not isinstance(manifest["route_weights"], dict) or not manifest["route_weights"]:
        raise ValueError("route weights must be a nonempty mapping")
    if not isinstance(manifest["seed"], int):
        raise TypeError("seed must be an integer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--schema", type=Path, default=HERE / "manifest_schema_v1.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    verify_manifest(manifest, schema)
    print(json.dumps({"decision": "GO", "selected_recipe_count": 1}, sort_keys=True))


if __name__ == "__main__":
    main()

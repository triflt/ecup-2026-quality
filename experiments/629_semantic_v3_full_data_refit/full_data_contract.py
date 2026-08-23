"""Fail-closed planning and acceptance checks for the single full-data refit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HEX_FIELDS_628 = {
    "frozen_recipe_manifest_sha256",
    "sealed_acceptance_policy_sha256",
    "sealed_evaluator_sha256",
    "sealed_input_identity_sha256",
}
REQUIRED_RECIPE_SHA_FIELDS = {
    "code_commit_sha",
    "model_base_revision_sha256",
    "component_artifact_sha256",
    "auxiliary_head_sha256",
    "training_config_sha256",
    "data_registry_sha256",
    "fold_registry_sha256",
    "threshold_manifest_sha256",
    "fusion_route_config_sha256",
    "explanation_renderer_sha256",
    "closed_concept_vocabulary_sha256",
    "reasoning_schema_sha256",
    "data_selection_sha256",
    "dependency_decisions_sha256",
    "sealed_acceptance_policy_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"invalid SHA-256 field: {field}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"non-hex SHA-256 field: {field}") from error
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def verify_accepted_628(result: dict[str, Any], *, recipe_sha256: str) -> None:
    if result.get("schema_version") != "sealed_acceptance_result_v1":
        raise ValueError("experiment 628 result schema mismatch")
    if result.get("experiment_id") != 628:
        raise ValueError("parent result is not experiment 628")
    if result.get("status") != "accepted" or result.get("decision") != "ACCEPT":
        raise ValueError("experiment 628 was not accepted")
    if result.get("sealed_evaluation_count") != 1:
        raise ValueError("accepted 628 must record exactly one sealed evaluation")
    for field in HEX_FIELDS_628:
        _sha256(result.get(field), field)
    if result["frozen_recipe_manifest_sha256"] != recipe_sha256:
        raise ValueError("experiment 628 used a different frozen recipe")


def verify_full_data_manifest(value: dict[str, Any]) -> list[dict[str, Any]]:
    if value.get("schema_version") != "permitted_full_data_manifest_v1":
        raise ValueError("full-data manifest schema mismatch")
    if value.get("sealed_training_rows") != 0:
        raise ValueError("sealed rows are forbidden in full-data training")
    _sha256(value.get("data_registry_sha256"), "data_registry_sha256")
    _sha256(value.get("data_selection_sha256"), "data_selection_sha256")
    components = value.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("full-data manifest needs a nonempty component list")
    names: set[str] = set()
    for item in components:
        if not isinstance(item, dict):
            raise TypeError("component record must be an object")
        name = item.get("component")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("component names must be nonempty and unique")
        names.add(name)
        if not isinstance(item.get("training_rows"), int) or item["training_rows"] <= 0:
            raise ValueError(f"invalid training row count for component {name}")
        if not isinstance(item.get("source_experiment"), int) or item["source_experiment"] <= 0:
            raise ValueError(f"invalid source experiment for component {name}")
        if item.get("sealed_training_rows") != 0:
            raise ValueError(f"sealed rows found in component {name}")
        _sha256(item.get("training_input_sha256"), f"{name}.training_input_sha256")
        _sha256(item.get("training_config_sha256"), f"{name}.training_config_sha256")
    return components


def build_fit_plan(
    *,
    recipe_path: Path,
    accepted_628_path: Path,
    full_data_path: Path,
) -> dict[str, Any]:
    recipe_sha = sha256_file(recipe_path)
    recipe = _read_json(recipe_path)
    if recipe.get("schema_version") != "final_recipe_manifest_v1":
        raise ValueError("frozen recipe schema mismatch")
    if recipe.get("selected_recipe_count") != 1 or recipe.get("sealed_holdout_used") is not False:
        raise ValueError("recipe is not a single pre-sealed freeze")
    for field in REQUIRED_RECIPE_SHA_FIELDS:
        _sha256(recipe.get(field), field)
    if not isinstance(recipe.get("component_weights"), dict) or not recipe["component_weights"]:
        raise ValueError("frozen recipe has no components")
    if not isinstance(recipe.get("route_weights"), dict) or not recipe["route_weights"]:
        raise ValueError("frozen recipe has no route weights")
    if not isinstance(recipe.get("seed"), int) or isinstance(recipe["seed"], bool):
        raise TypeError("frozen recipe seed must be an integer")
    parent = _read_json(accepted_628_path)
    verify_accepted_628(parent, recipe_sha256=recipe_sha)
    full_data = _read_json(full_data_path)
    components = verify_full_data_manifest(full_data)
    frozen_components = set(recipe.get("component_weights", {}))
    planned_components = {item["component"] for item in components}
    if frozen_components != planned_components:
        raise ValueError("full-data components differ from the frozen recipe")
    return {
        "schema_version": "full_data_fit_plan_v1",
        "experiment_id": 629,
        "status": "ready",
        "frozen_recipe_manifest_sha256": recipe_sha,
        "accepted_628_result_sha256": sha256_file(accepted_628_path),
        "permitted_full_data_manifest_sha256": sha256_file(full_data_path),
        "sealed_training_rows": 0,
        "fits_per_component": 1,
        "seed_search_allowed": False,
        "hyperparameter_search_allowed": False,
        "jobs": [
            {
                "component": item["component"],
                "source_experiment": item["source_experiment"],
                "fit_index": 1,
                "gpu_count": 1,
                "seed": recipe["seed"],
                "training_input_sha256": item["training_input_sha256"],
                "training_config_sha256": item["training_config_sha256"],
            }
            for item in components
        ],
    }


def verify_refit_manifest(
    value: dict[str, Any],
    *,
    plan: dict[str, Any],
    artifact_root: Path,
) -> None:
    if value.get("schema_version") != "full_data_refit_manifest_v1":
        raise ValueError("refit manifest schema mismatch")
    if value.get("experiment_id") != 629 or value.get("status") != "accepted":
        raise ValueError("full-data refit is not accepted")
    for field in (
        "frozen_recipe_manifest_sha256",
        "accepted_628_result_sha256",
        "permitted_full_data_manifest_sha256",
    ):
        _sha256(value.get(field), field)
        if value[field] != plan[field]:
            raise ValueError(f"refit provenance mismatch: {field}")
    if value.get("sealed_training_rows") != 0:
        raise ValueError("sealed rows are forbidden in refit manifest")
    expected = {job["component"]: job for job in plan["jobs"]}
    fits = value.get("component_fits")
    if not isinstance(fits, list) or len(fits) != len(expected):
        raise ValueError("refit must contain exactly one record per component")
    seen: set[str] = set()
    for fit in fits:
        if not isinstance(fit, dict):
            raise TypeError("component fit must be an object")
        name = fit.get("component")
        if name not in expected or name in seen:
            raise ValueError("unknown or duplicate component fit")
        seen.add(name)
        if fit.get("fit_count") != 1 or fit.get("fit_index") != 1:
            raise ValueError(f"component {name} was not fit exactly once")
        if fit.get("source_experiment") != expected[name]["source_experiment"]:
            raise ValueError(f"component {name} changed frozen source experiment")
        if fit.get("gpu_count") != 1:
            raise ValueError(f"component {name} must use exactly one GPU")
        if fit.get("sealed_training_rows") != 0:
            raise ValueError(f"component {name} used sealed rows")
        for field in (
            "training_input_sha256",
            "training_config_sha256",
            "artifact_sha256",
            "runtime_evidence_sha256",
            "provenance_sha256",
        ):
            _sha256(fit.get(field), f"{name}.{field}")
        for field in ("training_input_sha256", "training_config_sha256"):
            if fit[field] != expected[name][field]:
                raise ValueError(f"component {name} changed frozen {field}")
        if not isinstance(fit.get("artifact_size_bytes"), int) or fit["artifact_size_bytes"] <= 0:
            raise ValueError(f"invalid artifact size for component {name}")
        if fit.get("integrity_passed") is not True:
            raise ValueError(f"integrity did not pass for component {name}")
        for path_field, hash_field in (
            ("artifact_path", "artifact_sha256"),
            ("runtime_evidence_path", "runtime_evidence_sha256"),
            ("provenance_path", "provenance_sha256"),
        ):
            relative = fit.get(path_field)
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise ValueError(f"unsafe {path_field} for component {name}")
            path = artifact_root / relative
            if not path.is_file() or sha256_file(path) != fit[hash_field]:
                raise ValueError(f"{hash_field} mismatch for component {name}")
        artifact_path = artifact_root / fit["artifact_path"]
        if artifact_path.stat().st_size != fit["artifact_size_bytes"]:
            raise ValueError(f"artifact size mismatch for component {name}")
    if seen != set(expected):
        raise ValueError("missing component fit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--recipe", required=True, type=Path)
    plan_parser.add_argument("--accepted-628", required=True, type=Path)
    plan_parser.add_argument("--full-data-manifest", required=True, type=Path)
    plan_parser.add_argument("--output", required=True, type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--plan", required=True, type=Path)
    verify_parser.add_argument("--refit-manifest", required=True, type=Path)
    verify_parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        output = build_fit_plan(
            recipe_path=args.recipe,
            accepted_628_path=args.accepted_628,
            full_data_path=args.full_data_manifest,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        plan = _read_json(args.plan)
        refit = _read_json(args.refit_manifest)
        verify_refit_manifest(refit, plan=plan, artifact_root=args.artifact_root)
        print(json.dumps({"decision": "GO", "components": len(refit["component_fits"])}, sort_keys=True))


if __name__ == "__main__":
    main()

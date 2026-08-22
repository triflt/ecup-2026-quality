from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from position_protocol import (
    AUDIT_REQUIRED_KAPPA,
    AUDIT_REQUIRED_PASS_RATE,
    EXPERIMENT_ID,
    PARSER_VERSION,
    SCREEN_FOLDS,
    apply_augmentation_plan,
    build_augmentation_plan,
    canonical_sha256,
)
from position_shared import load_dependency_light_selector, load_exp600_dependencies, sha256_file


def _require_blind_gate(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "parser_version": PARSER_VERSION,
        "decision": "GO",
        "gpu_launch_allowed": True,
    }
    mismatches = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise ValueError(f"independent blind audit has not authorized training: {mismatches}")
    if float(report.get("strict_pass_rate", -1)) < AUDIT_REQUIRED_PASS_RATE:
        raise ValueError("independent semantic preservation is below 98 percent")
    if report.get("cohen_kappa") is None or float(report["cohen_kappa"]) < AUDIT_REQUIRED_KAPPA:
        raise ValueError("independent reviewer agreement is below kappa 0.80")
    return report


def _require_manifest(path: Path, *, fold: int) -> dict:
    audit = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "parser_version": PARSER_VERSION,
        "protocol_version": "semantic_family_v3",
        "selector_source_protocol": "semantic_v3_robust_base_strict_nested_v2",
        "outer_fold": fold,
        "decision": "GO",
        "counts_labels_ids_steps_byte_parity": True,
        "outer_validation_training_occurrences": 0,
        "sealed_holdout_training_occurrences": 0,
        "bad_rows_transformed": 0,
        "inference_changed": False,
    }
    mismatches = {
        key: {"expected": value, "actual": audit.get(key)}
        for key, value in expected.items()
        if audit.get(key) != value
    }
    if mismatches or audit.get("coverage_failures"):
        raise ValueError(
            f"frozen augmentation manifest has not authorized fold {fold}: "
            f"mismatches={mismatches}, failures={audit.get('coverage_failures')}"
        )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train strict position-only route-400 screen.")
    parser.add_argument("--fold", required=True, type=int, choices=SCREEN_FOLDS)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--augmentation-audit", required=True, type=Path)
    parser.add_argument("--blind-audit-result", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--vendor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    output.mkdir(parents=True, exist_ok=True)
    blind_gate = _require_blind_gate(args.blind_audit_result.resolve())
    expected_plan = _require_manifest(args.augmentation_audit.resolve(), fold=args.fold)
    protocol, trainer = load_exp600_dependencies()
    prepared = protocol.materialize_runtime_fold(
        runtime_dir=args.runtime_dir.resolve(),
        output_dir=output / "protocol_inputs",
        component="specialist",
        outer_fold=args.fold,
    )
    environment = trainer.configure_environment(
        component="specialist",
        fold=args.fold,
        environment=dict(os.environ),
    )
    environment.update(
        {
            "ECUP_DATA": str(prepared["data"]),
            "ECUP_OOF": str(prepared["selector"]),
            "ECUP_OUTPUT_DIR": str(output),
            "ECUP_MANIFEST": str(prepared["manifest"]),
            "ECUP_IMAGES": str(args.images.resolve()),
            "ECUP_MODEL_ROOT": str(args.model_root.resolve()),
            "ECUP_VENDOR": str(args.vendor.resolve()),
        }
    )
    os.environ.clear()
    os.environ.update(environment)
    parent = trainer._load_parent("specialist")
    trainer._assert_parent_recipe(parent, "specialist", args.fold)
    trainer._install_selection_audit(
        parent,
        component="specialist",
        fold=args.fold,
        output_dir=output,
    )
    exact_select = parent.select_training
    dependency_light = load_dependency_light_selector()

    def select_with_position_view(frame, oof):
        result = exact_select(frame, oof)
        records, parent_selection = result
        light_records, _ = dependency_light.select_parent_training_records(
            frame,
            oof,
            seed=42,
            holdout_fold=args.fold,
            full_train=False,
        )
        if records != light_records:
            raise ValueError("dependency-light manifest selector differs from exact parent")
        transforms, manifest, audit = build_augmentation_plan(
            frame,
            records,
            oof["fold_ids"].astype(np.int8),
            outer_fold=args.fold,
        )
        frozen_keys = (
            "parser_version",
            "protocol_version",
            "selector_source_protocol",
            "outer_fold",
            "training_records",
            "record_order_sha256",
            "id_sequence_sha256",
            "label_sequence_sha256",
            "category_sequence_sha256",
            "manifest_sha256",
            "plan_sha256",
            "decision",
        )
        mismatches = {
            key: {"expected": expected_plan.get(key), "actual": audit.get(key)}
            for key in frozen_keys
            if expected_plan.get(key) != audit.get(key)
        }
        if mismatches:
            raise ValueError(f"runtime augmentation differs from frozen manifest: {mismatches}")
        apply_augmentation_plan(frame, transforms)
        manifest_path = output / "augmentation_manifest.runtime.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
            encoding="utf-8",
        )
        runtime = {
            **audit,
            "blind_audit_result_sha256": sha256_file(args.blind_audit_result.resolve()),
            "blind_audit_sample_sha256": blind_gate["sample_immutable_sha256"],
            "frozen_augmentation_audit_sha256": sha256_file(args.augmentation_audit.resolve()),
            "runtime_manifest_file_sha256": sha256_file(manifest_path),
        }
        runtime["runtime_audit_sha256"] = canonical_sha256(runtime)
        (output / "position_augmentation_audit.runtime.json").write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        enriched = dict(parent_selection)
        enriched["strict_position_scope"] = runtime
        return records, enriched

    parent.select_training = select_with_position_view
    parent.main()
    base_contract = trainer.validate_output_contract(
        output_dir=output,
        component="specialist",
        fold=args.fold,
    )
    runtime_path = output / "position_augmentation_audit.runtime.json"
    if not runtime_path.is_file():
        raise ValueError("training did not execute the frozen position augmentation")
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "parser_version": PARSER_VERSION,
        "outer_fold": args.fold,
        "parent_component": "route400_flammable_specialist_260",
        "base_output_contract_sha256": base_contract["contract_sha256"],
        "augmentation_runtime_sha256": sha256_file(runtime_path),
        "blind_audit_result_sha256": sha256_file(args.blind_audit_result.resolve()),
        "inference_changed": False,
        "sealed_rows_in_predictions": 0,
        "decision": "GO",
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    (output / "position_output_contract.runtime.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from protocol import (
    COMPONENTS,
    DEVELOPMENT_FOLDS,
    EXPECTED_SEED,
    EXPERIMENT_ID,
    PROTOCOL_VERSION,
    canonical_sha256,
    expected_step_policy,
    id_sequence_sha256,
    materialize_runtime_fold,
    sha256_file,
)

PARENT_RUNNERS = {
    "original": ROOT / "research/qwen3vl_lora_holdout.py",
    "specialist": ROOT / "research/qwen35_bad_family_diverse_positives_lora.py",
}
PARENT_SHA256 = {
    "original": "c30e690ad260af72fcc625c8d3e6d9ab9c5a096d8443d6d9f5f7adbcaa52123c",
    "specialist": "404f6d07965f551dfe7c7ee0120ce0a16132e1103f1db7d13b3622993f4fd6c0",
}
LOCKED_COMMON_ENVIRONMENT = {
    "SEED": "42",
    "FULL_TRAIN": "0",
    "TRAINING_MODE": "hard",
    "MODEL_CLASS": "multimodal",
    "USE_CHAT_BATCH": "1",
    "DESCRIPTION_LIMIT": "1800",
}
LOCKED_SPECIALIST_ENVIRONMENT = {
    "FAMILY_BALANCE_FLAMMABLE": "0",
    "FAMILY_DIVERSE_BAD_POSITIVES": "1",
    "FAMILY_DIVERSE_FLAMMABLE_NEGATIVES": "0",
}
FORBIDDEN_PARENT_SWITCHES = {
    "SOFT_TARGETS": (None, ""),
    "DOWNSAMPLE_MODE": (None, ""),
    "MAX_SLICE_NUMS": (None, "", "0"),
    "LAST_LOGIT_ONLY": (None, "", "0", "false"),
    "LINEAR_ONLY_TARGETS": (None, "", "0", "false"),
}


def configure_environment(
    *,
    component: str,
    fold: int,
    environment: dict[str, str],
) -> dict[str, str]:
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}")
    if fold not in DEVELOPMENT_FOLDS:
        raise ValueError(f"fold must be one of {DEVELOPMENT_FOLDS}")
    locked = dict(LOCKED_COMMON_ENVIRONMENT)
    if component == "specialist":
        locked.update(LOCKED_SPECIALIST_ENVIRONMENT)
    for key, expected in locked.items():
        current = environment.get(key)
        if current not in (None, "", expected):
            raise ValueError(f"{key} must remain at the exact parent value {expected!r}")
        environment[key] = expected
    for key, allowed in FORBIDDEN_PARENT_SWITCHES.items():
        if environment.get(key) not in allowed:
            raise ValueError(f"non-parent switch is forbidden: {key}")
    environment["HOLDOUT_FOLD"] = str(fold)
    return environment


def _load_parent(component: str):
    path = PARENT_RUNNERS[component]
    if sha256_file(path) != PARENT_SHA256[component]:
        raise ValueError(f"exact {component} parent checksum mismatch")
    spec = importlib.util.spec_from_file_location(f"_exp600_{component}_parent", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load exact {component} parent")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_parent_recipe(parent: Any, component: str, fold: int) -> None:
    expected = {
        "SEED": EXPECTED_SEED,
        "HOLDOUT_FOLD": fold,
        "FULL_TRAIN": False,
        "TRAINING_MODE": "hard",
        "MODEL_CLASS": "multimodal",
        "USE_CHAT_BATCH": True,
        "DESCRIPTION_LIMIT": 1800,
        "BATCH_SIZE": 4,
        "GRAD_ACCUM": 4,
        "EPOCHS": 1,
    }
    if component == "specialist":
        expected.update(
            {
                "FAMILY_BALANCE_FLAMMABLE": False,
                "FAMILY_DIVERSE_BAD_POSITIVES": True,
                "FAMILY_DIVERSE_FLAMMABLE_NEGATIVES": False,
            }
        )
    mismatches = {
        key: {"expected": value, "actual": getattr(parent, key, None)}
        for key, value in expected.items()
        if getattr(parent, key, None) != value
    }
    if mismatches:
        raise ValueError(f"exact parent recipe mismatch: {mismatches}")


def _install_selection_audit(
    parent: Any,
    *,
    component: str,
    fold: int,
    output_dir: Path,
) -> None:
    original_select = parent.select_training

    def audited_select(frame, oof):
        result = original_select(frame, oof)
        records = result[0] if component == "specialist" else result
        if not records:
            raise ValueError("parent selector returned no records")
        fold_ids = oof["fold_ids"].astype(np.int8)
        selected = np.asarray(records, dtype=np.int64)
        if (selected < 0).any() or (selected >= len(frame)).any():
            raise ValueError("parent selector returned an out-of-range local index")
        if bool((fold_ids[selected] == fold).any()):
            raise ValueError("outer validation entered the training multiset")
        if set(fold_ids) != set(DEVELOPMENT_FOLDS):
            raise ValueError("parent selector did not receive the five local development folds")
        selected_ids = frame.iloc[selected]["id"].astype(str).tolist()
        validation_ids = frame.loc[fold_ids == fold, "id"].astype(str).tolist()
        audit = {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "component": component,
            "outer_fold": fold,
            "seed": EXPECTED_SEED,
            "training_records": len(records),
            "training_unique_rows": len({int(index) for index in records}),
            "record_multiset_sha256": canonical_sha256(
                sorted(Counter(int(index) for index in records).items())
            ),
            "selected_ids_multiset_sha256": canonical_sha256(sorted(Counter(selected_ids).items())),
            "outer_validation_ids_sha256": id_sequence_sha256(validation_ids),
            "outer_validation_training_occurrences": 0,
            "sealed_holdout_training_occurrences": 0,
            "steps_policy": expected_step_policy(len(records)),
            "decision": "GO",
        }
        audit["audit_sha256"] = canonical_sha256(audit)
        (output_dir / "selection_audit.runtime.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    parent.select_training = audited_select


def validate_output_contract(
    *,
    output_dir: Path,
    component: str,
    fold: int,
) -> dict[str, Any]:
    predictions_path = output_dir / "lora_holdout_predictions.csv"
    report_path = output_dir / "lora_holdout_report.json"
    adapter_path = output_dir / "adapter.zip"
    selection_audit_path = output_dir / "selection_audit.runtime.json"
    protocol_audit_path = output_dir / "protocol_inputs/protocol_input_audit.json"
    mapping_path = output_dir / "protocol_inputs/id_mapping.csv"
    required = (
        predictions_path,
        report_path,
        adapter_path,
        selection_audit_path,
        protocol_audit_path,
        mapping_path,
    )
    missing = [path.name for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"candidate output contract is incomplete: {missing}")
    mapping = pd.read_csv(mapping_path, dtype={"id": str})
    expected = mapping.loc[mapping["outer_role"].eq("validation"), "id"].tolist()
    predictions = pd.read_csv(predictions_path, dtype={"id": str})
    if predictions["id"].astype(str).tolist() != expected:
        raise ValueError("candidate prediction ids/order differ from the outer development fold")
    if not predictions["fold"].astype(int).eq(fold).all():
        raise ValueError("candidate predictions contain an unexpected fold")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_audit_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_audit_path.read_text(encoding="utf-8"))
    if report.get("holdout_fold") != fold:
        raise ValueError("parent report fold mismatch")
    if report.get("train_records") != selection.get("training_records"):
        raise ValueError("parent report and selector audit disagree on training records")
    if report.get("download_failures") != 0:
        raise ValueError("parent report contains image download failures")
    if protocol.get("decision") != "GO" or selection.get("decision") != "GO":
        raise ValueError("runtime input or selection audit is not GO")
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "component": component,
        "outer_fold": fold,
        "prediction_rows": len(predictions),
        "prediction_ids_sha256": id_sequence_sha256(expected),
        "predictions_sha256": sha256_file(predictions_path),
        "parent_report_sha256": sha256_file(report_path),
        "adapter_sha256": sha256_file(adapter_path),
        "protocol_audit_sha256": sha256_file(protocol_audit_path),
        "selection_audit_sha256": sha256_file(selection_audit_path),
        "sealed_rows_in_predictions": 0,
        "sealed_rows_used_for_threshold": 0,
        "sealed_rows_used_for_evaluation": 0,
        "decision": "GO",
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    (output_dir / "output_contract.runtime.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one semantic-family-v3 baseline fold.")
    parser.add_argument("--component", required=True, choices=COMPONENTS)
    parser.add_argument("--fold", required=True, type=int, choices=DEVELOPMENT_FOLDS)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--vendor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir = output_dir / "protocol_inputs"
    prepared = materialize_runtime_fold(
        runtime_dir=args.runtime_dir.resolve(),
        output_dir=protocol_dir,
        component=args.component,
        outer_fold=args.fold,
    )
    environment = configure_environment(
        component=args.component,
        fold=args.fold,
        environment=dict(os.environ),
    )
    environment.update(
        {
            "ECUP_DATA": str(prepared["data"]),
            "ECUP_OOF": str(prepared["selector"]),
            "ECUP_OUTPUT_DIR": str(output_dir),
            "ECUP_MANIFEST": str(prepared["manifest"]),
            "ECUP_IMAGES": str(args.images.resolve()),
            "ECUP_MODEL_ROOT": str(args.model_root.resolve()),
            "ECUP_VENDOR": str(args.vendor.resolve()),
        }
    )
    os.environ.clear()
    os.environ.update(environment)
    parent = _load_parent(args.component)
    _assert_parent_recipe(parent, args.component, args.fold)
    _install_selection_audit(
        parent,
        component=args.component,
        fold=args.fold,
        output_dir=output_dir,
    )
    parent.main()
    validate_output_contract(
        output_dir=output_dir,
        component=args.component,
        fold=args.fold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

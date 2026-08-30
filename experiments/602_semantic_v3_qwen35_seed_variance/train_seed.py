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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from seed_protocol import (
    EXPERIMENT_ID,
    FOLDS,
    PROTOCOL_VERSION,
    SEEDS,
    canonical_sha256,
    expected_step_policy,
    load_exp600_dependencies,
    runtime_input_paths,
    sha256_file,
    validate_seed,
)


def configure_environment(*, seed: int, fold: int, environment: dict[str, str]) -> dict[str, str]:
    validate_seed(seed)
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}")
    locked = {
        "SEED": str(seed),
        "HOLDOUT_FOLD": str(fold),
        "FULL_TRAIN": "0",
        "TRAINING_MODE": "hard",
        "MODEL_CLASS": "multimodal",
        "USE_CHAT_BATCH": "1",
        "DESCRIPTION_LIMIT": "1800",
    }
    forbidden = {
        "SOFT_TARGETS": (None, ""),
        "DOWNSAMPLE_MODE": (None, ""),
        "MAX_SLICE_NUMS": (None, "", "0"),
        "LAST_LOGIT_ONLY": (None, "", "0", "false"),
        "LINEAR_ONLY_TARGETS": (None, "", "0", "false"),
    }
    for key, expected in locked.items():
        if environment.get(key) not in (None, "", expected):
            raise ValueError(f"{key} must remain at the predeclared value {expected!r}")
        environment[key] = expected
    for key, allowed in forbidden.items():
        if environment.get(key) not in allowed:
            raise ValueError(f"non-parent switch is forbidden: {key}")
    return environment


def _load_exact_parent(path: Path, expected_sha256: str):
    if sha256_file(path) != expected_sha256:
        raise ValueError("exact original-190 parent checksum mismatch")
    spec = importlib.util.spec_from_file_location("_exp602_original_parent", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import exact parent: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_parent_recipe(parent: Any, *, seed: int, fold: int) -> None:
    expected = {
        "SEED": seed,
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
    mismatches = {
        key: {"expected": value, "actual": getattr(parent, key, None)}
        for key, value in expected.items()
        if getattr(parent, key, None) != value
    }
    if mismatches:
        raise ValueError(f"exact parent recipe mismatch: {mismatches}")


def _install_seed_selection_audit(parent: Any, *, seed: int, fold: int, output_dir: Path) -> None:
    original_select = parent.select_training

    def audited_select(frame, oof):
        records = original_select(frame, oof)
        if not records:
            raise ValueError("parent selector returned no records")
        fold_ids = oof["fold_ids"].astype(np.int8)
        selected = np.asarray(records, dtype=np.int64)
        if (selected < 0).any() or (selected >= len(frame)).any():
            raise ValueError("parent selector returned an out-of-range local index")
        if bool((fold_ids[selected] == fold).any()):
            raise ValueError("outer validation entered the training multiset")
        if set(fold_ids) != set(FOLDS):
            raise ValueError("parent selector did not receive five development folds")
        audit = {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "outer_fold": fold,
            "training_records": len(records),
            "training_unique_rows": len({int(index) for index in records}),
            "record_multiset_sha256": canonical_sha256(sorted(Counter(map(int, records)).items())),
            "outer_validation_training_occurrences": 0,
            "sealed_holdout_training_occurrences": 0,
            "steps_policy": expected_step_policy(len(records)),
            "decision": "GO",
        }
        audit["audit_sha256"] = canonical_sha256(audit)
        (output_dir / "seed_selection_audit.runtime.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return records

    parent.select_training = audited_select


def validate_output_contract(*, output_dir: Path, seed: int, fold: int) -> dict[str, Any]:
    required = [
        output_dir / "lora_holdout_predictions.csv",
        output_dir / "lora_holdout_report.json",
        output_dir / "adapter.zip",
        output_dir / "protocol_inputs/protocol_input_audit.json",
        output_dir / "protocol_inputs/id_mapping.csv",
        output_dir / "seed_selection_audit.runtime.json",
    ]
    missing = [path.name for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"candidate output contract is incomplete: {missing}")
    mapping = pd.read_csv(output_dir / "protocol_inputs/id_mapping.csv", dtype={"id": str})
    expected_ids = mapping.loc[mapping["outer_role"].eq("validation"), "id"].tolist()
    predictions = pd.read_csv(output_dir / "lora_holdout_predictions.csv", dtype={"id": str})
    required_prediction_columns = {"id", "category", "label", "fold", "lora_score"}
    if set(predictions.columns) != required_prediction_columns:
        raise ValueError("candidate predictions schema mismatch")
    if predictions["id"].tolist() != expected_ids or not predictions["fold"].astype(int).eq(fold).all():
        raise ValueError("candidate prediction ids/order or fold differ from the development mapping")
    runtime_audit = json.loads(
        (output_dir / "protocol_inputs/protocol_input_audit.json").read_text(encoding="utf-8")
    )
    if runtime_audit.get("sealed_rows_in_runtime_inputs") != 0:
        raise ValueError("scoped runtime audit does not prove that sealed rows were removed")
    selection = json.loads((output_dir / "seed_selection_audit.runtime.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "lora_holdout_report.json").read_text(encoding="utf-8"))
    if selection.get("seed") != seed or selection.get("outer_fold") != fold:
        raise ValueError("runtime seed selection audit does not match job identity")
    if selection.get("decision") != "GO" or report.get("holdout_fold") != fold:
        raise ValueError("runtime audit or parent report is not valid")
    if report.get("download_failures") != 0:
        raise ValueError("parent report contains image download failures")
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "outer_fold": fold,
        "prediction_rows": len(predictions),
        "predictions_sha256": sha256_file(output_dir / "lora_holdout_predictions.csv"),
        "adapter_sha256": sha256_file(output_dir / "adapter.zip"),
        "protocol_input_audit_sha256": sha256_file(output_dir / "protocol_inputs/protocol_input_audit.json"),
        "selection_audit_sha256": sha256_file(output_dir / "seed_selection_audit.runtime.json"),
        "sealed_rows_in_predictions": 0,
        "decision": "GO",
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    (output_dir / "seed_output_contract.runtime.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one independent semantic-v3 Qwen3.5 seed.")
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--fold", required=True, type=int, choices=FOLDS)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--vendor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed == 42:
        raise ValueError("seed 42 is supplied by experiment 600; experiment 602 runs only independent seeds")
    shared_protocol, shared_trainer = load_exp600_dependencies()
    runtime_input_paths(args.runtime_dir)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = shared_protocol.materialize_runtime_fold(
        runtime_dir=args.runtime_dir.resolve(),
        output_dir=output_dir / "protocol_inputs",
        component="original",
        outer_fold=args.fold,
    )
    environment = configure_environment(seed=args.seed, fold=args.fold, environment=dict(os.environ))
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
    parent = _load_exact_parent(shared_trainer.PARENT_RUNNERS["original"], shared_trainer.PARENT_SHA256["original"])
    _assert_parent_recipe(parent, seed=args.seed, fold=args.fold)
    _install_seed_selection_audit(parent, seed=args.seed, fold=args.fold, output_dir=output_dir)
    parent.main()
    validate_output_contract(output_dir=output_dir, seed=args.seed, fold=args.fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_gradient_control import (
    CONTROL_MODE,
    EFFECTIVE_BATCH_ROWS,
    EPOCHS,
    EXPECTED_RUNTIME_PACKAGES,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_UPDATES,
    EXPERIMENT_ID,
    GRADIENT_ACCUMULATION_PAIRS,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MICRO_BATCH_PAIRS,
    MICRO_BATCH_ROWS,
    MODEL_ID,
    MODEL_REVISION,
    NORM_CAP_RATIO,
    RANK_WEIGHT,
    SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    canonical_sha256,
    load_terminal_probe_selection,
    sha256_file,
    verify_diagnostic_row,
)

PARITY_FIELDS = (
    "schema_version",
    "experiment_id",
    "parent_experiment_id",
    "selector_experiment_id",
    "source_experiment_id",
    "outer_fold",
    "selected_candidate_mode",
    "changed_factor",
    "objective",
    "model_id",
    "model_revision",
    "model_tree_sha256",
    "model_tree_files",
    "initial_trainable_state_sha256",
    "pair_runtime_contract_sha256",
    "pair_runtime_acceptance_sha256",
    "transport_acceptance_sha256",
    "source_641_runtime_contract_sha256",
    "parent_code_bundle_sha256",
    "parent_code_revision",
    "parent_code_acceptance_sha256",
    "probe_code_bundle_sha256",
    "probe_code_revision",
    "probe_code_acceptance_sha256",
    "code_bundle_sha256",
    "code_revision",
    "vendor_zip_sha256",
    "vendor_bridge_sha256",
    "probe_scientific_decision",
    "probe_report_sha256",
    "probe_acceptance_sha256",
    "probe_report_file_sha256",
    "probe_acceptance_file_sha256",
    "ordered_pair_index_sha256",
    "seed",
    "epochs",
    "learning_rate",
    "optimizer",
    "scheduler",
    "micro_batch_pairs",
    "micro_batch_rows",
    "gradient_accumulation_pairs",
    "effective_batch_rows",
    "pairs",
    "train_rows",
    "optimizer_steps_executed",
    "hard_loss_weight",
    "candidate_rank_weight",
    "norm_cap_ratio",
    "gradient_clip_norm",
    "mode_independent_autograd_paths",
    "training_loss_mean",
    "gradient_diagnostics_schema_version",
    "validation_rows",
    "technical_smoke",
    "runtime_backend",
    "runtime_packages",
    "validation_labels_read",
    "sealed_rows_used",
    "public_used",
    "threshold",
    "threshold_tuned",
    "uses_27b_at_inference",
    "decision",
)
SHARED_DIAGNOSTIC_FIELDS = (
    "schema_version",
    "optimizer_step",
    "selected_candidate_mode",
    "hard_rank_dot",
    "hard_grad_norm",
    "rank_grad_norm",
    "weighted_rank_grad_norm",
    "hard_rank_cosine",
    "conflict",
    "projection_retention",
    "norm_cap_limit",
    "norm_cap_scale",
    "clip_max_norm",
)
ACCEPTANCE_FIELDS = {
    "schema_version",
    "experiment_id",
    "parent_experiment_id",
    "outer_fold",
    "mode",
    "selected_candidate_mode",
    "technical_smoke",
    "rows",
    "pairs",
    "optimizer_steps",
    "output_contract_sha256",
    "diagnostics_sha256",
    "predictions_sha256",
    "adapter_config_sha256",
    "adapter_model_sha256",
    "code_bundle_sha256",
    "code_revision",
    "probe_report_sha256",
    "probe_acceptance_sha256",
    "pair_runtime_contract_sha256",
    "pair_runtime_acceptance_sha256",
    "source_641_runtime_contract_sha256",
    "probe_code_bundle_sha256",
    "probe_code_revision",
    "probe_code_acceptance_sha256",
    "model_id",
    "model_revision",
    "exact_runtime_binding",
    "same_autograd_path_control",
    "deployable_4b_only",
    "validation_labels_read",
    "sealed_rows_used",
    "public_used",
    "decision",
    "acceptance_sha256",
}


def _self_hashed(path: Path, hash_field: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"paired smoke requires regular file: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    digest = body.pop(hash_field, None)
    if digest != canonical_sha256(body):
        raise ValueError(f"paired smoke self-hash mismatch: {path.name}")
    return value, str(digest)


def _load_arm(path: Path, expected_mode: str, selected_mode: str) -> dict[str, Any]:
    expected_inventory = {
        "output_contract.json",
        "gradient_diagnostics.jsonl",
        "predictions.jsonl",
        "acceptance.json",
        "adapter/README.md",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
    }
    observed = {
        member.relative_to(path).as_posix()
        for member in path.rglob("*")
        if member.is_file()
    }
    if not path.is_dir() or any(member.is_symlink() for member in path.rglob("*")):
        raise ValueError("paired smoke arm is not an isolated regular directory")
    if observed != expected_inventory:
        raise ValueError("paired smoke arm inventory mismatch")
    contract, contract_sha = _self_hashed(path / "output_contract.json", "contract_sha256")
    acceptance, acceptance_sha = _self_hashed(path / "acceptance.json", "acceptance_sha256")
    if set(acceptance) != ACCEPTANCE_FIELDS:
        raise ValueError("paired smoke arm acceptance schema mismatch")
    expected_identity = {
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": "686",
        "outer_fold": 3,
        "mode": expected_mode,
        "selected_candidate_mode": selected_mode,
        "technical_smoke": True,
        "rows": 2,
        "pairs": GRADIENT_ACCUMULATION_PAIRS,
        "optimizer_steps": 1,
        "output_contract_sha256": contract_sha,
        "code_bundle_sha256": contract.get("code_bundle_sha256"),
        "code_revision": contract.get("code_revision"),
        "probe_report_sha256": contract.get("probe_report_sha256"),
        "probe_acceptance_sha256": contract.get("probe_acceptance_sha256"),
        "pair_runtime_contract_sha256": contract.get("pair_runtime_contract_sha256"),
        "pair_runtime_acceptance_sha256": contract.get("pair_runtime_acceptance_sha256"),
        "source_641_runtime_contract_sha256": contract.get(
            "source_641_runtime_contract_sha256"
        ),
        "probe_code_bundle_sha256": contract.get("probe_code_bundle_sha256"),
        "probe_code_revision": contract.get("probe_code_revision"),
        "probe_code_acceptance_sha256": contract.get(
            "probe_code_acceptance_sha256"
        ),
        "model_id": contract.get("model_id"),
        "model_revision": contract.get("model_revision"),
        "exact_runtime_binding": True,
        "same_autograd_path_control": True,
        "deployable_4b_only": True,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "decision": "ACCEPT",
    }
    if any(acceptance.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("paired smoke arm acceptance/contract binding mismatch")
    artifact_bindings = {
        "diagnostics_sha256": path / "gradient_diagnostics.jsonl",
        "predictions_sha256": path / "predictions.jsonl",
        "adapter_config_sha256": path / "adapter/adapter_config.json",
        "adapter_model_sha256": path / "adapter/adapter_model.safetensors",
    }
    if any(
        acceptance.get(field) != sha256_file(artifact)
        for field, artifact in artifact_bindings.items()
    ):
        raise ValueError("paired smoke arm acceptance artifact binding mismatch")
    diagnostics = [
        json.loads(line)
        for line in (path / "gradient_diagnostics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    if len(diagnostics) != 1:
        raise ValueError("paired technical smoke must contain exactly one diagnostic")
    verify_diagnostic_row(
        diagnostics[0],
        optimizer_step=1,
        mode=expected_mode,
        selected_candidate_mode=selected_mode,
    )
    return {
        "contract": contract,
        "contract_sha256": contract_sha,
        "acceptance": acceptance,
        "acceptance_sha256": acceptance_sha,
        "acceptance_file_sha256": sha256_file(path / "acceptance.json"),
        "diagnostic": diagnostics[0],
    }


def _verify_code_acceptance(
    path: Path, *, expected_revision: str, expected_bundle_sha256: str
) -> tuple[dict[str, Any], str]:
    value, digest = _self_hashed(path, "acceptance_sha256")
    expected_fields = {
        "schema_version",
        "experiment_id",
        "parent_experiment_id",
        "selector_experiment_id",
        "scope",
        "git_revision",
        "bundle_sha256",
        "manifest_sha256",
        "files",
        "source_paths",
        "decision",
        "acceptance_sha256",
    }
    if set(value) != expected_fields:
        raise ValueError("paired smoke code acceptance schema mismatch")
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": "686",
        "selector_experiment_id": "687",
        "scope": "paired_technical_smoke",
        "git_revision": expected_revision,
        "bundle_sha256": expected_bundle_sha256,
        "decision": "ACCEPT_EXP688_CODE_BUNDLE",
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("paired smoke code acceptance mismatch")
    return value, digest


def verify(
    root: Path,
    *,
    probe_report: Path,
    probe_acceptance: Path,
    code_acceptance: Path,
) -> dict[str, Any]:
    selection = load_terminal_probe_selection(probe_report, probe_acceptance)
    selected_mode = selection["selected_candidate_mode"]
    if not root.is_dir() or root.is_symlink():
        raise ValueError("paired smoke output root is invalid")
    root_members = {path.name for path in root.iterdir()}
    if root_members not in (
        {"control", "candidate", "code_acceptance.json"},
        {"control", "candidate", "code_acceptance.json", "paired_acceptance.json"},
    ):
        raise ValueError("paired smoke root inventory mismatch")
    control = _load_arm(root / "control", CONTROL_MODE, selected_mode)
    candidate = _load_arm(root / "candidate", selected_mode, selected_mode)
    control_contract = control["contract"]
    candidate_contract = candidate["contract"]
    parity_mismatch = {
        field: {"control": control_contract.get(field), "candidate": candidate_contract.get(field)}
        for field in PARITY_FIELDS
        if control_contract.get(field) != candidate_contract.get(field)
    }
    if parity_mismatch:
        raise ValueError(f"paired technical-smoke parity mismatch: {parity_mismatch}")
    control_diagnostic = control["diagnostic"]
    candidate_diagnostic = candidate["diagnostic"]
    if any(
        control_diagnostic.get(field) != candidate_diagnostic.get(field)
        for field in SHARED_DIAGNOSTIC_FIELDS
    ):
        raise ValueError("paired smoke raw hard/rank gradient geometry differs")
    expected_common = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": "686",
        "selector_experiment_id": "687",
        "source_experiment_id": "641",
        "outer_fold": 3,
        "selected_candidate_mode": selected_mode,
        "changed_factor": "effective_batch_gradient_composition_only",
        "objective": "hard_primary_gradient_control_rank_distillation",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": {"name": "AdamW", "weight_decay": WEIGHT_DECAY},
        "scheduler": {
            "name": "linear_warmup_cosine_decay",
            "warmup_ratio": WARMUP_RATIO,
            "warmup_steps": max(1, int(EXPECTED_UPDATES * WARMUP_RATIO)),
        },
        "micro_batch_pairs": MICRO_BATCH_PAIRS,
        "micro_batch_rows": MICRO_BATCH_ROWS,
        "gradient_accumulation_pairs": GRADIENT_ACCUMULATION_PAIRS,
        "effective_batch_rows": EFFECTIVE_BATCH_ROWS,
        "pairs": GRADIENT_ACCUMULATION_PAIRS,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "optimizer_steps_executed": 1,
        "hard_loss_weight": 1.0,
        "candidate_rank_weight": RANK_WEIGHT,
        "norm_cap_ratio": NORM_CAP_RATIO,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "mode_independent_autograd_paths": {"hard": 1, "rank": 1},
        "technical_smoke": True,
        "runtime_backend": "legacy_eager",
        "runtime_packages": EXPECTED_RUNTIME_PACKAGES,
        "validation_rows": 2,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "threshold": 0.0,
        "threshold_tuned": False,
        "uses_27b_at_inference": False,
        "decision": "TECHNICAL_SMOKE_ONLY",
    }
    frozen_mismatch = {
        field: {"expected": value, "actual": control_contract.get(field)}
        for field, value in expected_common.items()
        if control_contract.get(field) != value
    }
    if frozen_mismatch:
        raise ValueError(f"paired smoke frozen common contract mismatch: {frozen_mismatch}")
    selector_bindings = {
        "pair_runtime_contract_sha256": selection["pair_runtime_contract_sha256"],
        "pair_runtime_acceptance_sha256": selection[
            "pair_runtime_acceptance_sha256"
        ],
        "source_641_runtime_contract_sha256": selection[
            "source_runtime_contract_sha256"
        ],
        "parent_code_bundle_sha256": selection["parent_code_bundle_sha256"],
        "parent_code_revision": selection["parent_code_revision"],
        "parent_code_acceptance_sha256": selection[
            "parent_code_acceptance_sha256"
        ],
        "probe_code_bundle_sha256": selection["probe_code_bundle_sha256"],
        "probe_code_revision": selection["probe_code_revision"],
        "probe_code_acceptance_sha256": selection[
            "probe_code_acceptance_sha256"
        ],
        "vendor_zip_sha256": selection["vendor_zip_sha256"],
        "vendor_bridge_sha256": selection["vendor_bridge_sha256"],
        "model_tree_sha256": selection["model_tree_sha256"],
        "initial_trainable_state_sha256": selection[
            "initial_trainable_state_sha256"
        ],
        "ordered_pair_index_sha256": selection["ordered_pair_index_sha256"],
        "probe_scientific_decision": selection["probe_scientific_decision"],
        "probe_report_sha256": selection["probe_report_sha256"],
        "probe_acceptance_sha256": selection["probe_acceptance_sha256"],
        "probe_report_file_sha256": selection["probe_report_file_sha256"],
        "probe_acceptance_file_sha256": selection[
            "probe_acceptance_file_sha256"
        ],
    }
    selector_mismatch = {
        field: {"selector": value, "arm": control_contract.get(field)}
        for field, value in selector_bindings.items()
        if control_contract.get(field) != value
    }
    if selector_mismatch:
        raise ValueError(f"paired smoke selector-lineage mismatch: {selector_mismatch}")
    if (
        control_contract.get("mode") != CONTROL_MODE
        or control_contract.get("applied_rank_weight") != 0.0
        or candidate_contract.get("mode") != selected_mode
        or candidate_contract.get("applied_rank_weight") != 0.5
    ):
        raise ValueError("paired smoke changed-factor contract mismatch")
    if (
        float(control_diagnostic.get("selected_rank_grad_norm", -1.0)) != 0.0
        or float(candidate_diagnostic.get("selected_rank_grad_norm", 0.0)) <= 0.0
        or control_contract.get("final_trainable_state_sha256")
        == candidate_contract.get("final_trainable_state_sha256")
    ):
        raise ValueError("paired smoke applied updates do not differ")
    _, code_acceptance_sha = _verify_code_acceptance(
        code_acceptance,
        expected_revision=str(control_contract["code_revision"]),
        expected_bundle_sha256=str(control_contract["code_bundle_sha256"]),
    )
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": "686",
        "selector_experiment_id": "687",
        "stage": "paired_technical_smoke",
        "outer_fold": control_contract["outer_fold"],
        "control_mode": CONTROL_MODE,
        "candidate_mode": selected_mode,
        "optimizer_steps_per_arm": 1,
        "sequential_single_h100": True,
        "same_initial_trainable_state": True,
        "same_raw_hard_rank_gradients": True,
        "applied_update_differs": True,
        "only_gradient_composition_differs": True,
        "parity_fields": list(PARITY_FIELDS),
        "shared_diagnostic_fields": list(SHARED_DIAGNOSTIC_FIELDS),
        "initial_trainable_state_sha256": control_contract[
            "initial_trainable_state_sha256"
        ],
        "ordered_pair_index_sha256": control_contract["ordered_pair_index_sha256"],
        "code_bundle_sha256": control_contract["code_bundle_sha256"],
        "code_revision": control_contract["code_revision"],
        "code_acceptance_sha256": code_acceptance_sha,
        "probe_report_sha256": selection["probe_report_sha256"],
        "probe_acceptance_sha256": selection["probe_acceptance_sha256"],
        "control_output_contract_sha256": control["contract_sha256"],
        "candidate_output_contract_sha256": candidate["contract_sha256"],
        "control_acceptance_sha256": control["acceptance_sha256"],
        "candidate_acceptance_sha256": candidate["acceptance_sha256"],
        "control_acceptance_file_sha256": control["acceptance_file_sha256"],
        "candidate_acceptance_file_sha256": candidate["acceptance_file_sha256"],
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "deployable": False,
        "decision": "ACCEPT_PAIRED_TECHNICAL_SMOKE_PARITY",
    }
    result["acceptance_sha256"] = canonical_sha256(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--probe-acceptance", type=Path, required=True)
    parser.add_argument("--code-acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_output = args.root / "paired_acceptance.json"
    if args.output.resolve() != expected_output.resolve():
        raise ValueError("paired acceptance must be root/paired_acceptance.json")
    value = verify(
        args.root,
        probe_report=args.probe_report,
        probe_acceptance=args.probe_acceptance,
        code_acceptance=args.code_acceptance,
    )
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != payload:
            raise FileExistsError("refusing to overwrite different paired acceptance")
    else:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")

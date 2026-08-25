from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "686_qwen35_4b_additive_rank_kd"
PROBE = HERE.parent / "687_qwen35_4b_gradient_conflict_probe"
SHARED = HERE.parent / "645_qwen_scale_2x3_gate"
for dependency_path in (SHARED, PARENT, PROBE, HERE):
    if str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))

import train_lora as control
import train_pair_fold as parent
import verify_probe_artifact as probe_verifier
from build_pair_runtime import canonical_sha256, occurrence_key, sha256_bytes
from gradient_metrics import CHECKPOINT_STEPS, DIAGNOSTIC_EFFECTIVE_BATCHES, pcgrad_gate

EXPERIMENT_ID = "688"
PARENT_EXPERIMENT_ID = "686"
SELECTOR_EXPERIMENT_ID = "687"
SOURCE_EXPERIMENT_ID = "641"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SEED = 42
EPOCHS = 1
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
GRADIENT_CLIP_NORM = 1.0
RANK_WEIGHT = 0.5
NORM_CAP_RATIO = 0.5
MICRO_BATCH_PAIRS = 1
MICRO_BATCH_ROWS = 2
GRADIENT_ACCUMULATION_PAIRS = 8
EFFECTIVE_BATCH_ROWS = 16
EXPECTED_PAIRS = 5440
EXPECTED_UPDATES = 680
EXPECTED_TRAIN_ROWS = 2280
CONTROL_MODE = "paired_hard_control"
PCGRAD_MODE = "asymmetric_hard_primary_pcgrad"
NORM_CAP_MODE = "hard_anchored_norm_cap"
CANDIDATE_MODES = (PCGRAD_MODE, NORM_CAP_MODE)
TRAINING_MODES = (CONTROL_MODE, *CANDIDATE_MODES)
CANDIDATE_BY_PROBE_DECISION = {
    "OPEN_ASYMMETRIC_PCGRAD_SCREEN": PCGRAD_MODE,
    "ROUTE_MAGNITUDE_CONTROL": NORM_CAP_MODE,
}
EXPECTED_RUNTIME_PACKAGES = {
    "torch": "2.10.0+cu128",
    "transformers": "5.14.1",
    "peft": "0.20.0",
}
DIAGNOSTIC_SCHEMA_VERSION = 1
DIAGNOSTIC_FIELDS = {
    "schema_version",
    "optimizer_step",
    "training_mode",
    "selected_candidate_mode",
    "hard_rank_dot",
    "hard_grad_norm",
    "rank_grad_norm",
    "weighted_rank_grad_norm",
    "hard_rank_cosine",
    "conflict",
    "projection_applied",
    "projection_retention",
    "norm_cap_limit",
    "norm_cap_scale",
    "norm_cap_applied",
    "selected_rank_retention",
    "selected_rank_grad_norm",
    "combined_grad_norm_preclip",
    "clip_grad_norm_return",
    "clip_max_norm",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _sum_product(left: Any, right: Any) -> float:
    value = (left * right).sum()
    return float(value.item() if hasattr(value, "item") else value)


def _copy_tensor(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value.clone()
    return value.copy()


def _gradient_geometry(
    hard_gradients: Sequence[Any], rank_gradients: Sequence[Any]
) -> dict[str, float | bool]:
    if not hard_gradients or len(hard_gradients) != len(rank_gradients):
        raise ValueError("hard/rank gradient coverage mismatch")
    hard_sq = 0.0
    rank_sq = 0.0
    dot = 0.0
    for hard, rank in zip(hard_gradients, rank_gradients, strict=True):
        if getattr(hard, "shape", None) != getattr(rank, "shape", None):
            raise ValueError("hard/rank gradient shape mismatch")
        hard_sq += _sum_product(hard, hard)
        rank_sq += _sum_product(rank, rank)
        dot += _sum_product(hard, rank)
    if not all(math.isfinite(value) for value in (hard_sq, rank_sq, dot)):
        raise FloatingPointError("non-finite gradient geometry")
    if hard_sq <= 0.0 or rank_sq <= 0.0:
        raise FloatingPointError("hard and rank gradient norms must be positive")
    hard_norm = math.sqrt(hard_sq)
    rank_norm = math.sqrt(rank_sq)
    cosine = max(-1.0, min(1.0, dot / (hard_norm * rank_norm)))
    conflict = dot < 0.0
    projected_rank_sq = (
        max(0.0, rank_sq - dot * dot / hard_sq) if conflict else rank_sq
    )
    projection_retention = math.sqrt(projected_rank_sq / rank_sq)
    weighted_rank_norm = RANK_WEIGHT * rank_norm
    cap_limit = NORM_CAP_RATIO * hard_norm
    cap_scale = min(1.0, cap_limit / weighted_rank_norm)
    return {
        "hard_sq": hard_sq,
        "rank_sq": rank_sq,
        "hard_rank_dot": dot,
        "hard_grad_norm": hard_norm,
        "rank_grad_norm": rank_norm,
        "weighted_rank_grad_norm": weighted_rank_norm,
        "hard_rank_cosine": cosine,
        "conflict": conflict,
        "projection_retention": projection_retention,
        "norm_cap_limit": cap_limit,
        "norm_cap_scale": cap_scale,
    }


def combine_gradients(
    hard_gradients: Sequence[Any],
    rank_gradients: Sequence[Any],
    *,
    mode: str,
    selected_candidate_mode: str,
) -> tuple[list[Any], dict[str, float | bool | str | int]]:
    """Apply the frozen effective-batch rule to already accumulated gradients."""
    validate_training_mode(mode, selected_candidate_mode)
    geometry = _gradient_geometry(hard_gradients, rank_gradients)
    conflict = bool(geometry["conflict"])
    dot = float(geometry["hard_rank_dot"])
    hard_sq = float(geometry["hard_sq"])
    projection_factor = dot / hard_sq if conflict else 0.0
    projected_rank = [
        rank - projection_factor * hard
        for hard, rank in zip(hard_gradients, rank_gradients, strict=True)
    ]

    if mode == CONTROL_MODE:
        combined = [_copy_tensor(hard) for hard in hard_gradients]
        selected_rank_retention = 0.0
    elif mode == PCGRAD_MODE:
        combined = [
            hard + RANK_WEIGHT * rank
            for hard, rank in zip(hard_gradients, projected_rank, strict=True)
        ]
        selected_rank_retention = float(geometry["projection_retention"])
    elif mode == NORM_CAP_MODE:
        cap_scale = float(geometry["norm_cap_scale"])
        combined = [
            hard + RANK_WEIGHT * cap_scale * rank
            for hard, rank in zip(hard_gradients, rank_gradients, strict=True)
        ]
        selected_rank_retention = cap_scale
    else:  # pragma: no cover - validate_training_mode rejects this first.
        raise ValueError("unknown gradient-control mode")

    combined_sq = sum(_sum_product(value, value) for value in combined)
    if not math.isfinite(combined_sq) or combined_sq <= 0.0:
        raise FloatingPointError("combined gradient norm must be finite and positive")
    combined_norm = math.sqrt(combined_sq)
    selected_rank_norm = (
        RANK_WEIGHT
        * float(geometry["rank_grad_norm"])
        * selected_rank_retention
    )
    diagnostic: dict[str, float | bool | str | int] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "optimizer_step": 0,
        "training_mode": mode,
        "selected_candidate_mode": selected_candidate_mode,
        "hard_rank_dot": dot,
        "hard_grad_norm": float(geometry["hard_grad_norm"]),
        "rank_grad_norm": float(geometry["rank_grad_norm"]),
        "weighted_rank_grad_norm": float(geometry["weighted_rank_grad_norm"]),
        "hard_rank_cosine": float(geometry["hard_rank_cosine"]),
        "conflict": conflict,
        "projection_applied": mode == PCGRAD_MODE and conflict,
        "projection_retention": float(geometry["projection_retention"]),
        "norm_cap_limit": float(geometry["norm_cap_limit"]),
        "norm_cap_scale": float(geometry["norm_cap_scale"]),
        "norm_cap_applied": mode == NORM_CAP_MODE
        and float(geometry["norm_cap_scale"]) < 1.0,
        "selected_rank_retention": selected_rank_retention,
        "selected_rank_grad_norm": selected_rank_norm,
        "combined_grad_norm_preclip": combined_norm,
        "clip_grad_norm_return": combined_norm,
        "clip_max_norm": GRADIENT_CLIP_NORM,
    }
    return combined, diagnostic


def validate_training_mode(mode: str, selected_candidate_mode: str) -> None:
    if selected_candidate_mode not in CANDIDATE_MODES:
        raise ValueError("terminal experiment 687 did not select a valid candidate")
    if mode not in {CONTROL_MODE, selected_candidate_mode}:
        raise ValueError("mode is neither paired control nor terminal-selected candidate")


def _assert_close(actual: float, expected: float, field: str) -> None:
    if not math.isclose(actual, expected, rel_tol=2e-5, abs_tol=2e-7):
        raise ValueError(f"gradient diagnostic formula mismatch: {field}")


def verify_diagnostic_row(
    row: dict[str, Any],
    *,
    optimizer_step: int,
    mode: str,
    selected_candidate_mode: str,
) -> None:
    if set(row) != DIAGNOSTIC_FIELDS:
        raise ValueError("gradient diagnostic schema mismatch")
    expected_exact = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "optimizer_step": optimizer_step,
        "training_mode": mode,
        "selected_candidate_mode": selected_candidate_mode,
        "clip_max_norm": GRADIENT_CLIP_NORM,
    }
    if any(row.get(key) != value for key, value in expected_exact.items()):
        raise ValueError("gradient diagnostic identity mismatch")
    for field in ("conflict", "projection_applied", "norm_cap_applied"):
        if not isinstance(row[field], bool):
            raise TypeError("gradient diagnostic boolean field mismatch")
    numeric_fields = DIAGNOSTIC_FIELDS - set(expected_exact) - {
        "training_mode",
        "selected_candidate_mode",
        "conflict",
        "projection_applied",
        "norm_cap_applied",
    }
    if any(
        isinstance(row[field], bool)
        or not isinstance(row[field], (int, float))
        or not math.isfinite(float(row[field]))
        for field in numeric_fields
    ):
        raise TypeError("gradient diagnostic numeric field mismatch")
    hard_norm = float(row["hard_grad_norm"])
    rank_norm = float(row["rank_grad_norm"])
    dot = float(row["hard_rank_dot"])
    if hard_norm <= 0.0 or rank_norm <= 0.0:
        raise ValueError("gradient diagnostic norms must be positive")
    conflict = dot < 0.0
    cosine = max(-1.0, min(1.0, dot / (hard_norm * rank_norm)))
    if conflict:
        projected_sq = max(0.0, rank_norm**2 - dot**2 / hard_norm**2)
        projection_retention = math.sqrt(projected_sq) / rank_norm
        projected_dot = 0.0
    else:
        projection_retention = 1.0
        projected_dot = dot
    weighted_rank_norm = RANK_WEIGHT * rank_norm
    cap_limit = NORM_CAP_RATIO * hard_norm
    cap_scale = min(1.0, cap_limit / weighted_rank_norm)
    if mode == CONTROL_MODE:
        selected_retention = 0.0
        combined_sq = hard_norm**2
    elif mode == PCGRAD_MODE:
        selected_retention = projection_retention
        selected_rank_sq = (RANK_WEIGHT * rank_norm * selected_retention) ** 2
        combined_sq = hard_norm**2 + selected_rank_sq + 2 * RANK_WEIGHT * projected_dot
    else:
        selected_retention = cap_scale
        combined_sq = (
            hard_norm**2
            + (RANK_WEIGHT * cap_scale * rank_norm) ** 2
            + 2 * RANK_WEIGHT * cap_scale * dot
        )
    _assert_close(float(row["hard_rank_cosine"]), cosine, "hard_rank_cosine")
    _assert_close(
        float(row["weighted_rank_grad_norm"]),
        weighted_rank_norm,
        "weighted_rank_grad_norm",
    )
    _assert_close(
        float(row["projection_retention"]),
        projection_retention,
        "projection_retention",
    )
    _assert_close(float(row["norm_cap_limit"]), cap_limit, "norm_cap_limit")
    _assert_close(float(row["norm_cap_scale"]), cap_scale, "norm_cap_scale")
    _assert_close(
        float(row["selected_rank_retention"]),
        selected_retention,
        "selected_rank_retention",
    )
    _assert_close(
        float(row["selected_rank_grad_norm"]),
        RANK_WEIGHT * rank_norm * selected_retention,
        "selected_rank_grad_norm",
    )
    _assert_close(
        float(row["combined_grad_norm_preclip"]),
        math.sqrt(max(0.0, combined_sq)),
        "combined_grad_norm_preclip",
    )
    if not math.isclose(
        float(row["clip_grad_norm_return"]),
        float(row["combined_grad_norm_preclip"]),
        rel_tol=5e-3,
        abs_tol=5e-5,
    ):
        raise ValueError("gradient diagnostic formula mismatch: clip_grad_norm_return")
    expected_flags = {
        "conflict": conflict,
        "projection_applied": mode == PCGRAD_MODE and conflict,
        "norm_cap_applied": mode == NORM_CAP_MODE and cap_scale < 1.0,
    }
    if any(row[field] is not value for field, value in expected_flags.items()):
        raise ValueError("gradient diagnostic transformation flag mismatch")


def load_terminal_probe_selection(
    report_path: Path, acceptance_path: Path
) -> dict[str, Any]:
    if (
        report_path.parent.resolve() != acceptance_path.parent.resolve()
        or report_path.name != "gradient_conflict_report.json"
        or acceptance_path.name != "acceptance.json"
    ):
        raise ValueError("terminal experiment-687 files must be the canonical sibling pair")
    expected_inventory = {"gradient_conflict_report.json", "acceptance.json"}
    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or not acceptance_path.is_file()
        or acceptance_path.is_symlink()
        or {path.name for path in report_path.parent.iterdir()} != expected_inventory
    ):
        raise ValueError("terminal experiment-687 artifact inventory mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_body = dict(report)
    report_sha = report_body.pop("report_sha256", None)
    if report_sha != canonical_sha256(report_body):
        raise ValueError("experiment-687 report self-hash mismatch")
    expected_report = {
        "schema_version": 1,
        "experiment_id": SELECTOR_EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "outer_fold": 3,
        "objective": "outer_validation_label_free_gradient_conflict_probe",
        "pairs": EXPECTED_PAIRS,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "optimizer_steps_executed": EXPECTED_UPDATES,
        "rank_loss_weight": RANK_WEIGHT,
        "hard_loss_weight": 1.0,
        "model_revision": MODEL_REVISION,
        "model_id": MODEL_ID,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "runtime_backend": "legacy_eager",
        "runtime_packages": EXPECTED_RUNTIME_PACKAGES,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "diagnostic_effective_batches": DIAGNOSTIC_EFFECTIVE_BATCHES,
        "inline_technical_preflight": True,
        "diagnostic_rng_noninterference": True,
        "outer_validation_transport_checksum_verified": True,
        "outer_validation_rows_consumed_by_probe": 0,
        "outer_validation_labels_read": 0,
        "outer_quality_metrics_computed": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "submission_artifact": False,
    }
    mismatch = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected_report.items()
        if report.get(key) != value
    }
    if mismatch:
        raise ValueError(f"experiment-687 report contract mismatch: {mismatch}")
    rows = report.get("measurements")
    if not isinstance(rows, list) or len(rows) != len(CHECKPOINT_STEPS) * DIAGNOSTIC_EFFECTIVE_BATCHES:
        raise ValueError("experiment-687 measurement coverage mismatch")
    for row in rows:
        probe_verifier.verify_measurement_schema(row, grouped=False)
    probe_verifier.verify_overall_coverage(rows)
    group_rows = report.get("group_measurements")
    groups = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if not isinstance(group_rows, list) or len(group_rows) != len(rows) * len(groups):
        raise ValueError("experiment-687 group-measurement coverage mismatch")
    for row in group_rows:
        probe_verifier.verify_measurement_schema(row, grouped=True)
    group_coverage = {
        (int(row["checkpoint_step"]), int(row["batch_index"]), str(row["group"]))
        for row in group_rows
    }
    expected_group_coverage = {
        (step, batch_index, group)
        for step in CHECKPOINT_STEPS
        for batch_index in range(DIAGNOSTIC_EFFECTIVE_BATCHES)
        for group in groups
    }
    if group_coverage != expected_group_coverage or len(group_coverage) != len(
        group_rows
    ):
        raise ValueError("experiment-687 group-measurement keys mismatch")
    checkpoint_rows = {
        step: [row for row in rows if int(row["checkpoint_step"]) == step]
        for step in CHECKPOINT_STEPS
    }
    recomputed = pcgrad_gate(checkpoint_rows)
    if recomputed != report.get("pcgrad_gate") or recomputed["decision"] != report.get(
        "decision"
    ):
        raise ValueError("experiment-687 scientific decision does not reproduce")
    for field in (
        "pair_runtime_contract_sha256",
        "pair_runtime_acceptance_sha256",
        "source_runtime_contract_sha256",
        "sanitizer_acceptance_sha256",
        "sanitized_validation_sha256",
        "parent_code_bundle_sha256",
        "parent_code_revision",
        "parent_code_acceptance_sha256",
        "probe_code_bundle_sha256",
        "probe_code_revision",
        "probe_code_acceptance_sha256",
        "vendor_zip_sha256",
        "vendor_bridge_sha256",
        "model_tree_sha256",
        "initial_trainable_state_sha256",
        "final_trainable_state_sha256",
        "diagnostic_pair_indices_sha256",
        "ordered_pair_index_sha256",
    ):
        value = report.get(field)
        if not _is_lower_hex(value, 40) and not _is_lower_hex(value, 64):
            raise ValueError(f"invalid experiment-687 provenance field: {field}")

    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance_body = dict(acceptance)
    acceptance_sha = acceptance_body.pop("acceptance_sha256", None)
    if acceptance_sha != canonical_sha256(acceptance_body):
        raise ValueError("experiment-687 acceptance self-hash mismatch")
    expected_acceptance = {
        "schema_version": 1,
        "experiment_id": SELECTOR_EXPERIMENT_ID,
        "decision": "ACCEPT_GRADIENT_CONFLICT_PROBE",
        "scientific_decision": report["decision"],
        "report_sha256": report_sha,
        "measurements": len(rows),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "diagnostic_effective_batches": DIAGNOSTIC_EFFECTIVE_BATCHES,
        "diagnostic_rng_noninterference": True,
        "outer_validation_transport_checksum_verified": True,
        "outer_validation_rows_consumed_by_probe": 0,
        "outer_validation_labels_read": 0,
        "outer_quality_metrics_computed": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "submission_artifact": False,
    }
    if acceptance_body != expected_acceptance:
        raise ValueError("experiment-687 acceptance contract mismatch")
    selected = CANDIDATE_BY_PROBE_DECISION.get(str(report["decision"]))
    if selected is None:
        raise ValueError("terminal experiment 687 does not open an exp688 candidate")
    return {
        "selected_candidate_mode": selected,
        "probe_scientific_decision": report["decision"],
        "probe_report_sha256": report_sha,
        "probe_acceptance_sha256": acceptance_sha,
        "probe_report_file_sha256": sha256_file(report_path),
        "probe_acceptance_file_sha256": sha256_file(acceptance_path),
        "pair_runtime_contract_sha256": report["pair_runtime_contract_sha256"],
        "pair_runtime_acceptance_sha256": report[
            "pair_runtime_acceptance_sha256"
        ],
        "source_runtime_contract_sha256": report["source_runtime_contract_sha256"],
        "parent_code_bundle_sha256": report["parent_code_bundle_sha256"],
        "parent_code_revision": report["parent_code_revision"],
        "parent_code_acceptance_sha256": report[
            "parent_code_acceptance_sha256"
        ],
        "probe_code_bundle_sha256": report["probe_code_bundle_sha256"],
        "probe_code_revision": report["probe_code_revision"],
        "probe_code_acceptance_sha256": report[
            "probe_code_acceptance_sha256"
        ],
        "vendor_zip_sha256": report["vendor_zip_sha256"],
        "vendor_bridge_sha256": report["vendor_bridge_sha256"],
        "model_tree_sha256": report["model_tree_sha256"],
        "initial_trainable_state_sha256": report[
            "initial_trainable_state_sha256"
        ],
        "ordered_pair_index_sha256": report["ordered_pair_index_sha256"],
    }


def load_probe_code_acceptance(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("probe-code acceptance must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "experiment_id",
        "git_revision",
        "bundle_sha256",
        "manifest_sha256",
        "files",
        "decision",
        "acceptance_sha256",
    }
    if set(value) != expected_fields:
        raise ValueError("probe-code acceptance schema mismatch")
    body = dict(value)
    digest = body.pop("acceptance_sha256", None)
    if digest != canonical_sha256(body):
        raise ValueError("probe-code acceptance self-hash mismatch")
    if (
        value.get("schema_version") != 1
        or value.get("experiment_id") != SELECTOR_EXPERIMENT_ID
        or value.get("decision") != "ACCEPT_PROBE_CODE_BUNDLE"
        or not _is_lower_hex(value.get("git_revision"), 40)
        or not _is_lower_hex(value.get("bundle_sha256"), 64)
        or not _is_lower_hex(value.get("manifest_sha256"), 64)
        or not isinstance(value.get("files"), int)
        or value["files"] <= 0
    ):
        raise ValueError("probe-code acceptance identity mismatch")
    return value


def pair_losses(
    model: Any,
    processor: Any,
    positive: SimpleNamespace,
    negative: SimpleNamespace,
    images: list[Any],
    zero_token: int,
    one_token: int,
    pair_target: float,
) -> tuple[Any, Any]:
    import torch
    from torch.nn import functional

    rows = [positive, negative]
    conversations = [
        control.messages(row, image, prompt_text=control.base_prompt(row))
        for row, image in zip(rows, images, strict=True)
    ]
    batch = control._processor_batch(
        processor, conversations, add_generation_prompt=True
    ).to(model.device)
    scores = control._last_logits(model, batch, zero_token, one_token).float()
    hard = functional.binary_cross_entropy_with_logits(
        scores,
        torch.tensor([1.0, 0.0], dtype=torch.float32, device=model.device),
    )
    rank = functional.binary_cross_entropy_with_logits(
        scores[0] - scores[1],
        torch.tensor(float(pair_target), dtype=torch.float32, device=model.device),
    )
    return hard, rank


def accumulate_effective_batch_gradients(
    *,
    model: Any,
    processor: Any,
    named_trainable: list[tuple[str, Any]],
    pair_indices: Sequence[int],
    pairs: list[dict[str, Any]],
    pair_lookup: dict[tuple[Any, ...], dict[str, Any]],
    images_dir: Path,
    zero_token: int,
    one_token: int,
) -> tuple[list[Any], list[Any], float, float]:
    """Accumulate h and unweighted r with one mode-independent autograd path."""
    import torch

    if len(pair_indices) != GRADIENT_ACCUMULATION_PAIRS:
        raise ValueError("effective batch must contain exactly eight pairs")
    parameters = [parameter for _, parameter in named_trainable]
    hard_accumulators = [
        torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters
    ]
    rank_accumulators = [
        torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters
    ]
    hard_loss_sum = 0.0
    rank_loss_sum = 0.0
    model.zero_grad(set_to_none=True)
    for pair_index in pair_indices:
        pair = pairs[pair_index]
        positive = pair_lookup[tuple(pair["positive_key"])]
        negative = pair_lookup[tuple(pair["negative_key"])]
        rows = [SimpleNamespace(**positive), SimpleNamespace(**negative)]
        images = [
            control.open_image(images_dir, positive),
            control.open_image(images_dir, negative),
        ]
        try:
            hard_loss, rank_loss = pair_losses(
                model,
                processor,
                rows[0],
                rows[1],
                images,
                zero_token,
                one_token,
                float(pair["pair_target"]),
            )
            if not torch.isfinite(hard_loss).item() or not torch.isfinite(rank_loss).item():
                raise FloatingPointError("non-finite hard or rank loss")
            hard_loss_sum += float(hard_loss.detach().item())
            rank_loss_sum += float(rank_loss.detach().item())
            (hard_loss / GRADIENT_ACCUMULATION_PAIRS).backward(retain_graph=True)
            for accumulator, parameter in zip(
                hard_accumulators, parameters, strict=True
            ):
                if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                    raise FloatingPointError("missing or non-finite hard gradient")
                accumulator.add_(parameter.grad.detach().float())
            model.zero_grad(set_to_none=True)
            (rank_loss / GRADIENT_ACCUMULATION_PAIRS).backward()
            for accumulator, parameter in zip(
                rank_accumulators, parameters, strict=True
            ):
                if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                    raise FloatingPointError("missing or non-finite rank gradient")
                accumulator.add_(parameter.grad.detach().float())
            model.zero_grad(set_to_none=True)
        finally:
            for image in images:
                image.close()
    return hard_accumulators, rank_accumulators, hard_loss_sum, rank_loss_sum


def summarize_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty gradient diagnostics")
    numeric = (
        "hard_rank_dot",
        "hard_rank_cosine",
        "hard_grad_norm",
        "rank_grad_norm",
        "weighted_rank_grad_norm",
        "projection_retention",
        "norm_cap_scale",
        "selected_rank_retention",
        "combined_grad_norm_preclip",
        "clip_grad_norm_return",
    )
    return {
        "optimizer_steps": len(rows),
        "conflicts": sum(bool(row["conflict"]) for row in rows),
        "conflict_rate": sum(bool(row["conflict"]) for row in rows) / len(rows),
        "projection_applied_steps": sum(bool(row["projection_applied"]) for row in rows),
        "norm_cap_applied_steps": sum(bool(row["norm_cap_applied"]) for row in rows),
        "means": {
            field: sum(float(row[field]) for row in rows) / len(rows) for field in numeric
        },
        "minimum_projection_retention": min(
            float(row["projection_retention"]) for row in rows
        ),
        "minimum_norm_cap_scale": min(float(row["norm_cap_scale"]) for row in rows),
        "maximum_combined_grad_norm_preclip": max(
            float(row["combined_grad_norm_preclip"]) for row in rows
        ),
    }


def _load_frozen_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.fold != 3:
        raise ValueError("exp688 selector lineage currently authorizes fold3 only")
    if args.runtime_backend != "legacy_eager" or args.micro_batch_size_override != 2:
        raise ValueError("exp688 requires the frozen exp686 legacy-eager micro2 runtime")
    if args.model_revision != MODEL_REVISION:
        raise ValueError("model revision differs from the frozen exp686 contract")
    if not _is_lower_hex(args.code_revision, 40):
        raise ValueError("code revision must be a full lowercase git SHA")
    if not _is_lower_hex(args.expected_code_bundle_sha256, 64):
        raise ValueError("expected code bundle SHA-256 is invalid")
    if args.code_bundle.is_symlink() or not args.code_bundle.is_file():
        raise ValueError("exp688 code bundle must be a regular non-symlink file")
    code_bundle_sha = sha256_file(args.code_bundle)
    if code_bundle_sha != args.expected_code_bundle_sha256:
        raise ValueError("exp688 code bundle SHA-256 mismatch")
    selection = load_terminal_probe_selection(args.probe_report, args.probe_acceptance)
    validate_training_mode(args.mode, selection["selected_candidate_mode"])
    train, validation, pairs, pair_acceptance, source_audit = parent.load_inputs(
        args.runtime_dir, args.pair_runtime, args.fold
    )
    if len(train) != EXPECTED_TRAIN_ROWS or len(pairs) != EXPECTED_PAIRS:
        raise ValueError("exp686 row or pair cardinality drifted")
    transport = parent.load_transport_acceptance(
        args.transport_acceptance,
        fold=args.fold,
        pair_acceptance=pair_acceptance,
        source_audit=source_audit,
    )
    parent_code = parent.load_code_acceptance(args.parent_code_acceptance)
    probe_code = load_probe_code_acceptance(args.probe_code_acceptance)
    vendor = parent.load_vendor_acceptance(args.vendor_acceptance, args.vendor_archive)
    lineage = {
        "pair_runtime_contract_sha256": pair_acceptance["runtime_contract_sha256"],
        "pair_runtime_acceptance_sha256": pair_acceptance["acceptance_sha256"],
        "source_runtime_contract_sha256": source_audit["contract_sha256"],
        "parent_code_bundle_sha256": parent_code["bundle_sha256"],
        "parent_code_revision": parent_code["git_revision"],
        "parent_code_acceptance_sha256": parent_code["acceptance_sha256"],
        "probe_code_bundle_sha256": probe_code["bundle_sha256"],
        "probe_code_revision": probe_code["git_revision"],
        "probe_code_acceptance_sha256": probe_code["acceptance_sha256"],
        "vendor_zip_sha256": vendor["vendor_zip_sha256"],
        "vendor_bridge_sha256": vendor["bridge_sha256"],
    }
    mismatch = {
        key: {"selector": selection.get(key), "current": value}
        for key, value in lineage.items()
        if selection.get(key) != value
    }
    if mismatch:
        raise ValueError(f"experiment-687/current-input lineage mismatch: {mismatch}")
    return {
        "train": train,
        "validation": validation,
        "pairs": pairs,
        "pair_acceptance": pair_acceptance,
        "source_audit": source_audit,
        "transport": transport,
        "parent_code": parent_code,
        "probe_code": probe_code,
        "vendor": vendor,
        "selection": selection,
        "code_bundle_sha256": code_bundle_sha,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    frozen = _load_frozen_inputs(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = frozen["train"]
    validation = frozen["validation"]
    pairs = frozen["pairs"]
    if args.technical_smoke:
        validation = validation[:2]
    if len(pairs) % GRADIENT_ACCUMULATION_PAIRS:
        raise ValueError("pair count must divide the frozen accumulation exactly")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    if not args.vendor.is_dir():
        raise FileNotFoundError("vendored PEFT 0.20.0 directory is missing")
    sys.path.insert(0, str(args.vendor.resolve()))
    import peft

    if peft.__version__ != "0.20.0":
        raise ValueError("exact vendored PEFT 0.20.0 is required")
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    spec = control.CELL_SPECS[SOURCE_EXPERIMENT_ID]
    if spec.model_id != MODEL_ID or spec.model_revision != MODEL_REVISION:
        raise ValueError("source model spec differs from the frozen exp686 contract")
    model_tree_sha, model_tree_files = parent.model_tree_sha256(args.model_root)
    processor = AutoProcessor.from_pretrained(
        args.model_root.resolve(), local_files_only=True, trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1 or zero == one:
        raise ValueError("0/1 must be distinct atomic tokens")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_root.resolve(),
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=True,
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_trainable:
        raise ValueError("model has no trainable LoRA parameters")
    parameters = [parameter for _, parameter in named_trainable]
    initial_state_sha = parent.trainable_state_sha256(model)
    model_lineage = {
        "model_tree_sha256": model_tree_sha,
        "initial_trainable_state_sha256": initial_state_sha,
    }
    mismatch = {
        key: {"selector": frozen["selection"].get(key), "current": value}
        for key, value in model_lineage.items()
        if frozen["selection"].get(key) != value
    }
    if mismatch:
        raise ValueError(f"experiment-687/current model-init lineage mismatch: {mismatch}")
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    updates = 1 if args.technical_smoke else len(pairs) // GRADIENT_ACCUMULATION_PAIRS
    schedule_updates = EXPECTED_UPDATES
    warmup_steps = max(1, int(schedule_updates * WARMUP_RATIO))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, schedule_updates - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    pair_indices = list(range(len(pairs)))
    random.Random(SEED).shuffle(pair_indices)
    ordered_pair_index_sha = hashlib.sha256(
        json.dumps(pair_indices, separators=(",", ":")).encode()
    ).hexdigest()
    if ordered_pair_index_sha != frozen["selection"]["ordered_pair_index_sha256"]:
        raise ValueError("experiment-687/current pair-order lineage mismatch")
    if args.technical_smoke:
        pair_indices = pair_indices[:GRADIENT_ACCUMULATION_PAIRS]
    pair_lookup = {occurrence_key(row): row for row in train}
    if len(pair_lookup) != len(train):
        raise ValueError("duplicate occurrence key in parent train runtime")

    optimizer.zero_grad(set_to_none=True)
    model.train()
    diagnostics: list[dict[str, Any]] = []
    hard_loss_sum = 0.0
    rank_loss_sum = 0.0
    started = time.monotonic()
    for batch_start in range(0, len(pair_indices), GRADIENT_ACCUMULATION_PAIRS):
        batch_indices = pair_indices[
            batch_start : batch_start + GRADIENT_ACCUMULATION_PAIRS
        ]
        hard_gradients, rank_gradients, hard_losses, rank_losses = (
            accumulate_effective_batch_gradients(
                model=model,
                processor=processor,
                named_trainable=named_trainable,
                pair_indices=batch_indices,
                pairs=pairs,
                pair_lookup=pair_lookup,
                images_dir=args.images,
                zero_token=zero[0],
                one_token=one[0],
            )
        )
        hard_loss_sum += hard_losses
        rank_loss_sum += rank_losses
        combined, diagnostic = combine_gradients(
            hard_gradients,
            rank_gradients,
            mode=args.mode,
            selected_candidate_mode=frozen["selection"]["selected_candidate_mode"],
        )
        step = len(diagnostics) + 1
        diagnostic["optimizer_step"] = step
        for parameter, gradient in zip(parameters, combined, strict=True):
            parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype)
        preclip_norm = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        if not torch.isfinite(preclip_norm).item():
            raise FloatingPointError("non-finite combined gradient norm")
        diagnostic["clip_grad_norm_return"] = float(preclip_norm.item())
        verify_diagnostic_row(
            diagnostic,
            optimizer_step=step,
            mode=args.mode,
            selected_candidate_mode=frozen["selection"]["selected_candidate_mode"],
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        diagnostics.append(diagnostic)
        if step == 1:
            print(json.dumps({"phase": "inline_technical_preflight_pass"}), flush=True)
        if step % 50 == 0 or step == updates:
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "phase": "training_progress",
                        "mode": args.mode,
                        "optimizer_steps_done": step,
                        "optimizer_steps_total": updates,
                        "pairs_done": step * GRADIENT_ACCUMULATION_PAIRS,
                        "pairs_per_second": (
                            step * GRADIENT_ACCUMULATION_PAIRS / max(elapsed, 1e-9)
                        ),
                    }
                ),
                flush=True,
            )
    if len(diagnostics) != updates:
        raise RuntimeError("optimizer-step count drifted")

    model.eval()
    model.config.use_cache = True
    predictions = control.predict_class_only(
        model, processor, validation, args.images, spec, zero[0], one[0]
    )
    predictions_path = args.output_dir / "predictions.jsonl"
    control.write_jsonl(predictions_path, predictions)
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    final_state_sha = parent.trainable_state_sha256(model)
    reload_max_abs_score_delta: float | None = None
    reload_prediction_mismatches: int | None = None
    if args.technical_smoke:
        base_model = model.unload()
        reloaded = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
        reloaded.eval()
        reloaded.config.use_cache = True
        reloaded_predictions = control.predict_class_only(
            reloaded, processor, validation, args.images, spec, zero[0], one[0]
        )
        reload_max_abs_score_delta = max(
            abs(float(before["score"]) - float(after["score"]))
            for before, after in zip(predictions, reloaded_predictions, strict=True)
        )
        reload_prediction_mismatches = sum(
            int(before["prediction"]) != int(after["prediction"])
            for before, after in zip(predictions, reloaded_predictions, strict=True)
        )
        if reload_max_abs_score_delta > 1e-5 or reload_prediction_mismatches:
            raise RuntimeError("adapter save/reload prediction parity failed")

    diagnostics_path = args.output_dir / "gradient_diagnostics.jsonl"
    diagnostics_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in diagnostics
    ).encode()
    diagnostics_path.write_bytes(diagnostics_payload)
    artifact_paths = {
        "gradient_diagnostics.jsonl": diagnostics_path,
        "predictions.jsonl": predictions_path,
        "adapter/README.md": adapter_dir / "README.md",
        "adapter/adapter_config.json": adapter_dir / "adapter_config.json",
        "adapter/adapter_model.safetensors": adapter_dir / "adapter_model.safetensors",
    }
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "selector_experiment_id": SELECTOR_EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "outer_fold": args.fold,
        "mode": args.mode,
        "selected_candidate_mode": frozen["selection"]["selected_candidate_mode"],
        "changed_factor": "effective_batch_gradient_composition_only",
        "objective": "hard_primary_gradient_control_rank_distillation",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_tree_sha256": model_tree_sha,
        "model_tree_files": model_tree_files,
        "initial_trainable_state_sha256": initial_state_sha,
        "final_trainable_state_sha256": final_state_sha,
        "pair_runtime_contract_sha256": frozen["pair_acceptance"][
            "runtime_contract_sha256"
        ],
        "pair_runtime_acceptance_sha256": frozen["pair_acceptance"][
            "acceptance_sha256"
        ],
        "transport_acceptance_sha256": frozen["transport"][
            "transport_acceptance_sha256"
        ],
        "source_641_runtime_contract_sha256": frozen["source_audit"][
            "contract_sha256"
        ],
        "parent_code_bundle_sha256": frozen["parent_code"]["bundle_sha256"],
        "parent_code_revision": frozen["parent_code"]["git_revision"],
        "parent_code_acceptance_sha256": frozen["parent_code"]["acceptance_sha256"],
        "probe_code_bundle_sha256": frozen["probe_code"]["bundle_sha256"],
        "probe_code_revision": frozen["probe_code"]["git_revision"],
        "probe_code_acceptance_sha256": frozen["probe_code"]["acceptance_sha256"],
        "code_bundle_sha256": frozen["code_bundle_sha256"],
        "code_revision": args.code_revision,
        "vendor_zip_sha256": frozen["vendor"]["vendor_zip_sha256"],
        "vendor_bridge_sha256": frozen["vendor"]["bridge_sha256"],
        "probe_scientific_decision": frozen["selection"]["probe_scientific_decision"],
        "probe_report_sha256": frozen["selection"]["probe_report_sha256"],
        "probe_acceptance_sha256": frozen["selection"]["probe_acceptance_sha256"],
        "probe_report_file_sha256": frozen["selection"]["probe_report_file_sha256"],
        "probe_acceptance_file_sha256": frozen["selection"][
            "probe_acceptance_file_sha256"
        ],
        "ordered_pair_index_sha256": ordered_pair_index_sha,
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": {"name": "AdamW", "weight_decay": WEIGHT_DECAY},
        "scheduler": {
            "name": "linear_warmup_cosine_decay",
            "warmup_ratio": WARMUP_RATIO,
            "warmup_steps": warmup_steps,
        },
        "micro_batch_pairs": MICRO_BATCH_PAIRS,
        "micro_batch_rows": MICRO_BATCH_ROWS,
        "gradient_accumulation_pairs": GRADIENT_ACCUMULATION_PAIRS,
        "effective_batch_rows": EFFECTIVE_BATCH_ROWS,
        "pairs": len(pair_indices),
        "train_rows": len(train),
        "optimizer_steps_executed": updates,
        "hard_loss_weight": 1.0,
        "candidate_rank_weight": RANK_WEIGHT,
        "applied_rank_weight": 0.0 if args.mode == CONTROL_MODE else RANK_WEIGHT,
        "norm_cap_ratio": NORM_CAP_RATIO,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "mode_independent_autograd_paths": {"hard": 1, "rank": 1},
        "training_loss_mean": {
            "hard": hard_loss_sum / len(pair_indices),
            "rank_unweighted": rank_loss_sum / len(pair_indices),
        },
        "gradient_diagnostics_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "gradient_diagnostics": summarize_diagnostics(diagnostics),
        "validation_rows": len(validation),
        "technical_smoke": bool(args.technical_smoke),
        "runtime_backend": "legacy_eager",
        "runtime_packages": {
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
            "peft": peft.__version__,
        },
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "runtime_minutes": (time.monotonic() - started) / 60,
        "reload_max_abs_score_delta": reload_max_abs_score_delta,
        "reload_prediction_mismatches": reload_prediction_mismatches,
        "artifacts": {
            relative: sha256_file(path) for relative, path in artifact_paths.items()
        },
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "threshold": 0.0,
        "threshold_tuned": False,
        "uses_27b_at_inference": False,
        "decision": "TECHNICAL_SMOKE_ONLY" if args.technical_smoke else "GO_EVALUATE",
    }
    report["contract_sha256"] = canonical_sha256(report)
    (args.output_dir / "output_contract.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "phase": "complete",
                "mode": args.mode,
                "contract_sha256": report["contract_sha256"],
            }
        ),
        flush=True,
    )
    return report


def _prediction_schema() -> set[str]:
    return {
        "category",
        "concept",
        "fold",
        "format_valid",
        "generated_verdict",
        "global_index",
        "grounded",
        "grounding_source",
        "id",
        "image_index",
        "model_id",
        "model_revision",
        "objective",
        "prediction",
        "preprocessing_version",
        "prompt_version",
        "quote",
        "raw_generation",
        "region_index",
        "score",
        "target_order",
    }


def verify_training_artifact(args: argparse.Namespace) -> dict[str, Any]:
    frozen = _load_frozen_inputs(args)
    required = {
        "output_contract.json": args.output_dir / "output_contract.json",
        "gradient_diagnostics.jsonl": args.output_dir / "gradient_diagnostics.jsonl",
        "predictions.jsonl": args.output_dir / "predictions.jsonl",
        "adapter/README.md": args.output_dir / "adapter/README.md",
        "adapter/adapter_config.json": args.output_dir / "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors": (
            args.output_dir / "adapter/adapter_model.safetensors"
        ),
    }
    if not args.output_dir.is_dir() or any(not path.is_file() for path in required.values()):
        raise ValueError("exp688 output directory schema is incomplete")
    observed: set[str] = set()
    for path in args.output_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError("exp688 output contains a symlink")
        if path.is_file():
            observed.add(path.relative_to(args.output_dir).as_posix())
        elif not path.is_dir():
            raise ValueError("exp688 output contains a special filesystem member")
    if observed not in (set(required), {*required, "acceptance.json"}):
        raise ValueError(f"exp688 output member set mismatch: {sorted(observed)}")

    contract = json.loads(required["output_contract.json"].read_text(encoding="utf-8"))
    contract_body = dict(contract)
    contract_sha = contract_body.pop("contract_sha256", None)
    if contract_sha != canonical_sha256(contract_body):
        raise ValueError("exp688 output contract self-hash mismatch")
    expected_pairs = GRADIENT_ACCUMULATION_PAIRS if args.technical_smoke else EXPECTED_PAIRS
    expected_updates = expected_pairs // GRADIENT_ACCUMULATION_PAIRS
    expected_validation_rows = 2 if args.technical_smoke else len(frozen["validation"])
    pair_indices = list(range(EXPECTED_PAIRS))
    random.Random(SEED).shuffle(pair_indices)
    expected_order_sha = hashlib.sha256(
        json.dumps(pair_indices, separators=(",", ":")).encode()
    ).hexdigest()
    expected = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "selector_experiment_id": SELECTOR_EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "outer_fold": args.fold,
        "mode": args.mode,
        "selected_candidate_mode": frozen["selection"]["selected_candidate_mode"],
        "changed_factor": "effective_batch_gradient_composition_only",
        "objective": "hard_primary_gradient_control_rank_distillation",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_tree_sha256": frozen["selection"]["model_tree_sha256"],
        "initial_trainable_state_sha256": frozen["selection"][
            "initial_trainable_state_sha256"
        ],
        "pair_runtime_contract_sha256": frozen["pair_acceptance"][
            "runtime_contract_sha256"
        ],
        "pair_runtime_acceptance_sha256": frozen["pair_acceptance"][
            "acceptance_sha256"
        ],
        "transport_acceptance_sha256": frozen["transport"][
            "transport_acceptance_sha256"
        ],
        "source_641_runtime_contract_sha256": frozen["source_audit"][
            "contract_sha256"
        ],
        "parent_code_bundle_sha256": frozen["parent_code"]["bundle_sha256"],
        "parent_code_revision": frozen["parent_code"]["git_revision"],
        "parent_code_acceptance_sha256": frozen["parent_code"]["acceptance_sha256"],
        "probe_code_bundle_sha256": frozen["probe_code"]["bundle_sha256"],
        "probe_code_revision": frozen["probe_code"]["git_revision"],
        "probe_code_acceptance_sha256": frozen["probe_code"]["acceptance_sha256"],
        "code_bundle_sha256": frozen["code_bundle_sha256"],
        "code_revision": args.code_revision,
        "vendor_zip_sha256": frozen["vendor"]["vendor_zip_sha256"],
        "vendor_bridge_sha256": frozen["vendor"]["bridge_sha256"],
        "probe_scientific_decision": frozen["selection"]["probe_scientific_decision"],
        "probe_report_sha256": frozen["selection"]["probe_report_sha256"],
        "probe_acceptance_sha256": frozen["selection"]["probe_acceptance_sha256"],
        "probe_report_file_sha256": frozen["selection"]["probe_report_file_sha256"],
        "probe_acceptance_file_sha256": frozen["selection"][
            "probe_acceptance_file_sha256"
        ],
        "ordered_pair_index_sha256": expected_order_sha,
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
        "pairs": expected_pairs,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "optimizer_steps_executed": expected_updates,
        "hard_loss_weight": 1.0,
        "candidate_rank_weight": RANK_WEIGHT,
        "applied_rank_weight": 0.0 if args.mode == CONTROL_MODE else RANK_WEIGHT,
        "norm_cap_ratio": NORM_CAP_RATIO,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "mode_independent_autograd_paths": {"hard": 1, "rank": 1},
        "gradient_diagnostics_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "validation_rows": expected_validation_rows,
        "technical_smoke": args.technical_smoke,
        "runtime_backend": "legacy_eager",
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "threshold": 0.0,
        "threshold_tuned": False,
        "uses_27b_at_inference": False,
        "decision": "TECHNICAL_SMOKE_ONLY" if args.technical_smoke else "GO_EVALUATE",
    }
    mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"exp688 output contract mismatch: {mismatch}")
    for field in (
        "model_tree_sha256",
        "initial_trainable_state_sha256",
        "final_trainable_state_sha256",
    ):
        if not _is_lower_hex(contract.get(field), 64):
            raise ValueError(f"invalid exp688 provenance field: {field}")
    if not isinstance(contract.get("model_tree_files"), int) or contract["model_tree_files"] <= 0:
        raise ValueError("invalid model-tree file count")
    if contract.get("runtime_packages") != EXPECTED_RUNTIME_PACKAGES:
        raise ValueError("runtime packages differ from frozen exp686")
    if not isinstance(contract.get("peak_cuda_bytes"), int) or contract["peak_cuda_bytes"] >= 75 * 1024**3:
        raise ValueError("peak CUDA memory exceeds one-H100 gate")
    losses = contract.get("training_loss_mean")
    if (
        not isinstance(losses, dict)
        or set(losses) != {"hard", "rank_unweighted"}
        or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in losses.values())
    ):
        raise ValueError("exp688 loss diagnostics are invalid")

    diagnostic_payload = required["gradient_diagnostics.jsonl"].read_bytes()
    diagnostic_rows = [
        json.loads(line) for line in diagnostic_payload.decode("utf-8").splitlines()
    ]
    if len(diagnostic_rows) != expected_updates:
        raise ValueError("gradient diagnostic row count mismatch")
    for step, row in enumerate(diagnostic_rows, start=1):
        verify_diagnostic_row(
            row,
            optimizer_step=step,
            mode=args.mode,
            selected_candidate_mode=frozen["selection"]["selected_candidate_mode"],
        )
    if contract.get("gradient_diagnostics") != summarize_diagnostics(diagnostic_rows):
        raise ValueError("gradient diagnostic summary does not reproduce")

    predictions_payload = required["predictions.jsonl"].read_bytes()
    predictions = [json.loads(line) for line in predictions_payload.decode().splitlines()]
    if len(predictions) != expected_validation_rows or any(
        set(row) != _prediction_schema() for row in predictions
    ):
        raise ValueError("exp688 prediction schema or row count mismatch")
    validation_rows = frozen["validation"][:expected_validation_rows]
    binding = ("global_index", "id", "fold", "category")
    if any(
        tuple(prediction[field] for field in binding)
        != tuple(runtime_row[field] for field in binding)
        for prediction, runtime_row in zip(predictions, validation_rows, strict=True)
    ):
        raise ValueError("exp688 prediction/runtime binding mismatch")
    if any(
        row["category"] != parent.FLAMMABLE
        or not math.isfinite(float(row["score"]))
        or int(row["prediction"]) != int(float(row["score"]) >= 0.0)
        for row in predictions
    ):
        raise ValueError("exp688 prediction value or threshold mismatch")

    adapter_config_payload = required["adapter/adapter_config.json"].read_bytes()
    adapter_config = json.loads(adapter_config_payload)
    if adapter_config.get("base_model_name_or_path") not in {MODEL_ID, "/hf_models"}:
        raise ValueError("adapter base model differs from deployable Qwen3.5-4B")
    if set(adapter_config.get("target_modules", [])) != {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }:
        raise ValueError("LoRA target modules drifted")
    if (
        int(adapter_config.get("r", -1)) != 16
        or int(adapter_config.get("lora_alpha", -1)) != 32
        or float(adapter_config.get("lora_dropout", -1)) != 0.05
        or adapter_config.get("use_rslora") is not True
        or adapter_config.get("bias") != "none"
        or adapter_config.get("task_type") != "CAUSAL_LM"
        or adapter_config.get("inference_mode") is not True
    ):
        raise ValueError("LoRA configuration drifted")
    artifact_payloads = {
        "gradient_diagnostics.jsonl": diagnostic_payload,
        "predictions.jsonl": predictions_payload,
        "adapter/README.md": required["adapter/README.md"].read_bytes(),
        "adapter/adapter_config.json": adapter_config_payload,
        "adapter/adapter_model.safetensors": required[
            "adapter/adapter_model.safetensors"
        ].read_bytes(),
    }
    if contract.get("artifacts") != {
        relative: sha256_bytes(payload) for relative, payload in artifact_payloads.items()
    }:
        raise ValueError("exp688 inner artifact checksum mismatch")
    if args.technical_smoke and (
        contract.get("reload_prediction_mismatches") != 0
        or float(contract.get("reload_max_abs_score_delta", math.inf)) > 1e-5
    ):
        raise ValueError("exp688 technical-smoke save/reload parity failed")

    acceptance = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "outer_fold": args.fold,
        "mode": args.mode,
        "selected_candidate_mode": frozen["selection"]["selected_candidate_mode"],
        "technical_smoke": args.technical_smoke,
        "rows": len(predictions),
        "pairs": expected_pairs,
        "optimizer_steps": expected_updates,
        "output_contract_sha256": contract_sha,
        "diagnostics_sha256": sha256_bytes(diagnostic_payload),
        "predictions_sha256": sha256_bytes(predictions_payload),
        "adapter_config_sha256": sha256_bytes(adapter_config_payload),
        "adapter_model_sha256": sha256_bytes(
            artifact_payloads["adapter/adapter_model.safetensors"]
        ),
        "code_bundle_sha256": frozen["code_bundle_sha256"],
        "code_revision": args.code_revision,
        "probe_report_sha256": frozen["selection"]["probe_report_sha256"],
        "probe_acceptance_sha256": frozen["selection"]["probe_acceptance_sha256"],
        "pair_runtime_contract_sha256": frozen["pair_acceptance"][
            "runtime_contract_sha256"
        ],
        "pair_runtime_acceptance_sha256": frozen["pair_acceptance"][
            "acceptance_sha256"
        ],
        "source_641_runtime_contract_sha256": frozen["source_audit"][
            "contract_sha256"
        ],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "exact_runtime_binding": True,
        "same_autograd_path_control": True,
        "deployable_4b_only": True,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "decision": "ACCEPT",
    }
    acceptance["acceptance_sha256"] = canonical_sha256(acceptance)
    existing = args.output_dir / "acceptance.json"
    payload = json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if existing.exists() and existing.read_text(encoding="utf-8") != payload:
        raise ValueError("existing exp688 acceptance differs from replay")
    return acceptance


def parser() -> argparse.ArgumentParser:
    result = control.parser_for(SOURCE_EXPERIMENT_ID)
    result.description = (
        "Train experiment 688 with terminal-selected hard-primary gradient control."
    )
    result.add_argument("--pair-runtime", type=Path, required=True)
    result.add_argument("--transport-acceptance", type=Path, required=True)
    result.add_argument("--parent-code-acceptance", type=Path, required=True)
    result.add_argument("--probe-code-acceptance", type=Path, required=True)
    result.add_argument("--probe-report", type=Path, required=True)
    result.add_argument("--probe-acceptance", type=Path, required=True)
    result.add_argument("--code-bundle", type=Path, required=True)
    result.add_argument("--expected-code-bundle-sha256", required=True)
    result.add_argument("--code-revision", required=True)
    result.add_argument("--vendor-acceptance", type=Path, required=True)
    result.add_argument("--vendor-archive", type=Path, required=True)
    result.add_argument("--mode", choices=TRAINING_MODES, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())

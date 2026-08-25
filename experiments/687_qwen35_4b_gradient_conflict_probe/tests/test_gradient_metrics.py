from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pytest
from build_remote_compute_preset import build as build_preset
from gradient_metrics import (
    CHECKPOINT_STEPS,
    DIAGNOSTIC_EFFECTIVE_BATCHES,
    gradient_metrics,
    pcgrad_gate,
    wilson_interval,
)
from sanitize_probe_source import canonical_sha256, sanitize
from verify_probe_artifact import verify_measurement_schema, verify_overall_coverage


def test_orthogonal_gradients_preserve_rank_signal() -> None:
    value = gradient_metrics(hard_sq=4.0, rank_sq=9.0, dot=0.0)
    assert value["cosine"] == pytest.approx(0.0)
    assert value["weighted_norm_ratio"] == pytest.approx(0.75)
    assert value["hard_cancellation"] == 0.0
    assert value["projection_retention"] == pytest.approx(1.0)


def test_opposing_gradients_report_cancellation_and_projection() -> None:
    value = gradient_metrics(hard_sq=4.0, rank_sq=4.0, dot=-2.0)
    assert value["conflict"] is True
    assert value["cosine"] == pytest.approx(-0.5)
    assert value["hard_cancellation"] == pytest.approx(0.25)
    assert value["projection_retention"] == pytest.approx(math.sqrt(0.75))


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(20, 80)
    assert low < 0.25 < high


def _row(*, conflict: bool, cancellation: float = 0.1, retention: float = 0.8):
    return {
        "cosine": -0.3 if conflict else 0.3,
        "weighted_norm_ratio": 0.4,
        "hard_cancellation": cancellation if conflict else 0.0,
        "combined_hard_alignment": 0.9,
        "projection_retention": retention,
        "hard_grad_norm": 1.0,
        "rank_grad_norm": 0.8,
        "conflict": conflict,
    }


def test_pcgrad_gate_opens_only_for_stable_conflict() -> None:
    rows = {
        step: [_row(conflict=index < 8) for index in range(DIAGNOSTIC_EFFECTIVE_BATCHES)]
        for step in CHECKPOINT_STEPS
    }
    assert pcgrad_gate(rows)["decision"] == "OPEN_ASYMMETRIC_PCGRAD_SCREEN"


def test_pcgrad_gate_rejects_low_conflict() -> None:
    rows = {
        step: [_row(conflict=False) for _ in range(DIAGNOSTIC_EFFECTIVE_BATCHES)]
        for step in CHECKPOINT_STEPS
    }
    assert pcgrad_gate(rows)["decision"] == "REJECT_PCGRAD_LOW_CONFLICT"


def test_pcgrad_gate_does_not_hide_low_conflict_retention_with_nonconflicts() -> None:
    rows = {
        step: [
            _row(conflict=index < 4, retention=0.0 if index < 4 else 1.0)
            for index in range(DIAGNOSTIC_EFFECTIVE_BATCHES)
        ]
        for step in CHECKPOINT_STEPS
    }
    result = pcgrad_gate(rows)
    assert result["aggregate"]["projection_retention"]["median"] == 1.0
    assert result["aggregate"]["conflicting_projection_retention_median"] == 0.0
    assert result["decision"] == "REJECT_PCGRAD_LOW_RETAINED_RANK_SIGNAL"


def test_checkpoint_coverage_is_fail_closed() -> None:
    rows = {
        step: [_row(conflict=True) for _ in range(DIAGNOSTIC_EFFECTIVE_BATCHES)]
        for step in CHECKPOINT_STEPS[:-1]
    }
    with pytest.raises(ValueError, match="checkpoint coverage"):
        pcgrad_gate(rows)


def _write_self_hashed(path: Path, value: dict, field: str) -> None:
    value[field] = canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_sanitizer_binds_full_validation_and_all_source_contracts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    pair = tmp_path / "pair"
    source.mkdir()
    pair.mkdir()
    validation = source / "validation.jsonl"
    validation.write_text('{"id":"x"}\n', encoding="utf-8")
    sha = __import__("hashlib").sha256(validation.read_bytes()).hexdigest()
    source_audit = {
        "outer_fold": 3,
        "output_sha256": {"validation.jsonl": sha},
    }
    _write_self_hashed(
        source / "runtime_audit.json",
        source_audit,
        "contract_sha256",
    )
    _write_self_hashed(
        pair / "runtime_audit.json",
        {
            "outer_fold": 3,
            "source_641_runtime_contract_sha256": source_audit["contract_sha256"],
            "derived_680_output_sha256": {"validation.jsonl": "1" * 64},
        },
        "contract_sha256",
    )
    transport = tmp_path / "transport.json"
    _write_self_hashed(
        transport,
        {
            "accepted_files": {
                "source_runtime/validation.jsonl": {"sha256": sha}
            }
        },
        "transport_acceptance_sha256",
    )
    result = sanitize(source, pair, transport)
    assert result["outer_validation_transport_checksum_verified"] is True
    assert result["validation_sha256"] == sha
    assert result["derived_filtered_validation_sha256"] == "1" * 64
    assert not validation.exists()


def test_sanitizer_rejects_transport_validation_checksum_disagreement(tmp_path: Path) -> None:
    source = tmp_path / "source"
    pair = tmp_path / "pair"
    source.mkdir()
    pair.mkdir()
    validation = source / "validation.jsonl"
    validation.write_text('{"id":"x"}\n', encoding="utf-8")
    sha = __import__("hashlib").sha256(validation.read_bytes()).hexdigest()
    source_audit = {
        "outer_fold": 3,
        "output_sha256": {"validation.jsonl": sha},
    }
    _write_self_hashed(
        source / "runtime_audit.json",
        source_audit,
        "contract_sha256",
    )
    _write_self_hashed(
        pair / "runtime_audit.json",
        {
            "outer_fold": 3,
            "source_641_runtime_contract_sha256": source_audit["contract_sha256"],
            "derived_680_output_sha256": {"validation.jsonl": "1" * 64},
        },
        "contract_sha256",
    )
    transport = tmp_path / "transport.json"
    _write_self_hashed(
        transport,
        {
            "accepted_files": {
                "source_runtime/validation.jsonl": {"sha256": "0" * 64}
            }
        },
        "transport_acceptance_sha256",
    )
    with pytest.raises(ValueError, match="differs from frozen"):
        sanitize(source, pair, transport)
    assert validation.exists()


def test_sanitizer_rejects_pair_source_contract_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    pair = tmp_path / "pair"
    source.mkdir()
    pair.mkdir()
    validation = source / "validation.jsonl"
    validation.write_text('{"id":"x"}\n', encoding="utf-8")
    sha = __import__("hashlib").sha256(validation.read_bytes()).hexdigest()
    _write_self_hashed(
        source / "runtime_audit.json",
        {"outer_fold": 3, "output_sha256": {"validation.jsonl": sha}},
        "contract_sha256",
    )
    _write_self_hashed(
        pair / "runtime_audit.json",
        {
            "outer_fold": 3,
            "source_641_runtime_contract_sha256": "0" * 64,
            "derived_680_output_sha256": {"validation.jsonl": "1" * 64},
        },
        "contract_sha256",
    )
    transport = tmp_path / "transport.json"
    _write_self_hashed(
        transport,
        {"accepted_files": {"source_runtime/validation.jsonl": {"sha256": sha}}},
        "transport_acceptance_sha256",
    )
    with pytest.raises(ValueError, match="not derived"):
        sanitize(source, pair, transport)
    assert validation.exists()


def test_sanitizer_rejects_filtered_checksum_equal_to_full_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    pair = tmp_path / "pair"
    source.mkdir()
    pair.mkdir()
    validation = source / "validation.jsonl"
    validation.write_text('{"id":"x"}\n', encoding="utf-8")
    sha = __import__("hashlib").sha256(validation.read_bytes()).hexdigest()
    source_audit = {
        "outer_fold": 3,
        "output_sha256": {"validation.jsonl": sha},
    }
    _write_self_hashed(
        source / "runtime_audit.json", source_audit, "contract_sha256"
    )
    _write_self_hashed(
        pair / "runtime_audit.json",
        {
            "outer_fold": 3,
            "source_641_runtime_contract_sha256": source_audit["contract_sha256"],
            "derived_680_output_sha256": {"validation.jsonl": sha},
        },
        "contract_sha256",
    )
    transport = tmp_path / "transport.json"
    _write_self_hashed(
        transport,
        {"accepted_files": {"source_runtime/validation.jsonl": {"sha256": sha}}},
        "transport_acceptance_sha256",
    )
    with pytest.raises(ValueError, match="filtered validation checksum equals"):
        sanitize(source, pair, transport)
    assert validation.exists()


def test_measurement_schema_rejects_extra_fields_and_bool_indices() -> None:
    row = {
        "checkpoint_step": 0,
        "batch_index": 0,
        "conflict": True,
        "hard_loss": 0.2,
        "rank_loss": 0.3,
        "hard_grad_norm": 1.0,
        "rank_grad_norm": 1.0,
        "cosine": -0.2,
        "weighted_norm_ratio": 0.5,
        "hard_cancellation": 0.1,
        "combined_hard_alignment": 0.9,
        "projection_retention": 0.8,
    }
    verify_measurement_schema(row, grouped=False)
    with pytest.raises(ValueError, match="schema"):
        verify_measurement_schema({**row, "extra": 1.0}, grouped=False)
    with pytest.raises(TypeError, match="key types"):
        verify_measurement_schema({**row, "batch_index": False}, grouped=False)


def test_overall_coverage_rejects_duplicate_and_missing_batch() -> None:
    rows = [
        {"checkpoint_step": step, "batch_index": batch}
        for step in CHECKPOINT_STEPS
        for batch in range(DIAGNOSTIC_EFFECTIVE_BATCHES)
    ]
    verify_overall_coverage(rows)
    invalid = [*rows[:-1], dict(rows[0])]
    with pytest.raises(ValueError, match="overall-measurement keys"):
        verify_overall_coverage(invalid)


def test_preset_rejects_shell_unsafe_bundle_basename_before_reading_inputs(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        output=tmp_path / "raw.yml",
        clean_output=None,
        overrides_output=None,
        probe_bundle_file="bad;name.tar.gz",
        probe_bundle_sha256="a" * 64,
        probe_code_revision="b" * 40,
        probe_bundle_src="/d.strizhakov/ecup/experiments/687/code/safe",
        output_dst="/d.strizhakov/ecup/experiments/687/probe/fold3/safe",
    )
    with pytest.raises(ValueError, match="shell-safe"):
        build_preset(args)

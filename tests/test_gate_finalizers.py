from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "research/finalize_bad_regulatory_gate.py"
)
SPEC = importlib.util.spec_from_file_location("finalize_bad_regulatory_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PRIOR_PROTOCOL = MODULE.PRIOR_PROTOCOL
REQUIRED_BOOLEAN_GATES = MODULE.REQUIRED_BOOLEAN_GATES
finalize_bad_regulatory_gate = MODULE.finalize_bad_regulatory_gate


def residual_fixture() -> dict[str, object]:
    acceptance = {key: True for key in REQUIRED_BOOLEAN_GATES}
    acceptance.update({"repeat_mean_delta": 0.004, "repeat_fold_wins": 12})
    return {
        "experiment_id": "320",
        "evaluation_version": "component_transfer_gate_v4",
        "acceptance": acceptance,
        "accepted_before_downstream_priors_runtime_schema_and_final_refit": True,
        "topologies": {
            "historical": {
                "baseline_category_f1": {
                    "БАД": 0.95,
                    "Легковоспламеняющиеся": 0.89,
                },
                "candidate_category_f1": {
                    "БАД": 0.96,
                    "Легковоспламеняющиеся": 0.89,
                },
                "baseline_macro_f1": 0.92,
                "candidate_macro_f1": 0.925,
                "delta_macro_f1": 0.005,
            },
            "repeat_0": {},
            "repeat_1": {},
            "repeat_2": {},
        },
    }


def priors_fixture() -> dict[str, object]:
    return {
        "protocol": PRIOR_PROTOCOL,
        "candidate": "exp320",
        "baseline_before_priors": {
            "bad_f1": 0.95,
            "flammable_f1": 0.89,
            "macro_f1": 0.92,
        },
        "candidate_before_priors": {
            "bad_f1": 0.96,
            "flammable_f1": 0.89,
            "macro_f1": 0.925,
        },
        "delta_before_priors": {
            "bad_f1": 0.01,
            "flammable_f1": 0.0,
            "macro_f1": 0.005,
        },
        "change_survival": {"survival_rate": 0.9},
        "delta_after_priors": {"macro_f1": 0.004},
    }


def run_finalizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    residual: dict[str, object],
    priors: dict[str, object],
) -> dict[str, object]:
    residual_path = tmp_path / "residual.json"
    priors_path = tmp_path / "priors.json"
    output_path = tmp_path / "final.json"
    residual_path.write_text(json.dumps(residual), encoding="utf-8")
    priors_path.write_text(json.dumps(priors), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_bad_regulatory_gate.py",
            "--residual-audit",
            str(residual_path),
            "--prior-replay",
            str(priors_path),
            "--output",
            str(output_path),
        ],
    )
    finalize_bad_regulatory_gate()
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_final_gate_accepts_only_semantically_bound_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = run_finalizer(
        monkeypatch, tmp_path, residual_fixture(), priors_fixture()
    )
    assert result["accepted_for_full_refit"] is True
    assert result["production_ready"] is False
    assert result["evaluation_version"] == "component_transfer_gate_v4"


def test_final_gate_rejects_replaced_required_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    residual = residual_fixture()
    acceptance = residual["acceptance"]
    assert isinstance(acceptance, dict)
    del acceptance["sports_delta_at_least_0_01"]
    acceptance["unrelated_replacement_gate"] = True
    with pytest.raises(ValueError, match="exact frozen v4 gate set"):
        run_finalizer(monkeypatch, tmp_path, residual, priors_fixture())


def test_final_gate_rejects_prior_replay_for_another_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    priors = copy.deepcopy(priors_fixture())
    priors["candidate"] = "exp260"
    with pytest.raises(ValueError, match="candidate must be exp320"):
        run_finalizer(monkeypatch, tmp_path, residual_fixture(), priors)


def test_final_gate_rejects_prior_scores_not_matching_residual(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    priors = copy.deepcopy(priors_fixture())
    candidate = priors["candidate_before_priors"]
    assert isinstance(candidate, dict)
    candidate["macro_f1"] = 0.999
    with pytest.raises(ValueError, match="does not match residual predictions"):
        run_finalizer(monkeypatch, tmp_path, residual_fixture(), priors)

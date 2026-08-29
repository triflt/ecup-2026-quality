from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/570_qwen35_weighted_grounded_auxiliary"
PARENT = ROOT / "experiments/520_qwen35_grounded_auxiliary_sft"
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(PARENT))

from loss_contract import (
    CONTRACT_VERSION,
    EVIDENCE_TOKEN_WEIGHT,
    VERDICT_TOKEN_WEIGHT,
    RuntimeLossAudit,
    build_loss_weights,
    exact_unit_weight_null_control,
    install_weighted_forward,
    weighted_causal_cross_entropy,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exp570_weights_only_the_first_atomic_verdict_at_one() -> None:
    labels = torch.tensor([[-100, 7, 4, 5], [-100, -100, 8, 6]])
    weights = build_loss_weights(labels, [7, 8])

    expected = torch.tensor(
        [
            [0.0, VERDICT_TOKEN_WEIGHT, EVIDENCE_TOKEN_WEIGHT, EVIDENCE_TOKEN_WEIGHT],
            [0.0, 0.0, VERDICT_TOKEN_WEIGHT, EVIDENCE_TOKEN_WEIGHT],
        ]
    )
    assert torch.equal(weights, expected)
    with pytest.raises(ValueError, match="atomic verdict"):
        build_loss_weights(labels, [9, 8])


def test_exp570_unit_weight_null_control_is_bit_exact() -> None:
    torch.manual_seed(570)
    logits = torch.randn(2, 5, 11)
    labels = torch.tensor([[-100, 1, 2, 3, 4], [-100, -100, 5, 6, 7]])
    candidate, reference = exact_unit_weight_null_control(logits, labels)

    assert torch.equal(candidate, reference)
    ones = labels.ne(-100).float()
    assert torch.equal(weighted_causal_cross_entropy(logits, labels, ones), reference)


def test_exp570_weighted_ce_uses_sum_of_active_weights() -> None:
    logits = torch.tensor(
        [[[0.0, 0.0], [4.0, -4.0], [-2.0, 2.0], [1.0, -1.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[-100, 0, 1, 0]])
    weights = build_loss_weights(labels, [0])
    token_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 2), labels[:, 1:].reshape(-1), reduction="none"
    )
    expected = (
        token_loss[0] * VERDICT_TOKEN_WEIGHT
        + token_loss[1] * EVIDENCE_TOKEN_WEIGHT
        + token_loss[2] * EVIDENCE_TOKEN_WEIGHT
    ) / (VERDICT_TOKEN_WEIGHT + 2 * EVIDENCE_TOKEN_WEIGHT)

    assert torch.allclose(weighted_causal_cross_entropy(logits, labels, weights), expected)


def test_exp570_forward_runs_null_control_once_and_overrides_loss() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.fixed_logits = torch.randn(1, 4, 9, requires_grad=True)

        def forward(self, *, labels=None):
            assert labels is None
            return SimpleNamespace(logits=self.fixed_logits, loss=None)

    labels = torch.tensor([[-100, 1, 2, 3]])
    weights = build_loss_weights(labels, [1])
    audit = RuntimeLossAudit()
    model = install_weighted_forward(FakeModel(), audit)

    first = model.forward(labels=labels, loss_weights=weights)
    second = model.forward(labels=labels, loss_weights=weights)

    assert audit.null_control_checks == 1
    assert torch.equal(first.loss, second.loss)
    first.loss.backward()
    assert model.fixed_logits.grad is not None


def test_exp570_runtime_audit_is_fail_closed(tmp_path: Path) -> None:
    labels = torch.tensor([[-100, 1, 2, 3]])
    weights = build_loss_weights(labels, [1])
    audit = RuntimeLossAudit()
    audit.observe(labels, weights)
    audit.record_null_control()
    report = audit.finalize(tmp_path / "audit.json", expected_training_occurrences=1)

    assert report["decision"] == "GO"
    assert report["contract_version"] == CONTRACT_VERSION
    assert report["verdict_tokens"] == 1
    assert report["evidence_tokens"] == 2
    with pytest.raises(ValueError, match="runtime audit failed"):
        audit.finalize(tmp_path / "bad.json", expected_training_occurrences=2)


def test_exp570_reuses_exact_exp520_target_and_prompt_contract() -> None:
    exp520_target = _load_module("_exp570_parent_target", PARENT / "structured_target.py")
    exp520_trainer = _load_module("_exp570_parent_trainer_contract", PARENT / "trainer.py")
    target = exp520_target.build_structured_target(
        row_id="570",
        category="БАД",
        name="Комплекс",
        description="Биологически активная добавка к пище",
        gold_verdict=1,
    )

    assert len(target.splitlines()) == 6
    assert target.splitlines()[0] == "1"
    assert exp520_trainer._GROUNDED_ANSWER_INSTRUCTION.startswith(
        "Сначала выведи ровно одну цифру"
    )


def test_exp570_screen_report_requires_weighted_and_frozen_parent_contracts(
    tmp_path: Path,
) -> None:
    screen = _load_module("_exp570_screen", EXPERIMENT / "evaluate_screen.py")
    frozen = json.loads((PARENT / "analysis/coverage_fold_0.json").read_text())
    predictions = tmp_path / "lora_holdout_predictions.csv"
    predictions.write_text("id,label,lora_logit\n", encoding="utf-8")
    runtime = {
        "experiment_id": "570",
        "holdout_fold": 0,
        "seed": 42,
        "training_records": 5390,
        "format_version": screen.FORMAT_VERSION,
        "decision": "GO",
        "outer_validation_training_occurrences": 0,
        "format_failures": [],
        "gates": {"failures": []},
        "target_plan_sha256": frozen["target_plan_sha256"],
        "record_multiset_sha256": frozen["record_multiset_sha256"],
    }
    loss_runtime = {
        "experiment_id": "570",
        "contract_version": screen.LOSS_CONTRACT_VERSION,
        "contract_sha256": screen.LOSS_CONTRACT_SHA256,
        "verdict_token_weight": 1.0,
        "evidence_token_weight": 0.05,
        "batches": 1348,
        "training_occurrences": 5390,
        "verdict_tokens": 5390,
        "evidence_tokens": 100,
        "decision": "GO",
        "failures": [],
        "exact_null_control": {"executed": True, "bit_exact": True},
    }
    report = {
        "holdout_fold": 0,
        "train_records": 5390,
        "download_failures": 0,
        "flammable_selection": {
            "grounded_auxiliary_sft": {
                "format_version": screen.FORMAT_VERSION,
                "target_plan_sha256": frozen["target_plan_sha256"],
            },
            "weighted_grounded_auxiliary": {
                "contract_version": screen.LOSS_CONTRACT_VERSION,
                "verdict_token_weight": 1.0,
                "evidence_token_weight": 0.05,
            },
        },
    }
    predictions.with_name("grounded_target_audit.runtime.json").write_text(
        json.dumps(runtime)
    )
    predictions.with_name("weighted_loss_audit.runtime.json").write_text(
        json.dumps(loss_runtime)
    )
    predictions.with_name("lora_holdout_report.json").write_text(json.dumps(report))

    contract = screen.validate_report(predictions, 0)
    assert set(contract) == {
        "report",
        "report_sha256",
        "grounded_runtime_audit",
        "grounded_runtime_audit_sha256",
        "loss_runtime_audit",
        "loss_runtime_audit_sha256",
    }
    loss_runtime["evidence_token_weight"] = 1.0
    predictions.with_name("weighted_loss_audit.runtime.json").write_text(
        json.dumps(loss_runtime)
    )
    with pytest.raises(ValueError, match="weighted loss contract mismatch"):
        screen.validate_report(predictions, 0)


def test_exp570_copies_the_actual_engine_npz_name() -> None:
    screen = _load_module("_exp570_screen_output_names", EXPERIMENT / "evaluate_screen.py")

    assert screen.engine_output_names(False) == ("screen_audit", "screen_predictions")
    assert screen.engine_output_names(True) == (
        "null_screen_control",
        "null_screen_control",
    )

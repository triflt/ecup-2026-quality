from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PARENT_EXPERIMENT = ROOT / "experiments/520_qwen35_grounded_auxiliary_sft"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PARENT_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(PARENT_EXPERIMENT))

from loss_contract import (
    CONTRACT_VERSION,
    EVIDENCE_TOKEN_WEIGHT,
    EXPERIMENT_ID,
    VERDICT_TOKEN_WEIGHT,
    RuntimeLossAudit,
    build_loss_weights,
    contract_sha256,
    install_weighted_forward,
)


def _load_exp520_trainer():
    path = PARENT_EXPERIMENT / "trainer.py"
    spec = importlib.util.spec_from_file_location("_exp570_frozen_exp520_trainer", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the frozen experiment 520 trainer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = _load_exp520_trainer()


def _retag_coverage_audit(audit: dict[str, Any]) -> dict[str, Any]:
    updated = dict(audit)
    updated["experiment_id"] = EXPERIMENT_ID
    updated.pop("audit_sha256", None)
    payload = json.dumps(
        updated, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    updated["audit_sha256"] = hashlib.sha256(payload).hexdigest()
    return updated


def main() -> int:
    original_load_parent = baseline.load_parent_module
    original_build_coverage_audit = baseline.build_coverage_audit
    runtime_loss_audit = RuntimeLossAudit()

    def weighted_coverage_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return _retag_coverage_audit(original_build_coverage_audit(*args, **kwargs))

    def load_weighted_parent():
        parent = original_load_parent()
        original_main = parent.main

        def weighted_parent_main() -> None:
            grounded_training_batch = parent.training_batch
            grounded_select_training = parent.select_training
            parent_validation_scores = parent.validation_scores
            original_loader = parent.AutoModelForMultimodalLM

            def weighted_training_batch(processor, rows):
                batch = grounded_training_batch(processor, rows)
                expected_first_token_ids = []
                for row in rows:
                    token_ids = processor.tokenizer.encode(
                        str(int(row.label)), add_special_tokens=False
                    )
                    if len(token_ids) != 1:
                        raise ValueError("verdict is not one atomic token")
                    expected_first_token_ids.append(int(token_ids[0]))
                weights = build_loss_weights(batch["labels"], expected_first_token_ids)
                runtime_loss_audit.observe(batch["labels"], weights)
                batch["loss_weights"] = weights
                return batch

            def weighted_select_training(frame, oof):
                records, selection_audit = grounded_select_training(frame, oof)
                selection_audit = dict(selection_audit)
                selection_audit["weighted_grounded_auxiliary"] = {
                    "experiment_id": EXPERIMENT_ID,
                    "contract_version": CONTRACT_VERSION,
                    "contract_sha256": contract_sha256(),
                    "verdict_token_weight": VERDICT_TOKEN_WEIGHT,
                    "evidence_token_weight": EVIDENCE_TOKEN_WEIGHT,
                    "changes_record_multiset": False,
                    "changes_steps": False,
                }
                return records, selection_audit

            def audited_validation_scores(*args: Any, **kwargs: Any):
                runtime_loss_audit.finalize(
                    parent.OUTPUT / "weighted_loss_audit.runtime.json",
                    expected_training_occurrences=5390,
                )
                return parent_validation_scores(*args, **kwargs)

            class WeightedLoader:
                @staticmethod
                def from_pretrained(*args: Any, **kwargs: Any):
                    model = original_loader.from_pretrained(*args, **kwargs)
                    return install_weighted_forward(model, runtime_loss_audit)

            parent.training_batch = weighted_training_batch
            parent.select_training = weighted_select_training
            parent.validation_scores = audited_validation_scores
            parent.AutoModelForMultimodalLM = WeightedLoader
            original_main()

        parent.main = weighted_parent_main
        return parent

    baseline.build_coverage_audit = weighted_coverage_audit
    baseline.load_parent_module = load_weighted_parent
    try:
        return baseline.main()
    finally:
        baseline.build_coverage_audit = original_build_coverage_audit
        baseline.load_parent_module = original_load_parent


if __name__ == "__main__":
    raise SystemExit(main())

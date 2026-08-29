from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = ROOT / "experiments/140_dual_lora_fusion/final/verify.py"
SPEC = importlib.util.spec_from_file_location("solution_140_verify", VERIFY_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_solution_140_repository_contract() -> None:
    report = VERIFY.verify_repository()
    assert report["solution"] == "140"
    assert report["decision"] == "REPOSITORY_CONTRACT_PASS"
    assert report["runtime_network_imports"] == []
    assert report["weights_published"] is False


def test_solution_140_records_distinct_validation_protocols() -> None:
    champion = json.loads((ROOT / "reports/champion.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (ROOT / "experiments/140_dual_lora_fusion/results/metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert champion["validation"]["nested_recurrence_macro_f1"] == 0.942878
    assert metrics["evaluation_version"] == "nested_grouped_v1"
    assert float(metrics["historical_results"][0]["macro_f1"]) == 0.9118425205786493


def test_solution_140_training_entrypoints_exist() -> None:
    required = (
        "experiments/110_qwen3vl_lora/run.py",
        "experiments/130_qwen35_lora/run.py",
        "experiments/140_dual_lora_fusion/run.py",
        "research/qwen3vl_lora_holdout.py",
        "research/aggregate_lora_oof.py",
        "research/nested_multimodel_fusion.py",
    )
    assert all((ROOT / path).is_file() for path in required)

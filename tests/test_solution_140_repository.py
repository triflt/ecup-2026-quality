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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_solution_140_visual_inputs_reach_all_three_qwen_components() -> None:
    source = (
        ROOT / "experiments/140_dual_lora_fusion/submission/run.py"
    ).read_text(encoding="utf-8")
    assert "conversation(row.text, row.image_paths)" in source
    assert "paths = [row.image_paths[0] if row.image_paths else None for row in rows]" in source
    assert 'INSTRUCT_MODEL_PATH, QWEN3VL_ADAPTER_PATH, "image_text"' in source
    assert 'QWEN35_MODEL_PATH, QWEN35_ADAPTER_PATH, "multimodal"' in source
    assert "qwen35_scores = compute_lora_scores(" in source


def test_exp714_preserves_verdict_and_fails_closed() -> None:
    experiment = ROOT / "experiments/714_qwen35_4b_explanation_only"
    runtime = _load_module("exp714_runtime_contract", experiment / "runtime_contract.py")
    output = _load_module(
        "exp714_output_contract", ROOT / "research/explanation_submission_contract.py"
    )
    fallback = (
        "Текст и первое изображение не подтверждают обязательную маркировку "
        "товара как биологически активной добавки."
    )
    audited = (
        "Товар — электретная лечебная плёнка для наложения на рану. В описании "
        "указано, что она усиливает действие БАД, но сама не является биологически "
        "активной добавкой и не имеет соответствующей маркировки. Отрицательный "
        "электрический заряд не делает её БАД."
    )

    comment, status = runtime.normalize_generated_comment(
        audited, fallback, category="БАД", verdict=0
    )
    assert (comment, status) == (audited, "generated_plain_text")
    assert output.format_result(comment, 0) == (
        f"<комментарий>{audited}<вердикт>бан"
    )
    assert output.verdict_from_label(1) == "не бан"

    contradiction = (
        "Карточка имеет маркировку БАД и является биологически активной добавкой, "
        "поэтому соответствует заявленной категории."
    )
    assert runtime.normalize_generated_comment(
        contradiction, fallback, category="БАД", verdict=0
    ) == (fallback, "fallback_verdict_mismatch")
    assert runtime.normalize_generated_comment(
        "Это незаконченное объяснение длиной больше пятидесяти символов без точки",
        fallback,
        category="БАД",
        verdict=0,
    ) == (fallback, "fallback_incomplete_sentence")

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
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
    assert report["solution"] == "140+714"
    assert report["classifier_solution"] == "140"
    assert report["explanation_solution"] == "714"
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
    assert "frozen_predictions = predictions.copy()" in source
    assert "attach_explanation_adapter(qwen35_model, EXPLANATION_ADAPTER_PATH)" in source
    assert "np.array_equal(predictions, frozen_predictions)" in source


def test_exp714_preserves_verdict_and_fails_closed() -> None:
    submission = ROOT / "experiments/140_dual_lora_fusion/submission"
    contract = _load_module(
        "final_explanation_contract", submission / "explanation_contract.py"
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

    comment, status = contract.normalize_generated_comment(
        audited, fallback, category="БАД", verdict=0
    )
    assert (comment, status) == (audited, "generated_plain_text")
    assert contract.format_result(comment, 0) == (
        f"<комментарий>{audited}<вердикт>бан"
    )
    assert contract.verdict_from_prediction(1) == "не бан"

    contradiction = (
        "Карточка имеет маркировку БАД и является биологически активной добавкой, "
        "поэтому соответствует заявленной категории."
    )
    assert contract.normalize_generated_comment(
        contradiction, fallback, category="БАД", verdict=0
    ) == (fallback, "fallback_verdict_mismatch")
    assert contract.normalize_generated_comment(
        "Это незаконченное объяснение длиной больше пятидесяти символов без точки",
        fallback,
        category="БАД",
        verdict=0,
    ) == (fallback, "fallback_incomplete_sentence")


def test_final_source_archive_contains_reasoner(tmp_path: Path) -> None:
    output = tmp_path / "solution140-exp714-source.zip"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/140_dual_lora_fusion/build_submission.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert {
        "run.py",
        "metadata.json",
        "explanation_contract.py",
        "explanation_runtime.py",
    }.issubset(names)
    assert all("__pycache__" not in name and not name.endswith(".pyc") for name in names)

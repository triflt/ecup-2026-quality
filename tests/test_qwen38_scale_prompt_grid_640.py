from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/640_qwen38_scale_prompt_grid/run_prompt_scores.py"
SPEC = importlib.util.spec_from_file_location("exp640_prompt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

EVALUATOR_PATH = ROOT / "experiments/640_qwen38_scale_prompt_grid/evaluate.py"
EVALUATOR_SPEC = importlib.util.spec_from_file_location("exp640_evaluate", EVALUATOR_PATH)
assert EVALUATOR_SPEC is not None and EVALUATOR_SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(EVALUATOR)


def test_prompt_is_atomic_and_category_specific() -> None:
    row = SimpleNamespace(category="БАД", name="Товар", description="Описание")
    text = MODULE.prompt(row)
    assert "Ответь только одной цифрой: 1 или 0" in text
    assert "прямое указание БАД" in text
    assert "Легковоспламеняющиеся" not in text


def test_messages_have_one_image_and_one_text() -> None:
    image = object()
    row = SimpleNamespace(category="Легковоспламеняющиеся", name="Товар", description="Описание")
    result = MODULE.messages(row, image)
    assert result[0]["content"][0]["image"] is image
    assert result[0]["content"][1]["text"].endswith("Ответь только одной цифрой: 1 или 0.")


def test_f1_is_binary_positive_class_f1() -> None:
    assert EVALUATOR.f1(np.asarray([1, 1, 0]), np.asarray([1, 0, 0])) == 2 / 3

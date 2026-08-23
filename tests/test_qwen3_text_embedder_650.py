from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/650_qwen3_text_embedder/run_embeddings.py"
SPEC = importlib.util.spec_from_file_location("exp650_embeddings", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

EVALUATOR_PATH = ROOT / "experiments/650_qwen3_text_embedder/evaluate.py"
EVALUATOR_SPEC = importlib.util.spec_from_file_location("exp650_evaluate", EVALUATOR_PATH)
assert EVALUATOR_SPEC is not None and EVALUATOR_SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(EVALUATOR)


def test_prototypes_cover_both_categories_and_labels() -> None:
    assert set(MODULE.PROTOTYPES) == {
        "БАД|0",
        "БАД|1",
        "Легковоспламеняющиеся|0",
        "Легковоспламеняющиеся|1",
    }


def test_render_contains_only_declared_input_fields() -> None:
    row = {"category": "БАД", "name": "Название", "description": "Описание", "id": "1"}
    assert MODULE.render(row) == "Категория: БАД\nНазвание: Название\nОписание: Описание"


def test_f1_handles_binary_confusion() -> None:
    assert EVALUATOR.f1(np.asarray([1, 1, 0, 0]), np.asarray([1, 0, 1, 0])) == 0.5

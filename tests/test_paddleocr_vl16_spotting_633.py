from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/633_paddleocr_vl16_spotting/run_spotting.py"
SPEC = importlib.util.spec_from_file_location("exp633_spotting", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_spotting_quadrilateral() -> None:
    raw = "Газ<|LOC_100|><|LOC_200|><|LOC_300|><|LOC_200|><|LOC_300|><|LOC_400|><|LOC_100|><|LOC_400|>"
    detections = MODULE.parse_spotting(raw, width=2000, height=1000, sequence_confidence=0.8)
    assert detections == [
        {
            "text": "Газ",
            "polygon": [[200, 200], [600, 200], [600, 400], [200, 400]],
            "normalized_polygon": [[100, 200], [300, 200], [300, 400], [100, 400]],
            "confidence": 0.8,
            "confidence_type": "sequence_geomean_token_probability",
        }
    ]


def test_parse_spotting_rejects_invalid_blocks() -> None:
    raw = "Текст<|LOC_1|><|LOC_2|><|LOC_3|><|LOC_4|>\nПусто"
    assert MODULE.parse_spotting(raw, width=10, height=10, sequence_confidence=None) == []


def test_parse_spotting_handles_multiple_blocks() -> None:
    block = "<|LOC_0|><|LOC_0|><|LOC_1000|><|LOC_0|><|LOC_1000|><|LOC_1000|><|LOC_0|><|LOC_1000|>"
    detections = MODULE.parse_spotting(f"Первый{block}\nВторой{block}", width=20, height=30, sequence_confidence=1.0)
    assert [item["text"] for item in detections] == ["Первый", "Второй"]
    assert all(item["polygon"][2] == [20, 30] for item in detections)


def test_image_size_comes_from_pinned_model_config(tmp_path: Path) -> None:
    (tmp_path / "preprocessor_config.json").write_text(
        '{"min_pixels": 112896, "max_pixels": 1003520}\n',
        encoding="utf-8",
    )
    assert MODULE.image_size_from_model_config(tmp_path) == {
        "shortest_edge": 112896,
        "longest_edge": 2048 * 28 * 28,
    }


def test_canonical_empty_page_generation_is_parseable() -> None:
    assert MODULE.is_parseable_generation("</s>", []) is True
    assert MODULE.is_parseable_generation("", []) is True
    assert MODULE.is_parseable_generation("malformed", []) is False

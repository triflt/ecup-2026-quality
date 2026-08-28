from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_runtime():
    spec = importlib.util.spec_from_file_location(
        "exp699_exact_family_prior_runtime",
        ROOT / "exact_family_prior_runtime.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load_runtime()


def test_priority_prefers_exact_text() -> None:
    labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
    selected = runtime._select_unanimous_family(
        {
            "exact_text": [0],
            "image_exact": [2],
            "normalized_text": [2, 3],
        },
        labels,
    )
    assert selected == ("exact_text", [0], 1)


def test_conflicting_higher_priority_family_abstains() -> None:
    labels = np.asarray([1, 0, 1, 1], dtype=np.int8)
    assert (
        runtime._select_unanimous_family(
            {
                "exact_text": [0, 1],
                "image_exact": [2],
                "normalized_text": [2, 3],
            },
            labels,
        )
        is None
    )


def test_normalized_text_requires_two_donors() -> None:
    labels = np.asarray([1, 1], dtype=np.int8)
    assert (
        runtime._select_unanimous_family(
            {"exact_text": [], "image_exact": [], "normalized_text": [0]},
            labels,
        )
        is None
    )
    assert runtime._select_unanimous_family(
        {"exact_text": [], "image_exact": [], "normalized_text": [0, 1]},
        labels,
    ) == ("normalized_text", [0, 1], 1)


def test_no_similarity_channels_exist() -> None:
    assert runtime.FAMILY_PRIORITY == (
        "exact_text",
        "image_exact",
        "normalized_text",
    )
    assert "tfidf" not in runtime.FAMILY_PRIORITY
    assert "bm25" not in runtime.FAMILY_PRIORITY
    assert "image_near" not in runtime.FAMILY_PRIORITY

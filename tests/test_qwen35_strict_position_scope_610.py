from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/610_qwen35_strict_position_scope"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(EXP))
protocol = _load("exp610_position_protocol_test", EXP / "position_protocol.py")
blind = _load("exp610_blind_audit_test", EXP / "blind_audit.py")
trainer = _load("exp610_train_screen_test", EXP / "train_screen.py")


def test_strict_parser_closes_all_four_legacy_failure_modes() -> None:
    multiple = (
        "Описание. комплектация включает футляр для зажигалки "
        "(зажигалка не входит в комплект)."
    )
    operational = "Горелка с пьезоподжигом позволяет обойтись без спичек или зажигалки."
    anaphora = (
        "Походная плита. Плита совместима с газовыми баллонами. "
        "Это дает гибкость выбора топлива."
    )
    implicit = "Описание товара. В комплект входит набор 2 штуки."
    assert protocol.analyze_scope_sentence(multiple).reason in {
        "ambiguous_scope_types",
        "multiple_or_implicit_scope_relations",
    }
    assert protocol.analyze_scope_sentence(operational).reason == "no_scope_sentence"
    assert protocol.analyze_scope_sentence(anaphora).reason == "following_anaphora"
    assert protocol.analyze_scope_sentence(implicit).reason == "implicit_object"


def test_strict_parser_moves_only_one_complete_relation_sentence() -> None:
    source = (
        "Туристическая горелка для походов. "
        "Газовый баллон не входит в комплект. "
        "Перед использованием прочитайте инструкцию."
    )
    analysis = protocol.analyze_scope_sentence(source)
    assert analysis.reason == "eligible"
    transformed = protocol.move_whole_sentence_to_front(source, analysis.match)
    assert transformed == (
        "Газовый баллон не входит в комплект. "
        "Туристическая горелка для походов. "
        "Перед использованием прочитайте инструкцию."
    )


def _selected_frame() -> tuple[pd.DataFrame, list[int], np.ndarray]:
    rows = 8
    frame = pd.DataFrame(
        {
            "id": [f"row-{index}" for index in range(rows)],
            "category": [protocol.FLAMMABLE] * rows,
            "label": [index % 2 for index in range(rows)],
            "description": [
                f"Походная горелка номер {index}. Газовый баллон не входит в комплект. Инструкция."
                for index in range(rows)
            ],
        }
    )
    records = [index for index in range(rows) for _ in range(2)]
    return frame, records, np.asarray([1, 1, 2, 2, 4, 4, 1, 2], dtype=np.int8)


def test_plan_is_label_blind_and_preserves_exact_record_contract() -> None:
    frame, records, folds = _selected_frame()
    transforms_a, manifest_a, audit_a = protocol.build_augmentation_plan(
        frame, records, folds, outer_fold=0
    )
    relabeled = frame.copy()
    relabeled["label"] = 1 - relabeled["label"]
    transforms_b, manifest_b, audit_b = protocol.build_augmentation_plan(
        relabeled, records, folds, outer_fold=0
    )
    assert transforms_a == transforms_b
    assert manifest_a == manifest_b
    assert audit_a["label_sequence_sha256"] != audit_b["label_sequence_sha256"]
    assert audit_a["record_order_sha256"] == audit_a["candidate_record_order_sha256"]
    assert audit_a["id_sequence_sha256"] == audit_a["candidate_id_sequence_sha256"]
    assert audit_a["label_sequence_sha256"] == audit_a["candidate_label_sequence_sha256"]
    assert audit_a["half_selection"]["observed_rate"] == 0.5
    assert audit_a["all_character_multisets_equal"] is True
    assert audit_a["all_token_multisets_equal"] is True


def test_plan_rejects_outer_validation_occurrence() -> None:
    frame, records, folds = _selected_frame()
    folds[0] = 0
    with pytest.raises(ValueError, match="outer-validation"):
        protocol.build_augmentation_plan(frame, records, folds, outer_fold=0)


def test_blind_sample_uses_only_independent_folds_and_excludes_legacy_ids() -> None:
    ids = [f"id-{index}" for index in range(120)] + ["legacy", "screen"]
    data = pd.DataFrame(
        {
            "id": ids,
            "category": [protocol.FLAMMABLE] * len(ids),
            "name": ["Горелка"] * len(ids),
            "description": [
                f"Горелка {index}. Баллон не входит в комплект. Инструкция."
                for index in range(len(ids))
            ],
            "label": [index % 2 for index in range(len(ids))],
        }
    )
    folds = pd.DataFrame(
        {
            "id": ids,
            "split": ["development"] * len(ids),
            "development_fold": [[1, 2, 4][index % 3] for index in range(120)] + [2, 0],
        }
    )
    sample, report = blind.build_sample(data=data, folds=folds, legacy_ids={"legacy"})
    assert len(sample) == 120
    assert report["decision"] == "READY_FOR_BLIND_REVIEW"
    assert report["label_columns_read"] is False
    assert report["screen_folds_used"] == []
    assert "legacy" not in sample.to_csv(index=False)
    assert "screen" not in sample.to_csv(index=False)


def test_blind_scoring_requires_98_percent_and_kappa_0_80() -> None:
    sample = pd.DataFrame(
        {
            **{
                column: [f"{column}-{index}" for index in range(100)]
                for column in blind.IMMUTABLE_SAMPLE_COLUMNS
            },
            "reviewer_a": ["preserved"] * 98 + ["rejected"] * 2,
            "reviewer_b": ["preserved"] * 98 + ["rejected"] * 2,
            "failure_reason_a": [""] * 100,
            "failure_reason_b": [""] * 100,
        }
    )
    provenance = {
        "sample_immutable_sha256": blind._sample_hash(sample),
        "screen_folds_used": [],
        "label_columns_read": False,
        "model_outputs_read": False,
    }
    report = blind.score_reviews(sample, provenance)
    assert report["strict_pass_rate"] == 0.98
    assert report["cohen_kappa"] == 1.0
    assert report["decision"] == "GO"


def test_current_preflight_is_honest_no_go_and_cannot_unlock_trainer() -> None:
    provenance = json.loads(
        (EXP / "analysis/preflight/blind_audit_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["independent_pool_rows"] == 91
    assert provenance["sample_rows"] == 91
    assert provenance["decision"] == "NO_GO_INSUFFICIENT_INDEPENDENT_PAIRS"
    assert provenance["gpu_launch_allowed"] is False
    with pytest.raises(ValueError, match="has not authorized"):
        trainer._require_blind_gate(EXP / "analysis/preflight/blind_audit_provenance.json")


def test_semantic_v3_dependency_contract_is_strict() -> None:
    shared = _load("exp610_shared_test", EXP / "position_shared.py")
    semantic, base_trainer = shared.load_exp600_dependencies()
    assert semantic.SELECTOR_SOURCE_PROTOCOL == "semantic_v3_robust_base_strict_nested_v2"
    assert base_trainer.PARENT_SHA256["specialist"] == (
        "404f6d07965f551dfe7c7ee0120ce0a16132e1103f1db7d13b3622993f4fd6c0"
    )

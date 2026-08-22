from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "experiments"
    / "500_transaction_scope_position_aug"
    / "transaction_scope_position_aug.py"
)
SPEC = importlib.util.spec_from_file_location("transaction_scope_position_aug_500", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUG
SPEC.loader.exec_module(AUG)


def _row_id_selected_by_exp500(fold: int, selected: bool) -> str:
    for index in range(10_000):
        row_id = f"row-{index}"
        digest = AUG.stable_row_fold_hash(row_id, fold)
        if AUG.hash_selects_half(digest) is selected:
            return row_id
    raise AssertionError("could not find deterministic hash fixture")


def test_exp500_moves_one_whole_scope_sentence_without_token_changes() -> None:
    original = (
        "Туристическая горелка для походов. "
        "Газовый баллон не входит в комплект. "
        "Перед использованием прочитайте инструкцию."
    )
    analysis = AUG.analyze_scope_sentence(original)
    assert analysis.reason == "eligible"
    assert analysis.match is not None
    moved = original[analysis.match.start : analysis.match.end]
    transformed = AUG.move_whole_sentence_to_front(original, analysis.match)
    assert transformed.startswith(moved)
    assert len(transformed) == len(original)
    assert Counter(transformed) == Counter(original)
    assert Counter(AUG.audit_tokens(transformed)) == Counter(AUG.audit_tokens(original))
    assert transformed == (
        "Газовый баллон не входит в комплект. "
        "Туристическая горелка для походов. "
        "Перед использованием прочитайте инструкцию."
    )


def test_exp500_rejects_ambiguous_multiple_and_already_first_scope_sentences() -> None:
    ambiguous = (
        "Описание товара. В комплект входит баллон, но запасной баллон "
        "приобретается отдельно. Завершение."
    )
    multiple = (
        "Описание товара. Баллон не входит в комплект. Горелка совместима с газовым баллоном."
    )
    already_first = "Баллон не входит в комплект. Описание товара."
    assert AUG.analyze_scope_sentence(ambiguous).reason == "ambiguous_scope_types"
    assert AUG.analyze_scope_sentence(multiple).reason == "multiple_scope_sentences"
    assert AUG.analyze_scope_sentence(already_first).reason == "scope_sentence_already_first"
    assert (
        AUG.analyze_scope_sentence("<p>Горелка.</p><p>Баллон не входит в комплект.</p>").reason
        == "html_markup_present"
    )


def test_exp500_hash_selection_is_stable_and_independent_of_training_seed() -> None:
    row_id = "stable-row"
    digest_a = AUG.stable_row_fold_hash(row_id, 3)
    digest_b = AUG.stable_row_fold_hash(row_id, 3)
    digest_other_fold = AUG.stable_row_fold_hash(row_id, 4)
    assert digest_a == digest_b
    assert digest_a != digest_other_fold
    assert AUG.hash_selects_half(digest_a) == AUG.hash_selects_half(digest_b)


def test_exp500_plan_is_label_blind_and_preserves_parent_record_multiset() -> None:
    fold = 2
    selected_id = _row_id_selected_by_exp500(fold, True)
    control_id = _row_id_selected_by_exp500(fold, False)
    descriptions = [
        "Горелка. Баллон не входит в комплект. Инструкция.",
        "Горелка. В комплект входит газовый баллон. Инструкция.",
        "Добавка. Описание без transaction scope.",
    ]
    frame = pd.DataFrame(
        {
            "id": [selected_id, control_id, "bad-row"],
            "category": [AUG.FLAMMABLE, AUG.FLAMMABLE, "БАД"],
            "description": descriptions,
            "label": [0, 1, 0],
        }
    )
    records = [0, 0, 1, 2, 2]
    fold_ids = np.asarray([0, 1, 1], dtype=np.int8)
    transforms_a, manifest_a, audit_a = AUG.build_augmentation_plan(
        frame, records, fold_ids, holdout_fold=fold, full_train=False
    )
    relabeled = frame.copy()
    relabeled["label"] = [1, 0, 1]
    transforms_b, manifest_b, audit_b = AUG.build_augmentation_plan(
        relabeled, records, fold_ids, holdout_fold=fold, full_train=False
    )
    assert transforms_a == transforms_b
    assert manifest_a == manifest_b
    assert audit_a == audit_b
    assert set(transforms_a) == {0}
    assert audit_a["training_records"] == len(records)
    assert audit_a["parent_record_multiset_sha256"] == audit_a["candidate_record_multiset_sha256"]
    assert audit_a["parent_record_order_sha256"] == audit_a["candidate_record_order_sha256"]
    assert audit_a["multiplicity_unchanged"] is True
    assert audit_a["steps_unchanged"] is True
    assert audit_a["outer_validation_rows_transformed"] == 0
    assert audit_a["bad_rows_transformed"] == 0
    assert audit_a["candidate_unique_rows"] == 2
    assert audit_a["candidate_training_occurrences"] == 3
    assert audit_a["eligible_unique_row_coverage"] == 1.0
    assert audit_a["transformed_occurrence_weighted_coverage"] == pytest.approx(2 / 3)
    assert audit_a["observed_unique_row_selection_rate"] == 0.5
    assert audit_a["observed_occurrence_weighted_selection_rate"] == pytest.approx(2 / 3)
    assert audit_a["transformed_training_occurrences"] == 2


def test_exp500_plan_fails_closed_if_outer_validation_enters_training_records() -> None:
    frame = pd.DataFrame(
        {
            "id": ["validation-row"],
            "category": [AUG.FLAMMABLE],
            "description": ["Горелка. Баллон не входит в комплект."],
        }
    )
    with pytest.raises(AssertionError, match="outer-validation row"):
        AUG.build_augmentation_plan(
            frame,
            [0],
            np.asarray([3], dtype=np.int8),
            holdout_fold=3,
            full_train=False,
        )

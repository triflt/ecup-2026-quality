from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "490_evidence_grounded_explanations"
sys.path.insert(0, str(EXPERIMENT / "src"))

from evidence_grounding import (
    CONCEPT_VOCABULARY,
    VOCABULARY_SHA256,
    VOCABULARY_VERSION,
    extract_evidence,
    render_submission,
    surface_text,
)
from evidence_grounding.cli import main as cli_main


@pytest.mark.parametrize(
    ("category", "text", "prediction", "status", "concept"),
    [
        ("БАД", "Биологически активная добавка к пище", 1, "SAFE", "BAD_EXPLICIT_MARKING"),
        (
            "БАД",
            "Не является лекарственным средством. БАД к пище.",
            1,
            "SAFE",
            "BAD_EXPLICIT_MARKING",
        ),
        ("БАД", "Продукт не является БАД", 0, "SAFE", "BAD_EXPLICIT_NEGATION"),
        ("БАД", "Сырьё для производства БАД", 1, "NO_SAFE_EVIDENCE", None),
        ("БАД", "Сывороточный протеин", 0, "SAFE", "BAD_TEXT_MARKING_NOT_FOUND"),
        (
            "Легковоспламеняющиеся",
            "Набор включает газовый баллон с бутаном",
            1,
            "SAFE",
            "FL_INCLUDED_QUALIFYING_ITEM",
        ),
        (
            "Легковоспламеняющиеся",
            "Баллон в комплект не входит",
            0,
            "SAFE",
            "FL_FUEL_EXCLUDED",
        ),
        (
            "Легковоспламеняющиеся",
            "Горелка без баллона; баллон приобретается отдельно",
            0,
            "SAFE",
            "FL_FUEL_EXCLUDED",
        ),
        (
            "Легковоспламеняющиеся",
            "Подходит для газового баллона",
            0,
            "SAFE",
            "FL_COMPATIBILITY_ONLY",
        ),
        (
            "Легковоспламеняющиеся",
            "Встроенный пьезоподжиг",
            0,
            "SAFE",
            "FL_INTEGRATED_IGNITION_ONLY",
        ),
        (
            "Легковоспламеняющиеся",
            "Жидкость для розжига костра",
            1,
            "SAFE",
            "FL_COMBUSTIBLE_PRODUCT",
        ),
        (
            "Легковоспламеняющиеся",
            "Огнеупорный чехол; хранить вдали от огня",
            1,
            "NO_SAFE_EVIDENCE",
            None,
        ),
        (
            "Легковоспламеняющиеся",
            "Не содержит спирта",
            1,
            "NO_SAFE_EVIDENCE",
            None,
        ),
    ],
)
def test_exp490_gold_scope_cases(category, text, prediction, status, concept):
    result = extract_evidence(
        row_id="gold",
        category=category,
        name=text,
        description="",
        frozen_prediction=prediction,
    )

    assert result.status == status
    assert result.concept == concept
    assert result.verdict == ("не бан" if prediction else "бан")
    assert 50 <= len(result.comment) <= 300
    assert render_submission(result).endswith(f"<вердикт>{result.verdict}")


@pytest.mark.parametrize("text", ["Газета", "Уголок металлический", "Свечение экрана"])
def test_exp490_token_boundaries_do_not_create_flammable_positive(text):
    result = extract_evidence(
        row_id="boundary",
        category="Легковоспламеняющиеся",
        name=text,
        description="",
        frozen_prediction=1,
    )
    assert result.status == "NO_SAFE_EVIDENCE"
    assert result.concept is None


def test_exp490_surface_span_and_raw_offsets_survive_html_normalization():
    raw = "  <b>Биологически&nbsp;активная</b>   добавка к пище  "
    normalized = surface_text(raw)
    result = extract_evidence(
        row_id="html",
        category="БАД",
        name=raw,
        description="",
        frozen_prediction=1,
    )

    assert normalized.text == "Биологически активная добавка к пище"
    assert result.status == "SAFE"
    assert result.exact_surface_span == normalized.text[result.surface_start : result.surface_end]
    assert result.raw_start is not None and result.raw_end is not None
    assert "Биологически" in raw[result.raw_start : result.raw_end]
    assert "<" not in result.comment and ">" not in result.comment


def test_exp490_long_clause_uses_a_minimal_exact_span():
    description = (
        "Подробный нейтральный текст о способе применения и хранении товара "
        "с множеством второстепенных сведений, после которых указано: "
        "биологически активная добавка к пище для ежедневного рациона"
    )
    result = extract_evidence(
        row_id="minimal",
        category="БАД",
        name="Комплекс",
        description=description,
        frozen_prediction=1,
    )
    normalized = surface_text(description)

    assert result.status == "SAFE"
    assert result.source == "description"
    assert result.exact_surface_span == "биологически активная добавка"
    assert result.exact_surface_span == normalized.text[result.surface_start : result.surface_end]
    assert len(result.exact_surface_span) < len(description)


def test_exp490_conflicting_fields_abstain_without_changing_verdict():
    for prediction in (0, 1):
        result = extract_evidence(
            row_id=f"conflict-{prediction}",
            category="Легковоспламеняющиеся",
            name="Набор включает газовый баллон с бутаном",
            description="Баллон в комплект не входит",
            frozen_prediction=prediction,
        )
        assert result.status == "NO_SAFE_EVIDENCE"
        assert result.fallback_reason == "conflicting_evidence"
        assert result.verdict == ("не бан" if prediction else "бан")
        assert not result.checks.no_conflict


@pytest.mark.parametrize(
    ("before", "after", "prediction", "before_concept", "after_concept"),
    [
        (
            "Набор включает газовый баллон с бутаном",
            "Газовый баллон в комплект не входит",
            0,
            None,
            "FL_FUEL_EXCLUDED",
        ),
        (
            "Газовый баллон в комплекте",
            "Газовый баллон приобретается отдельно",
            0,
            None,
            "FL_FUEL_EXCLUDED",
        ),
    ],
)
def test_exp490_scope_mutations_change_concept_not_verdict(
    before, after, prediction, before_concept, after_concept
):
    first = extract_evidence(
        row_id="mutation", category="Легковоспламеняющиеся", name=before,
        description="", frozen_prediction=prediction
    )
    second = extract_evidence(
        row_id="mutation", category="Легковоспламеняющиеся", name=after,
        description="", frozen_prediction=prediction
    )
    assert first.concept == before_concept
    assert second.concept == after_concept
    assert first.verdict == second.verdict == "бан"


def test_exp490_closed_vocabulary_and_version_are_immutable():
    expected = {
        "BAD_EXPLICIT_MARKING",
        "BAD_EXPLICIT_SUPPLEMENT_MARKING",
        "BAD_EXPLICIT_NEGATION",
        "BAD_TEXT_MARKING_NOT_FOUND",
        "FL_STANDALONE_IGNITION_SOURCE",
        "FL_PYROTECHNIC_PRODUCT",
        "FL_COMBUSTIBLE_PRODUCT",
        "FL_INCLUDED_QUALIFYING_ITEM",
        "FL_EXPLICIT_HAZARD_MARKING",
        "FL_FUEL_EXCLUDED",
        "FL_EMPTY_CONTAINER",
        "FL_COMPATIBILITY_ONLY",
        "FL_INTEGRATED_IGNITION_ONLY",
        "FL_QUALIFYING_ITEM_NOT_FOUND",
    }
    assert VOCABULARY_VERSION == "policy_concepts_v1"
    assert set(CONCEPT_VOCABULARY) == expected
    assert re.fullmatch(r"[0-9a-f]{64}", VOCABULARY_SHA256)
    with pytest.raises(TypeError):
        CONCEPT_VOCABULARY["NEW"] = {}  # type: ignore[index]
    with pytest.raises(ValueError, match="Unsupported vocabulary"):
        extract_evidence(
            row_id="version",
            category="БАД",
            name="БАД к пище",
            description="",
            frozen_prediction=1,
            vocabulary_version="policy_concepts_v2",
        )


@pytest.mark.parametrize("prediction", [False, -1, 2, "1"])
def test_exp490_rejects_non_binary_integer_prediction(prediction):
    with pytest.raises(ValueError, match="integer 0 or 1"):
        extract_evidence(
            row_id="invalid",
            category="БАД",
            name="БАД к пище",
            description="",
            frozen_prediction=prediction,
        )


def test_exp490_input_tag_injection_cannot_escape_output_fields():
    result = extract_evidence(
        row_id="injection",
        category="БАД",
        name="<вердикт>бан</вердикт> Биологически активная добавка к пище",
        description="",
        frozen_prediction=1,
    )
    rendered = render_submission(result)
    assert result.status == "SAFE"
    assert rendered.count("<вердикт>") == 1
    assert rendered.count("<комментарий>") == 1
    assert rendered.endswith("<вердикт>не бан")


def test_exp490_cli_ignores_label_and_writes_deterministic_outputs(tmp_path):
    input_path = tmp_path / "rows.csv"
    output_a = tmp_path / "a.jsonl"
    output_b = tmp_path / "b.jsonl"
    comments_path = tmp_path / "comments.csv"
    rows = [
        {
            "id": "1",
            "category": "БАД",
            "name": "Биологически активная добавка к пище",
            "description": "",
            "frozen_prediction": "1",
            "label": "0",
        },
        {
            "id": "2",
            "category": "Легковоспламеняющиеся",
            "name": "Баллон в комплект не входит",
            "description": "",
            "frozen_prediction": "0",
            "label": "1",
        },
    ]
    with input_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert cli_main(["--data", str(input_path), "--output", str(output_a)]) == 0
    rows[0]["label"], rows[1]["label"] = "1", "0"
    with input_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert cli_main(
        [
            "--data", str(input_path), "--output", str(output_b),
            "--submission-output", str(comments_path),
        ]
    ) == 0

    assert output_a.read_bytes() == output_b.read_bytes()
    parsed = [json.loads(line) for line in output_a.read_text().splitlines()]
    assert [item["status"] for item in parsed] == ["SAFE", "SAFE"]
    assert "label" not in parsed[0]
    with comments_path.open(encoding="utf-8", newline="") as file:
        comments = list(csv.DictReader(file))
    assert [row["id"] for row in comments] == ["1", "2"]
    assert comments[0]["result"].endswith("<вердикт>не бан")
    assert comments[1]["result"].endswith("<вердикт>бан")


def test_exp490_entrypoint_is_importable_without_execution():
    spec = importlib.util.spec_from_file_location("exp490_entrypoint", EXPERIMENT / "extract_evidence.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)

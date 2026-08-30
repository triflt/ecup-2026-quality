from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
EXP712 = ROOT / "experiments" / "712_qwen35_397b_evidence_teacher"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prompt = load("exp712_prompt", EXP712 / "prompt.py")
targets = load("exp712_targets", EXP712 / "build_student_targets.py")
label_last = load(
    "exp713_contract",
    ROOT / "experiments" / "713_qwen35_4b_rationale_then_label" / "target_contract.py",
)
explanation_only_contract = load(
    "exp714_contract",
    ROOT / "experiments" / "714_qwen35_4b_explanation_only" / "target_contract.py",
)
submission_contract = load(
    "explanation_submission_contract",
    ROOT / "research" / "explanation_submission_contract.py",
)
explanation_runtime_contract = load(
    "exp714_runtime_contract",
    ROOT / "experiments" / "714_qwen35_4b_explanation_only" / "runtime_contract.py",
)


def sample_teacher():
    return {
        "label": 1,
        "reason": "included_flammable_item",
        "evidence": {
            "source": "description",
            "value": "газовый баллон входит в комплект",
        },
        "explanation": (
            "В комплект горелки прямо включён газовый баллон, поэтому карточка "
            "содержит продаваемый горючий газ и относится к опасной категории."
        ),
    }


def test_teacher_schema_and_exact_span():
    teacher = sample_teacher()
    result = prompt.validate_output(
        teacher,
        category="Легковоспламеняющиеся",
        expected_label=1,
        name="горелка",
        description="Новая газовая горелка; газовый баллон входит в комплект.",
        image_count=2,
    )
    assert result.accepted, result.errors
    teacher["evidence"]["value"] = "баллон якобы входит"
    result = prompt.validate_output(
        teacher,
        category="Легковоспламеняющиеся",
        expected_label=1,
        name="горелка",
        description="Новая газовая горелка; газовый баллон входит в комплект.",
        image_count=2,
    )
    assert not result.accepted


def test_label_is_final_character():
    target = targets.rationale_then_label(sample_teacher(), 1)
    parsed = label_last.parse_target(target)
    assert target.endswith("LABEL=1")
    assert parsed["label"] == 1


def test_rationale_and_digit_control_share_neutral_user_prompt():
    row = SimpleNamespace(
        id="1",
        category="Легковоспламеняющиеся",
        name="Газовая горелка",
        description="Баллон входит в комплект.",
        target="1",
    )
    user = label_last.user_text(row)
    assert "LABEL=" not in user
    assert "JSON" not in user
    rationale_messages = label_last.messages(row, with_answer=False, image="image0")
    digit_messages = label_last.messages(row, with_answer=False, image="image0")
    assert rationale_messages == digit_messages


def test_explanation_target_has_no_label():
    target = targets.explanation_only(sample_teacher(), 1)
    parsed = explanation_only_contract.parse_target(target)
    assert "LABEL=" not in target
    assert not target.startswith("{")
    assert parsed["explanation"] == sample_teacher()["explanation"]


def test_inaccessible_later_image_is_rejected_for_first_image_student():
    teacher = sample_teacher()
    teacher["evidence"] = {"source": "image:3", "value": "виден газовый баллон"}
    try:
        targets.rationale_then_label(teacher, 1, max_student_image_index=0)
    except ValueError:
        pass
    else:
        raise AssertionError("target builder used evidence from an unseen gallery image")


def test_secondary_or_unbound_visual_claim_is_rejected_for_first_image_student():
    teacher = sample_teacher()
    teacher["explanation"] = (
        "В описании горелки указан газовый баллон, а на фотографиях также виден "
        "горючий газ, входящий в комплект продаваемого товара."
    )
    assert "SECONDARY_IMAGE_LANGUAGE" in targets.student_scope_errors(
        teacher, max_student_image_index=0
    )
    teacher["explanation"] = (
        "В описании горелки указан газовый баллон, а на упаковке виден горючий "
        "газ, входящий в комплект продаваемого товара."
    )
    assert "UNBOUND_VISUAL_CLAIM" in targets.student_scope_errors(
        teacher, max_student_image_index=0
    )
    teacher["explanation"] = (
        "В комплект горелки входит газовый баллон, поэтому товар соответствует "
        "требованиям категории легковоспламеняющихся предметов."
    )
    assert "CLASSIFIER_META_LANGUAGE" in targets.student_scope_errors(
        teacher, max_student_image_index=0
    )


def test_missing_bad_marking_cannot_override_literal_marker():
    output = {
        "label": 0,
        "reason": "bad_marking_missing",
        "evidence": {
            "source": "absence",
            "value": "Обязательная маркировка БАД не найдена в названии и описании.",
        },
        "explanation": "Порошок аминокислот представлен как спортивное питание, однако обязательная маркировка на карточке отсутствует.",
    }
    result = prompt.validate_output(
        output,
        category="БАД",
        expected_label=0,
        name="BCAA",
        description="Биологически активная добавка не заменяет питание.",
        image_count=1,
    )
    assert not result.accepted


def test_classifier_meta_language_is_rejected():
    output = sample_teacher()
    output["explanation"] = (
        "Газовый баллон прямо включён в комплект горелки, поэтому карточка "
        "соответствует правилу категории и метке 1."
    )
    result = prompt.validate_output(
        output,
        category="Легковоспламеняющиеся",
        expected_label=1,
        name="горелка",
        description="газовый баллон входит в комплект",
        image_count=1,
    )
    assert not result.accepted


def test_image_bad_marking_requires_allowed_literal_marker():
    output = {
        "label": 1,
        "reason": "bad_marking_present",
        "evidence": {"source": "image:0", "value": "VITAMIN SUPPLEMENT"},
        "explanation": (
            "На этикетке флакона указано только, что это витаминная добавка; "
            "обязательная формулировка о биологически активной добавке не видна."
        ),
    }
    result = prompt.validate_output(
        output,
        category="БАД",
        expected_label=1,
        name="Витамин B12",
        description="Витамин в таблетках.",
        image_count=1,
    )
    assert not result.accepted


def test_bad_warning_is_not_category_negation():
    output = {
        "label": 0,
        "reason": "bad_marking_negated",
        "evidence": {
            "source": "description",
            "value": "Биологически активная добавка не может использоваться вместо питания.",
        },
        "explanation": (
            "Аминокислотный порошок назван биологически активной добавкой, "
            "а предупреждение ограничивает способ применения продукта."
        ),
    }
    result = prompt.validate_output(
        output,
        category="БАД",
        expected_label=0,
        name="BCAA",
        description="Биологически активная добавка не может использоваться вместо питания.",
        image_count=1,
    )
    assert not result.accepted


def test_explicit_bad_status_negation_is_accepted():
    output = {
        "label": 0,
        "reason": "bad_marking_negated",
        "evidence": {"source": "description", "value": "Продукт НЕ является БАД."},
        "explanation": (
            "Продаются капсулы для снижения веса, однако описание прямо сообщает, "
            "что продукт не является биологически активной добавкой."
        ),
    }
    result = prompt.validate_output(
        output,
        category="БАД",
        expected_label=0,
        name="Капсулы",
        description="Продукт НЕ является БАД.",
        image_count=1,
    )
    assert result.accepted, result.errors


def test_competition_result_format_has_no_closing_tags():
    comment = (
        "В комплект продаваемой горелки входит газовый баллон с горючим газом, "
        "поэтому источник топлива является частью товара."
    )
    result = submission_contract.format_result(comment, 1)
    assert result == f"<комментарий>{comment}<вердикт>не бан"
    parsed = submission_contract.parse_result(result)
    assert parsed["label"] == 1
    assert "</" not in result


def test_explanation_runtime_repair_and_fallback():
    fallback = (
        "Описание карточки не подтверждает заявленную категорию товара, "
        "поэтому итоговый вердикт остаётся отрицательным."
    )
    comment, status = explanation_runtime_contract.normalize_generated_comment(
        '{"explanation":"Продаётся пустая газовая горелка без баллона в комплекте, поэтому горючий газ вместе с товаром не поставляется."}',
        fallback,
    )
    assert status == "recovered_json_explanation"
    assert 50 <= len(comment) <= 300
    comment, status = explanation_runtime_contract.normalize_generated_comment("коротко", fallback)
    assert comment == fallback
    assert status == "fallback_invalid_generation"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"contract_tests={len(tests)} PASS")

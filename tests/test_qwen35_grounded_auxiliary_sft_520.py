from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "520_qwen35_grounded_auxiliary_sft"
sys.path.insert(0, str(EXPERIMENT))

from evaluate_outputs import main as evaluator_main
from structured_target import (
    NO_SAFE_EVIDENCE,
    build_structured_target,
    parse_first_atomic_verdict,
    parse_structured_target,
    render_explanation,
    validate_exact_evidence,
)


def _load_screen_evaluator():
    spec = importlib.util.spec_from_file_location(
        "exp520_screen_evaluator", EXPERIMENT / "evaluate_screen.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exp520_safe_target_has_atomic_gold_verdict_and_closed_six_line_format() -> None:
    target = build_structured_target(
        row_id="safe",
        category="БАД",
        name="Комплекс",
        description="Биологически активная добавка к пище",
        gold_verdict=1,
    )
    parsed = parse_structured_target(target)

    assert target.splitlines()[0] == "1"
    assert len(target.splitlines()) == 6
    assert parse_first_atomic_verdict(target) == 1
    assert parsed.verdict == 1
    assert parsed.concept == "BAD_EXPLICIT_MARKING"
    assert parsed.source == "description"
    assert parsed.evidence == "Биологически активная добавка"
    assert validate_exact_evidence(
        parsed,
        name="Комплекс",
        description="Биологически активная добавка к пище",
    )


@pytest.mark.parametrize(
    "output",
    ["verdict=1", " 1\nCONCEPT=x", "10\nCONCEPT=x", "2\nCONCEPT=x", ""],
)
def test_exp520_first_atomic_verdict_parser_rejects_noncontract_prefixes(output: str) -> None:
    with pytest.raises(ValueError, match="atomic 0/1"):
        parse_first_atomic_verdict(output)


def test_exp520_html_surface_offsets_recover_exact_evidence() -> None:
    description = "<b>Биологически&nbsp;активная</b> добавка к пище"
    parsed = parse_structured_target(
        build_structured_target(
            row_id="offsets",
            category="БАД",
            name="Комплекс",
            description=description,
            gold_verdict=1,
        )
    )

    assert parsed.evidence == "Биологически активная добавка"
    assert validate_exact_evidence(parsed, name="Комплекс", description=description)
    assert not validate_exact_evidence(
        parsed,
        name="Комплекс",
        description="Другая строка без исходного evidence",
    )


def test_exp520_no_safe_evidence_fallback_is_closed_and_keeps_verdict_first() -> None:
    target = build_structured_target(
        row_id="fallback",
        category="Легковоспламеняющиеся",
        name="Нейтральный товар",
        description="Описание без однозначного основания",
        gold_verdict=1,
    )
    parsed = parse_structured_target(target)

    assert parse_first_atomic_verdict(target) == 1
    assert parsed.concept == NO_SAFE_EVIDENCE
    assert (parsed.source, parsed.start, parsed.end, parsed.evidence) == ("none", -1, -1, "")
    assert "основание не найдено" in render_explanation(parsed)


def test_exp520_parser_rejects_span_bearing_no_safe_fallback() -> None:
    malformed = '0\nCONCEPT=NO_SAFE_EVIDENCE\nSOURCE=name\nSTART=0\nEND=3\nEVIDENCE="БАД"'
    with pytest.raises(ValueError, match="frozen empty fallback"):
        parse_structured_target(malformed)


def test_exp520_gold_verdict_changes_target_without_inventing_opposite_evidence() -> None:
    positive = build_structured_target(
        row_id="gold-lock",
        category="БАД",
        name="БАД к пище",
        description="",
        gold_verdict=1,
    )
    negative = build_structured_target(
        row_id="gold-lock",
        category="БАД",
        name="БАД к пище",
        description="",
        gold_verdict=0,
    )

    assert parse_structured_target(positive).concept == "BAD_EXPLICIT_MARKING"
    assert parse_structured_target(negative).concept == NO_SAFE_EVIDENCE
    assert parse_first_atomic_verdict(positive) == 1
    assert parse_first_atomic_verdict(negative) == 0


def test_exp520_evaluator_hook_preserves_first_verdict_and_checks_exact_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = build_structured_target(
        row_id="eval",
        category="БАД",
        name="БАД к пище",
        description="",
        gold_verdict=1,
    )
    input_path = tmp_path / "generated.csv"
    output_path = tmp_path / "report.json"
    rendered_path = tmp_path / "comments.csv"
    with input_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["id", "category", "name", "description", "label", "generated_output"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "eval",
                "category": "БАД",
                "name": "БАД к пище",
                "description": "",
                "label": 1,
                "generated_output": generated,
            }
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_outputs.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--rendered-output",
            str(rendered_path),
        ],
    )

    assert evaluator_main() == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["first_token_valid_rate"] == 1.0
    assert report["structured_valid_rate"] == 1.0
    assert report["exact_evidence_rate"] == 1.0
    assert report["verdict_accuracy"] == 1.0
    assert "BAD_EXPLICIT_MARKING" in rendered_path.read_text(encoding="utf-8")


def test_exp520_screen_evaluator_requires_frozen_grounded_contract(tmp_path: Path) -> None:
    screen = _load_screen_evaluator()
    frozen = json.loads((EXPERIMENT / "analysis/coverage_fold_0.json").read_text())
    report = {
        "holdout_fold": 0,
        "train_records": 5390,
        "download_failures": 0,
        "flammable_selection": {
            "grounded_auxiliary_sft": {
                "format_version": screen.FORMAT_VERSION,
                "target_plan_sha256": frozen["target_plan_sha256"],
            }
        },
    }
    runtime = {
        "experiment_id": "520",
        "holdout_fold": 0,
        "seed": 42,
        "training_records": 5390,
        "format_version": screen.FORMAT_VERSION,
        "decision": "GO",
        "outer_validation_training_occurrences": 0,
        "format_failures": [],
        "gates": {"failures": []},
        "target_plan_sha256": frozen["target_plan_sha256"],
        "record_multiset_sha256": frozen["record_multiset_sha256"],
    }
    predictions = tmp_path / "lora_holdout_predictions.csv"
    predictions.write_text("id,label,lora_logit\n", encoding="utf-8")
    predictions.with_name("lora_holdout_report.json").write_text(json.dumps(report))
    predictions.with_name("grounded_target_audit.runtime.json").write_text(
        json.dumps(runtime)
    )
    contract = screen.validate_report(predictions, 0)
    assert set(contract) == {
        "report",
        "report_sha256",
        "runtime_audit",
        "runtime_audit_sha256",
    }
    runtime["outer_validation_training_occurrences"] = 1
    predictions.with_name("grounded_target_audit.runtime.json").write_text(
        json.dumps(runtime)
    )
    with pytest.raises(ValueError, match="runtime contract mismatch"):
        screen.validate_report(predictions, 0)

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/672_qwen36_causal_evidence_pilot"


def _module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_locks_verdict_and_exact_text_quote() -> None:
    runner = _module("exp672_runner_test", "run_explanations.py")
    row = SimpleNamespace(
        frozen_prediction=0,
        name="Газовая горелка",
        description="Баллон приобретается отдельно.",
    )
    payload = {
        "verdict": 0,
        "sold_object": "газовая горелка",
        "regulated_substance_or_marker": "газовый баллон",
        "relation": "compatible_external",
        "evidence_source": "description",
        "evidence_quote": "Баллон приобретается отдельно.",
        "explanation": (
            "Продаётся пустая горелка, а газовый баллон прямо указан как приобретаемый отдельно."
        ),
    }
    parsed, errors = runner.validate_payload(json.dumps(payload, ensure_ascii=False), row)
    assert parsed == payload
    assert errors == []
    payload["verdict"] = 1
    _, errors = runner.validate_payload(json.dumps(payload, ensure_ascii=False), row)
    assert "verdict_lock_failed" in errors
    payload["verdict"] = 0
    payload["evidence_quote"] = "баллон не входит"
    _, errors = runner.validate_payload(json.dumps(payload, ensure_ascii=False), row)
    assert "description_quote_not_exact" in errors


def test_parser_rejects_extra_text_and_schema_keys() -> None:
    runner = _module("exp672_runner_test2", "run_explanations.py")
    row = SimpleNamespace(frozen_prediction=1, name="БАД", description="БАД к пище")
    parsed, errors = runner.validate_payload("```json\n{}\n```", row)
    assert parsed is None
    assert errors == ["invalid_json_or_extra_text"]
    parsed, errors = runner.validate_payload("{}", row)
    assert parsed is None
    assert errors == ["schema_keys_mismatch"]


def _write_review_fixture(
    base: Path,
    *,
    q27_relevant: int,
    q4_relevant: int,
    q27_contract_valid: int = 40,
) -> tuple[Path, Path, Path]:
    candidates = base / "candidates.csv"
    reviews = base / "reviews.csv"
    manifest_path = base / "manifest.json"
    candidate_fields = ["audit_id", "automatic_contract_valid"]
    review_fields = [
        "audit_id",
        "review_verdict_consistent",
        "review_evidence_relevant",
        "review_object_relation_correct",
        "review_unsupported_fact",
        "review_visual_decisive",
        "review_notes",
    ]
    mapping = {}
    with candidates.open("w", encoding="utf-8", newline="") as cstream, reviews.open(
        "w", encoding="utf-8", newline=""
    ) as rstream:
        cw = csv.DictWriter(cstream, fieldnames=candidate_fields)
        rw = csv.DictWriter(rstream, fieldnames=review_fields)
        cw.writeheader()
        rw.writeheader()
        for alias, relevant in (("qwen35_4b", q4_relevant), ("qwen36_27b", q27_relevant)):
            for index in range(40):
                audit_id = f"{alias}-{index}"
                mapping[audit_id] = {"row_id": str(index), "candidate_alias": alias}
                contract_valid = alias != "qwen36_27b" or index < q27_contract_valid
                cw.writerow(
                    {
                        "audit_id": audit_id,
                        "automatic_contract_valid": str(contract_valid).lower(),
                    }
                )
                rw.writerow(
                    {
                        "audit_id": audit_id,
                        "review_verdict_consistent": "yes",
                        "review_evidence_relevant": "yes" if index < relevant else "no",
                        "review_object_relation_correct": "yes",
                        "review_unsupported_fact": "no",
                        "review_visual_decisive": "no",
                        "review_notes": "",
                    }
                )
    evaluator = _module("exp672_evaluator_fixture", "evaluate_review.py")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "exp672_blind_manifest_v1",
                "blind_candidates_sha256": evaluator.sha256_file(candidates),
                "mapping": mapping,
            }
        ),
        encoding="utf-8",
    )
    return candidates, reviews, manifest_path


def test_evaluator_enforces_integerized_paired_gain(tmp_path: Path) -> None:
    evaluator = _module("exp672_evaluator_test", "evaluate_review.py")
    candidates, reviews, manifest = _write_review_fixture(
        tmp_path, q27_relevant=34, q4_relevant=30
    )
    result = evaluator.evaluate(
        candidates=candidates,
        reviews=reviews,
        private_manifest=manifest,
        output=tmp_path / "score.json",
    )
    assert result["decision"] == "AUTHORIZE_FRESH_200_ROW_EXPLANATION_AUDIT_ONLY"
    assert result["classification_training_authorized"] is False
    assert result["deltas_qwen36_minus_qwen35"]["evidence_relevant_count"] == 4


def test_evaluator_waits_for_complete_review(tmp_path: Path) -> None:
    evaluator = _module("exp672_evaluator_wait_test", "evaluate_review.py")
    candidates, reviews, manifest = _write_review_fixture(
        tmp_path, q27_relevant=34, q4_relevant=30
    )
    rows = list(csv.DictReader(reviews.open(encoding="utf-8")))
    rows[0]["review_evidence_relevant"] = ""
    with reviews.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    result = evaluator.evaluate(
        candidates=candidates,
        reviews=reviews,
        private_manifest=manifest,
        output=tmp_path / "wait.json",
    )
    assert result["decision"] == "WAIT_FOR_COMPLETE_80_CANDIDATE_REVIEW"


def test_evaluator_rejects_before_human_review_when_automatic_gate_is_impossible(
    tmp_path: Path,
) -> None:
    evaluator = _module("exp672_evaluator_early_stop_test", "evaluate_review.py")
    candidates, reviews, manifest = _write_review_fixture(
        tmp_path, q27_relevant=34, q4_relevant=30, q27_contract_valid=38
    )
    rows = list(csv.DictReader(reviews.open(encoding="utf-8")))
    for row in rows:
        for field in evaluator.RATING_FIELDS:
            row[field] = ""
    with reviews.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    result = evaluator.evaluate(
        candidates=candidates,
        reviews=reviews,
        private_manifest=manifest,
        output=tmp_path / "early_stop.json",
    )
    assert result["decision"] == "NO_GO_REJECT_27B_EXPLANATION_SCALE_UP"
    assert result["scores"]["qwen36_27b"]["automatic_contract_valid"] == 38
    assert result["human_review_skipped"] is True

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/622_semantic_v3_contrastive_evidence_verifier"
SPEC = importlib.util.spec_from_file_location("exp622_preflight", EXPERIMENT / "build_preflight.py")
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)
EVAL_SPEC = importlib.util.spec_from_file_location(
    "exp622_audit_eval", EXPERIMENT / "evaluate_human_audit.py"
)
assert EVAL_SPEC and EVAL_SPEC.loader
audit_eval = importlib.util.module_from_spec(EVAL_SPEC)
sys.modules[EVAL_SPEC.name] = audit_eval
EVAL_SPEC.loader.exec_module(audit_eval)


def _csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path, count: int = 310):
    features, membership = [], []
    for index in range(count):
        row_id = str(index)
        features.append(
            {
                "id": row_id,
                "category": "БАД",
                "name": f"БАД пищевая добавка номер {index}",
                "description": "Биологически активная добавка к пище для взрослых.",
            }
        )
        membership.append(
            {
                "id": row_id,
                "category": "БАД",
                "semantic_component": f"family-{index}",
                "component_size": "1",
                "split": "development",
                "development_fold": str(index % 5),
            }
        )
    feature_path, member_path = tmp_path / "features.csv", tmp_path / "membership.csv"
    _csv(feature_path, list(builder.FEATURE_COLUMNS), features)
    _csv(member_path, list(builder.MEMBERSHIP_COLUMNS), membership)
    donors = {}
    for fold in (0, 3):
        path = tmp_path / f"labels-{fold}.csv"
        _csv(
            path,
            ["id", "label"],
            [{"id": str(i), "label": "1"} for i in range(count) if i % 5 != fold],
        )
        donors[fold] = path
    return feature_path, member_path, donors


def test_builds_unrated_fresh_unique_family_packet(tmp_path: Path, monkeypatch) -> None:
    features, membership, donors = _fixture(tmp_path, 380)
    legacy = tmp_path / "legacy"
    _csv(legacy / "old_audit.csv", ["row_id"], [{"row_id": str(i)} for i in range(20)])
    private = tmp_path / ".local" / "packet"
    public = tmp_path / "summary.json"
    summary = builder.build_preflight(
        features_path=features,
        membership_path=membership,
        donor_label_paths=donors,
        private_output_dir=private,
        public_summary_path=public,
        legacy_roots=[legacy],
    )
    with (private / "human_audit_300.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 300
    assert len({row["row_id"] for row in rows}) == 300
    assert len({row["semantic_component"] for row in rows}) == 300
    assert not ({row["row_id"] for row in rows} & {str(i) for i in range(20)})
    assert all(row["strict_pass"] == row["critical_unsupported"] == "" for row in rows)
    assert all(row["scope_or_negation_failure"] == row["review_notes"] == "" for row in rows)
    assert summary["fresh_audit"]["human_ratings_present"] is False
    assert summary["fresh_audit"]["overlap_with_discovered_legacy_ids"] == 0
    assert summary["label_isolation"]["sealed_rows_written"] == 0
    excluded = set(builder.ONTOLOGY["excluded_non_extractive_concepts"])
    for fold in (0, 3):
        candidates = json.loads((private / f"fold_{fold}/candidate_bank.json").read_text())
        assert not ({row["concept"] for row in candidates} & excluded)


def test_rejects_label_leak_and_validation_donor(tmp_path: Path) -> None:
    features, membership, donors = _fixture(tmp_path)
    columns, rows = builder._read_csv(features)
    for row in rows:
        row["label"] = "1"
    _csv(features, columns + ["label"], rows)
    with pytest.raises(ValueError, match="label-like"):
        builder.build_preflight(
            features_path=features,
            membership_path=membership,
            donor_label_paths=donors,
            private_output_dir=tmp_path / ".local" / "packet",
            public_summary_path=tmp_path / "summary.json",
            legacy_roots=[],
        )


def test_rejects_non_private_output_and_component_crossing_folds(tmp_path: Path) -> None:
    features, membership, donors = _fixture(tmp_path)
    with pytest.raises(ValueError, match=".local"):
        builder.build_preflight(
            features_path=features,
            membership_path=membership,
            donor_label_paths=donors,
            private_output_dir=tmp_path / "packet",
            public_summary_path=tmp_path / "summary.json",
            legacy_roots=[],
        )
    columns, rows = builder._read_csv(membership)
    rows[1]["semantic_component"] = rows[0]["semantic_component"]
    _csv(membership, columns, rows)
    with pytest.raises(ValueError, match="cross development folds"):
        builder.build_preflight(
            features_path=features,
            membership_path=membership,
            donor_label_paths=donors,
            private_output_dir=tmp_path / ".local" / "packet",
            public_summary_path=tmp_path / "summary.json",
            legacy_roots=[],
        )


def test_schemas_and_frozen_contract_are_strict() -> None:
    candidate = json.loads((EXPERIMENT / "candidate_bank_schema_v1.json").read_text())
    review = json.loads((EXPERIMENT / "human_audit_schema_v1.json").read_text())
    frozen = json.loads((EXPERIMENT / "frozen_spec.json").read_text())
    assert candidate["additionalProperties"] is False
    assert review["additionalProperties"] is False
    assert frozen["audit"]["sample_size"] == 300
    assert frozen["audit"]["minimum_human_strict_pass"] == 282
    assert frozen["audit"]["maximum_critical_unsupported"] == 0
    assert frozen["audit"]["maximum_scope_or_negation_failure"] == 0


def test_human_gate_rejects_blanks_and_evaluates_only_complete_private_reviews(
    tmp_path: Path,
) -> None:
    rows = []
    for index in range(1, 301):
        rows.append(
            {
                "schema_version": "exp622_human_audit_row_v1",
                "audit_id": f"A{index:03d}",
                "row_id": str(index),
                "semantic_component": f"family-{index}",
                "category": "БАД",
                "product_name": "БАД тест",
                "source": "name",
                "source_text": "БАД тестовая добавка",
                "raw_start": "0",
                "raw_end": "8",
                "exact_surface_span": "БАД тестовая добавка",
                "claim_concept": "BAD_EXPLICIT_MARKING",
                "counterclaim_concept": "BAD_EXPLICIT_NEGATION",
                "hard_negative_mapping": "bad_identity_polarity",
                "strict_pass": "1",
                "critical_unsupported": "0",
                "scope_or_negation_failure": "0",
                "review_notes": "",
            }
        )
    private = tmp_path / ".local" / "review.csv"
    _csv(private, list(builder.REVIEW_COLUMNS), rows)
    summary = audit_eval.evaluate(private, tmp_path / "gate.json")
    assert summary["decision"] == "GO"
    assert summary["strict_pass"] == 300
    rows[0]["strict_pass"] = ""
    blank = tmp_path / ".local" / "blank.csv"
    _csv(blank, list(builder.REVIEW_COLUMNS), rows)
    with pytest.raises(ValueError, match="missing/invalid"):
        audit_eval.evaluate(blank, tmp_path / "blank-gate.json")

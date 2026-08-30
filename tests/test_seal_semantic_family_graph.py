from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation/seal_semantic_family_graph.py"
SPEC = importlib.util.spec_from_file_location("seal_semantic_family_graph_tested", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SEALER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SEALER
SPEC.loader.exec_module(SEALER)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path) -> tuple[Path, Path]:
    draft = root / "draft"
    draft.mkdir()
    rows = [
        ["h", "БАД", 1, "a" * 64, 1, 0, True, -1],
        ["d0a", "БАД", 0, "b" * 64, 2, 1, False, 0],
        ["d0b", "БАД", 0, "b" * 64, 2, 1, False, 0],
        ["d1", "Легковоспламеняющиеся", 0, "c" * 64, 1, 2, False, 1],
        ["d2", "Легковоспламеняющиеся", 1, "d" * 64, 1, 3, False, 2],
        ["d3", "БАД", 1, "e" * 64, 1, 4, False, 3],
        ["d4", "БАД", 0, "f" * 64, 1, 5, False, 4],
    ]
    with (draft / "rows.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(SEALER.EXPECTED_ROW_COLUMNS)
        writer.writerows(rows)
    payloads = {
        "edges.csv": "stage,left_id,right_id\n",
        "key_degrees.csv": "stage,key_hash,key_degree\n",
        "incremental_components.json": "[]\n",
        "manual_audit_samples.csv": "sample_type,left_id,right_id\n",
        "image_fingerprints.csv.gz": "synthetic-fingerprint-bytes",
    }
    for filename, payload in payloads.items():
        (draft / filename).write_bytes(payload.encode())
    output_sha = {
        key: _sha(draft / filename) for key, filename in SEALER.EXPECTED_DRAFT_FILES.items()
    }
    manifest = {
        "graph_version": SEALER.GRAPH_VERSION,
        "status": "draft_requires_manual_audit",
        "draft": True,
        "requires_manual_audit": True,
        "sealed": False,
        "valid_for_candidate_scoring": False,
        "topology": {"label_blind": True, "candidate_scores_used": False},
        "rows": 7,
        "components": 6,
        "largest_component": 2,
        "draft_assignment": {"holdout_candidate_fold": 0},
        "invariants": {
            "all_ids_unique": True,
            "all_rows_accounted_for": True,
            "component_crosses_partition_7": 0,
            "component_crosses_draft_role": 0,
            "component_crosses_dev_fold": 0,
            "draft_candidate_has_dev_fold_minus_one": True,
            "development_has_assigned_fold": True,
            "topology_uses_labels": False,
            "candidate_scores_used": False,
        },
        "input_sha256": {"data": "1" * 64},
        "output_sha256": output_sha,
    }
    (draft / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    audit = {
        "audit_version": SEALER.AUDIT_VERSION,
        "graph_candidate": SEALER.GRAPH_VERSION,
        "status": "complete",
        "decision": "GO",
        "seal_candidate": True,
        "audit_protocol": {
            "label_blind": True,
            "target_annotations_inspected": False,
            "model_outputs_inspected": False,
            "data_partitions_inspected": False,
        },
        "criterion_checks": {key: True for key in SEALER.EXPECTED_CRITERIA},
    }
    audit_path = root / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return draft, audit_path


def _refresh_rows_sha(draft: Path) -> None:
    manifest_path = draft / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"]["rows"] = _sha(draft / "rows.csv")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_semantic_v3_sealer_promotes_without_repartitioning_or_raw_fields(tmp_path: Path) -> None:
    draft, audit = _write_fixture(tmp_path)
    output = tmp_path / "sealed"
    manifest = SEALER.seal(draft, audit, output)
    with (output / "folds.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == SEALER.OUTPUT_COLUMNS
    assert rows[0]["id"] == "h"
    assert rows[0]["split"] == "sealed_holdout"
    assert rows[0]["development_fold"] == "-1"
    assert [row["development_fold"] for row in rows[1:]] == ["0", "0", "1", "2", "3", "4"]
    assert "name" not in rows[0] and "description" not in rows[0] and "images" not in rows[0]
    assert manifest["sealed"] is True
    assert manifest["valid_for_candidate_scoring"] is True
    assert manifest["promotion"]["repartitioned"] is False
    assert manifest["counts"]["rows"] == 7
    assert manifest["counts"]["components"] == 6
    assert manifest["output_sha256"]["folds"] == _sha(output / "folds.csv")


def test_semantic_v3_sealer_fails_closed_on_draft_sha_mismatch(tmp_path: Path) -> None:
    draft, audit = _write_fixture(tmp_path)
    with (draft / "edges.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    output = tmp_path / "sealed"
    with pytest.raises(ValueError, match="SHA mismatch"):
        SEALER.seal(draft, audit, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("decision", "NO_GO", "not GO"),
        ("seal_candidate", False, "did not authorize"),
        ("audit_version", "wrong", "version mismatch"),
    ],
)
def test_semantic_v3_sealer_rejects_invalid_audit_contract(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    draft, audit_path = _write_fixture(tmp_path)
    audit = json.loads(audit_path.read_text())
    audit[field] = value
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        SEALER.seal(draft, audit_path, tmp_path / "sealed")


def test_semantic_v3_sealer_requires_all_three_exact_criteria(tmp_path: Path) -> None:
    draft, audit_path = _write_fixture(tmp_path)
    audit = json.loads(audit_path.read_text())
    audit["criterion_checks"]["accepted_risky_precision"] = False
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="not every seal criterion"):
        SEALER.seal(draft, audit_path, tmp_path / "sealed")


def test_semantic_v3_sealer_recomputes_component_split_invariants(tmp_path: Path) -> None:
    draft, audit = _write_fixture(tmp_path)
    rows_path = draft / "rows.csv"
    rows = list(csv.reader(rows_path.open(encoding="utf-8", newline="")))
    rows[1][3] = "b" * 64
    rows[1][4] = "3"
    rows[2][4] = "3"
    rows[3][4] = "3"
    with rows_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    manifest = json.loads((draft / "manifest.json").read_text())
    manifest["components"] = 5
    manifest["largest_component"] = 3
    (draft / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_rows_sha(draft)
    with pytest.raises(ValueError, match="crosses draft partition"):
        SEALER.seal(draft, audit, tmp_path / "sealed")


def test_semantic_v3_sealer_refuses_to_overwrite_immutable_output(tmp_path: Path) -> None:
    draft, audit = _write_fixture(tmp_path)
    output = tmp_path / "sealed"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        SEALER.seal(draft, audit, output)

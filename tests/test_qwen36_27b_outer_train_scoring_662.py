from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/662_qwen36_27b_outer_train_scoring"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_runtime_and_archive(
    tmp_path: Path,
    *,
    outer_fold: int = 0,
    mode: str = "smoke8",
):
    verifier = load("verify662_fixture", EXP / "verify_artifact.py")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    rows = [
        {
            "global_index": 10,
            "id": "a",
            "category": "БАД",
            "fold": (outer_fold + 1) % 5,
            "occurrence_index": 0,
            "name": "n",
            "description": "d",
            "image_url": "https://example.invalid/a.jpg",
        },
        {
            "global_index": 10,
            "id": "a",
            "category": "БАД",
            "fold": (outer_fold + 1) % 5,
            "occurrence_index": 1,
            "name": "n",
            "description": "d",
            "image_url": "https://example.invalid/a.jpg",
        },
    ]
    input_path = runtime / "score_input.jsonl"
    input_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    keys = [
        [row[key] for key in verifier.KEY_FIELDS]
        for row in rows
    ]
    source_train_sha256 = f"{outer_fold + 1:064x}"
    audit = {
        "experiment_id": "662",
        "outer_fold": outer_fold,
        "mode": mode,
        "rows": 2,
        "score_input_sha256": verifier.sha256_bytes(input_path.read_bytes()),
        "source_train_sha256": source_train_sha256,
        "teacher_adapter_manifest_sha256": "b" * 64,
        "ordered_occurrence_key_sha256": verifier.canonical_sha256(keys),
    }
    audit["contract_sha256"] = verifier.canonical_sha256(audit)
    (runtime / "runtime_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    scores = [
        {key: row[key] for key in verifier.KEY_FIELDS} | {"score": float(index)}
        for index, row in enumerate(rows)
    ]
    score_payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in scores).encode()
    report = {
        "experiment_id": "662",
        "source_experiment_id": "654",
        "outer_fold": outer_fold,
        "mode": mode,
        "rows": 2,
        "score_semantics": "raw_last_token_logit_1_minus_logit_0",
        "source_train_sha256": source_train_sha256,
        "runtime_contract_sha256": audit["contract_sha256"],
        "teacher_adapter_manifest_sha256": "b" * 64,
        "ordered_occurrence_key_sha256": audit["ordered_occurrence_key_sha256"],
        "teacher_scores_sha256": verifier.sha256_bytes(score_payload),
        "validation_rows_read": 0,
        "validation_labels_read": 0,
        "labels_read": 0,
        "sealed_rows": 0,
        "public_used": False,
        "cpu_or_disk_offload": False,
        "decision": "READY_FOR_FAIL_CLOSED_ACCEPTANCE",
    }
    report["report_contract_sha256"] = verifier.canonical_sha256(report)
    archive = tmp_path / "artifact.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("report.json", json.dumps(report))
        bundle.writestr("teacher_scores.jsonl", score_payload)
    return verifier, runtime, archive


def test_verifier_accepts_repeated_item_occurrences(tmp_path: Path) -> None:
    verifier, runtime, archive = make_runtime_and_archive(tmp_path)
    result = verifier.verify(archive, runtime)
    assert result["decision"] == "ACCEPT_ARTIFACT"
    assert result["rows"] == 2
    assert result["exact_ordered_occurrence_binding"] is True


def test_verifier_rejects_reordered_occurrences(tmp_path: Path) -> None:
    verifier, runtime, archive = make_runtime_and_archive(tmp_path)
    with zipfile.ZipFile(archive) as bundle:
        report = bundle.read("report.json")
        scores = bundle.read("teacher_scores.jsonl").decode().splitlines()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("report.json", report)
        bundle.writestr("teacher_scores.jsonl", "\n".join(reversed(scores)) + "\n")
    with pytest.raises(ValueError, match="immutable occurrence order"):
        verifier.verify(archive, runtime)


def make_target_set(tmp_path: Path):
    aggregate = load("verify_target_set_662_fixture", EXP / "verify_target_set.py")
    runtime_root = tmp_path / "runtime_root"
    runtime_root.mkdir()
    fold_dirs = {}
    for fold in range(5):
        fixture = tmp_path / f"fixture_{fold}"
        fixture.mkdir()
        verifier, runtime, archive = make_runtime_and_archive(
            fixture,
            outer_fold=fold,
            mode="full",
        )
        runtime.rename(runtime_root / f"fold{fold}_full")
        artifact_dir = tmp_path / f"artifact_{fold}"
        (artifact_dir / "extracted").mkdir(parents=True)
        inner = artifact_dir / "extracted" / "teacher_outer_train_scores.zip"
        archive.rename(inner)
        acceptance = verifier.verify(inner, runtime_root / f"fold{fold}_full")
        (artifact_dir / "acceptance_audit.json").write_text(
            json.dumps(acceptance), encoding="utf-8"
        )
        fold_dirs[fold] = artifact_dir
    return aggregate, fold_dirs, runtime_root


def test_target_set_verifier_accepts_exact_five_fold_set(tmp_path: Path) -> None:
    aggregate, fold_dirs, runtime_root = make_target_set(tmp_path)
    result = aggregate.verify_target_set(
        fold_dirs,
        runtime_root,
        expected_rows={fold: 2 for fold in range(5)},
    )
    assert result["decision"] == "ACCEPT_FULL_TARGET_SET"
    assert result["total_occurrences"] == 10
    assert result["folds"] == [0, 1, 2, 3, 4]


def test_target_set_verifier_rejects_missing_fold(tmp_path: Path) -> None:
    aggregate, fold_dirs, runtime_root = make_target_set(tmp_path)
    fold_dirs.pop(4)
    with pytest.raises(ValueError, match="exactly folds"):
        aggregate.verify_target_set(
            fold_dirs,
            runtime_root,
            expected_rows={fold: 2 for fold in range(5)},
        )


def test_target_set_verifier_rejects_tampered_acceptance(tmp_path: Path) -> None:
    aggregate, fold_dirs, runtime_root = make_target_set(tmp_path)
    path = fold_dirs[3] / "acceptance_audit.json"
    audit = json.loads(path.read_text())
    audit["rows"] = 1
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="persisted acceptance audit mismatch"):
        aggregate.verify_target_set(
            fold_dirs,
            runtime_root,
            expected_rows={fold: 2 for fold in range(5)},
        )

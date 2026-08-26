from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"exp693_{name}", HERE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONSUMER = load("exp691_consumer")
TRAIN = load("train_fold")
EVAL = load("evaluate")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def teacher_fixture(
    root: Path, *, with_acceptance: bool
) -> tuple[list[dict], str, Path | None, str | None]:
    runtime_sha = "a" * 64
    train = [
        {"global_index": 1, "id": "fuel", "fold": 1, "category": TRAIN.FLAMMABLE, "label": 1},
        {"global_index": 2, "id": "bad", "fold": 2, "category": "БАД", "label": 0},
    ]
    directory = root / "fold0"
    directory.mkdir(parents=True)
    targets = [
        {
            "global_index": row["global_index"],
            "id": row["id"],
            "fold": row["fold"],
            "category": row["category"],
            "occurrence_index": index,
            "score": 2.0 - index,
        }
        for index, row in enumerate(train)
    ]
    targets_path = directory / "teacher_targets.jsonl"
    write_jsonl(targets_path, targets)
    evidence = {
        "sold_object": {"value": "баллон", "source": "text", "quote": "баллон"},
        "substance": {"value": "газ", "source": "text", "quote": "газ"},
        "relation": {"value": "included", "source": "text", "quote": "в комплекте"},
        "verdict": 1,
        "confidence": 0.9,
        "abstain": False,
        "grounded": True,
        "parse_valid": True,
    }
    evidence_rows = [
        {
            "global_index": 1,
            "id": "fuel",
            "fold": 1,
            "category": TRAIN.FLAMMABLE,
            "evidence": evidence,
            "raw_response": "{}",
        }
    ]
    evidence_path = directory / "evidence.jsonl"
    write_jsonl(evidence_path, evidence_rows)
    report = {
        "schema_version": CONSUMER.FOLD_SCHEMA,
        "experiment_id": "691",
        "scope": "all",
        "fold": 0,
        "runtime_contract_sha256": runtime_sha,
        "validation_labels_loaded_after_adapter_frozen": True,
        "public_used": False,
        "sealed_rows": 0,
        "decision": "FOLD_COMPLETE",
        "teacher_targets_sha256": CONSUMER.sha256_file(targets_path),
        "teacher_target_rows": len(targets),
        "evidence_rows": len(evidence_rows),
        "evidence_grounded": 1,
    }
    report["report_sha256"] = CONSUMER.canonical_sha256(report)
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    if not with_acceptance:
        return train, runtime_sha, None, None
    hexes = "12345"
    bindings = []
    report_hashes = []
    for fold in range(5):
        report_self = report["report_sha256"] if fold == 0 else hexes[fold] * 64
        report_hashes.append(report_self)
        bindings.append(
            {
                "fold": fold,
                "fold_report_file_sha256": (
                    CONSUMER.sha256_file(directory / "report.json")
                    if fold == 0
                    else hexes[fold] * 64
                ),
                "fold_report_self_sha256": report_self,
                "predictions_sha256": hexes[fold] * 64,
                "teacher_targets_sha256": (
                    report["teacher_targets_sha256"] if fold == 0 else hexes[fold] * 64
                ),
                "evidence_sha256": (
                    CONSUMER.sha256_file(evidence_path) if fold == 0 else hexes[fold] * 64
                ),
                "evidence_rows": 1 if fold == 0 else 0,
            }
        )
    acceptance = {
        "schema_version": CONSUMER.ACCEPTANCE_SCHEMA,
        "experiment_id": "692",
        "teacher_experiment_id": "691",
        "teacher_scope": "all",
        "folds": {str(fold): {} for fold in range(5)},
        "pooled": {},
        "gate": {
            "fold_wins": 5,
            "flammable_ap_delta": 0.1,
            "flammable_f1_delta": 0.1,
            "flammable_fn_nonincrease": True,
            "bad_byte_identical": True,
        },
        "artifact_bindings": bindings,
        "teacher_fold_report_self_sha256": report_hashes,
        "runtime_labels_read_after_teacher_terminal": True,
        "public_used": False,
        "sealed_rows": 0,
        "decision": "OPEN_THREE_STUDENT_METHODS",
    }
    acceptance["acceptance_sha256"] = CONSUMER.canonical_sha256(acceptance)
    acceptance_path = root / "student_consumer_acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")
    return train, runtime_sha, acceptance_path, CONSUMER.sha256_file(acceptance_path)


def test_actual_exp691_targets_merge_only_into_flammable(tmp_path: Path):
    train, runtime_sha, acceptance_path, acceptance_sha = teacher_fixture(
        tmp_path, with_acceptance=True
    )
    assert acceptance_path is not None and acceptance_sha is not None
    enriched, binding = CONSUMER.load_fold(
        tmp_path,
        fold=0,
        train=train,
        runtime_contract_sha256=runtime_sha,
        require_evidence=False,
        acceptance_path=acceptance_path,
        expected_acceptance_file_sha256=acceptance_sha,
    )
    assert enriched[0]["teacher_score"] == 2.0
    assert "teacher_score" not in enriched[1]
    assert binding["teacher_targets_sha256"]
    assert binding["evidence_sha256"] is None


def test_actual_exp691_evidence_shape_drives_closed_aux_target(tmp_path: Path):
    train, runtime_sha, acceptance_path, acceptance_sha = teacher_fixture(
        tmp_path, with_acceptance=True
    )
    assert acceptance_path is not None and acceptance_sha is not None
    enriched, _ = CONSUMER.load_fold(
        tmp_path,
        fold=0,
        train=train,
        runtime_contract_sha256=runtime_sha,
        require_evidence=True,
        acceptance_path=acceptance_path,
        expected_acceptance_file_sha256=acceptance_sha,
    )
    targets = TRAIN.structured_targets(SimpleNamespace(**enriched[0]))
    assert tuple(targets) == TRAIN.AUXILIARY_COMPONENTS
    assert targets["sold_object"] == '{"value":"баллон"}'
    assert '"sold_object":{"quote":"баллон"' in targets["evidence_pointer"]
    assert all("verdict" not in target for target in targets.values())
    assert TRAIN.combine_losses(3.0, 2.0, mode="hard_bce_control") == 3.0
    assert (
        TRAIN.combine_losses(
            3.0,
            {component: 2.0 for component in TRAIN.AUXILIARY_COMPONENTS},
            mode="causal_candidate",
        )
        == 3.2
    )


def test_causal_acceptance_requires_exact_file_sha(tmp_path: Path):
    train, runtime_sha, acceptance_path, _ = teacher_fixture(tmp_path, with_acceptance=True)
    assert acceptance_path is not None
    with pytest.raises(ValueError, match="acceptance file SHA-256 mismatch"):
        CONSUMER.load_fold(
            tmp_path,
            fold=0,
            train=train,
            runtime_contract_sha256=runtime_sha,
            require_evidence=True,
            acceptance_path=acceptance_path,
            expected_acceptance_file_sha256="0" * 64,
        )


def test_rejected_exp692_acceptance_cannot_open_any_student(tmp_path: Path):
    _, _, acceptance_path, _ = teacher_fixture(tmp_path, with_acceptance=True)
    assert acceptance_path is not None
    value = json.loads(acceptance_path.read_text(encoding="utf-8"))
    value.pop("acceptance_sha256")
    value["decision"] = "REJECT_TEACHER_TARGETS"
    value["acceptance_sha256"] = CONSUMER.canonical_sha256(value)
    acceptance_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="routed acceptance contract mismatch"):
        CONSUMER.verify_routed_acceptance(
            acceptance_path,
            expected_file_sha256=CONSUMER.sha256_file(acceptance_path),
        )


def test_evaluator_reports_required_metrics_and_bad_exact():
    rows = [
        {
            "label": 1,
            "category": "БАД",
            "baseline": 1,
            "control": 1,
            "candidate": 1,
            "control_score": 1.0,
            "candidate_score": 1.0,
            "component_size": 1,
        },
        {
            "label": 1,
            "category": TRAIN.FLAMMABLE,
            "baseline": 0,
            "control": 0,
            "candidate": 1,
            "control_score": 0.1,
            "candidate_score": 0.9,
            "component_size": 1,
        },
        {
            "label": 0,
            "category": TRAIN.FLAMMABLE,
            "baseline": 0,
            "control": 0,
            "candidate": 0,
            "control_score": 0.1,
            "candidate_score": 0.2,
            "component_size": 2,
        },
    ]
    report = EVAL.evaluate_rows(rows)
    assert report["bad_exact"] is True
    assert report["corrected"] == 1
    assert report["corrections_regressions_ratio"] == "inf"
    assert report["singleton_delta"] == 1
    assert "tie_aware_ap" in report and "macro" in report


def test_average_precision_matches_canonical_sklearn_tie_fixtures():
    assert EVAL.average_precision([1, 1, 0], [0.5, 0.5, 0.5]) == pytest.approx(2 / 3)
    assert EVAL.average_precision([1, 0, 1, 0], [0.5, 0.5, 0.2, 0.1]) == pytest.approx(7 / 12)


def test_label_policy_slice_reports_all_three_fp_and_blocks_regression():
    rows = [
        {
            "fold": index % 5,
            "label": 0,
            "category": TRAIN.FLAMMABLE,
            "name": f"Топливный элемент для зажигалки {index}",
            "baseline": 0,
            "control": 0,
            "candidate": int(index == 0),
            "control_score": -1.0,
            "candidate_score": 1.0 if index == 0 else -1.0,
            "singleton": False,
            "slices": ["ignition_products"],
        }
        for index in range(EVAL.LABEL_POLICY_ROWS)
    ]
    rows.extend(
        [
            {
                "fold": fold,
                "label": 1,
                "category": "БАД",
                "name": "",
                "baseline": 1,
                "control": 1,
                "candidate": 1,
                "control_score": 1.0,
                "candidate_score": 1.0,
                "singleton": False,
                "slices": [],
            }
            for fold in range(5)
        ]
    )
    report = EVAL.build_report(rows)
    policy = report["pooled"]["label_policy_slice"]
    assert policy == {
        "rows": 25,
        "labels_zero": 25,
        "fp": {"baseline": 0, "control": 0, "candidate": 1},
    }
    assert report["gate"]["label_policy_fp_nonincrease"] is False
    assert report["gate"]["evidence_slice_systematic_regression_guard"] is False
    assert report["gate"]["evidence_slice_regression_failures"] == ["ignition_products"]
    assert report["decision"] == "REJECT_CANDIDATE"
    assert "ignition_products" in report["pooled"]["evidence_slices"]


def test_label_policy_contract_rejects_nonzero_gold_without_relabeling():
    rows = [
        {
            "fold": index % 5,
            "label": int(index == 0),
            "category": TRAIN.FLAMMABLE,
            "name": f"топливо для зажигалки {index}",
            "baseline": 0,
            "control": 0,
            "candidate": 0,
            "control_score": -1.0,
            "candidate_score": -1.0,
            "singleton": False,
            "slices": [],
        }
        for index in range(EVAL.LABEL_POLICY_ROWS)
    ]
    with pytest.raises(ValueError, match="25/25 label=0"):
        EVAL.build_report(rows)


def test_exp692_evidence_slice_definitions_are_reused_exactly():
    row = {"name": "Топливо для зажигалки", "description": "угольный розжиг"}
    evidence = {
        "evidence": {
            "sold_object": {"source": "text"},
            "substance": {"source": "text"},
            "relation": {"source": "text", "value": "sold_separately"},
            "abstain": False,
            "grounded": True,
        }
    }
    assert EVAL.evidence_slices(row, evidence) == {
        "fuel_sold_separately",
        "text_sufficient",
        "ignition_products",
        "charcoal_fire_starting",
    }


def test_captain_scientific_gate_is_the_exact_strict_conjunction():
    assert EVAL.MIN_POOLED_MACRO_DELTA == 0.006
    assert EVAL.MIN_FLAMMABLE_F1_DELTA == 0.012
    assert EVAL.MIN_FOLD_WINS == 4
    assert EVAL.MIN_CORRECTIONS_REGRESSIONS_RATIO == 1.5
    assert EVAL.MIN_BOOTSTRAP_P_GAIN == 0.90
    passing = {field: True for field in EVAL.SCIENTIFIC_PASS_FIELDS}
    assert EVAL.scientific_gate_passes(passing) is True
    for field in EVAL.SCIENTIFIC_PASS_FIELDS:
        failing = dict(passing)
        failing[field] = False
        assert EVAL.scientific_gate_passes(failing) is False


def test_resource_gate_reports_contract_values_and_stages_missing_limits():
    observations = {
        "control": {
            "runtime_minutes": [1.0] * 5,
            "peak_gpu_memory_bytes": [10.0, 20.0],
        },
        "candidate": {
            "runtime_minutes": [1.0] * 5,
            "peak_gpu_memory_bytes": [30.0],
        },
    }
    pending = EVAL.resource_summary(observations, None)
    assert pending["status"] == "STAGE_PENDING"
    assert pending["recorded_paired_training_runtime_minutes"] == 10.0
    assert pending["peak_gpu_memory_bytes_by_arm"] == {
        "control": 20.0,
        "candidate": 30.0,
    }
    assert EVAL.resource_summary(observations, 10.0)["status"] == "PASS"
    assert EVAL.resource_summary(observations, 9.9)["status"] == "FAIL"
    assert EVAL.resource_summary(None, 10.0)["status"] == "STAGE_PENDING"


def evaluation_fixture(root: Path, *, validation_has_label: bool = False) -> tuple[Path, ...]:
    runtime = root / "runtime"
    baseline = root / "baseline"
    control = root / "control"
    candidate = root / "candidate"
    rows = [
        {
            "global_index": fold,
            "id": f"row-{fold}",
            "fold": fold,
            "category": TRAIN.FLAMMABLE if fold % 2 else "БАД",
            "label": fold % 2,
        }
        for fold in range(5)
    ]
    for fold in range(5):
        runtime_fold = runtime / f"fold{fold}"
        runtime_fold.mkdir(parents=True)
        write_jsonl(runtime_fold / "train.jsonl", [row for row in rows if row["fold"] != fold])
        validation = {key: value for key, value in rows[fold].items() if key != "label"}
        if validation_has_label and fold == 0:
            validation["label"] = rows[fold]["label"]
        write_jsonl(runtime_fold / "validation.jsonl", [validation])
        prediction = {
            "global_index": fold,
            "id": rows[fold]["id"],
            "fold": fold,
            "category": rows[fold]["category"],
            "score": float(rows[fold]["label"] * 2 - 1),
            "prediction": rows[fold]["label"],
        }
        baseline_fold = baseline / f"fold{fold}"
        baseline_fold.mkdir(parents=True)
        write_jsonl(baseline_fold / "predictions.jsonl", [prediction])
        for arm in (control, candidate):
            arm_fold = arm / f"fold{fold}"
            arm_fold.mkdir(parents=True)
            write_jsonl(arm_fold / "predictions.jsonl", [prediction])
            (arm_fold / "output_contract.json").write_text(
                json.dumps(
                    {
                        "outer_fold": fold,
                        "decision": "GO_EVALUATE",
                        "technical_smoke": False,
                    }
                ),
                encoding="utf-8",
            )
    return runtime, baseline, control, candidate


def test_labels_are_joined_from_cross_fold_registry_only_after_both_arms_frozen(
    tmp_path: Path,
):
    paths = evaluation_fixture(tmp_path)
    rows = EVAL.aligned_rows(*paths)
    assert [row["label"] for row in rows] == [0, 1, 0, 1, 0]
    (paths[3] / "fold0" / "output_contract.json").unlink()
    with pytest.raises(ValueError, match="paired arm is not frozen"):
        EVAL.aligned_rows(*paths)


def test_outer_validation_label_is_rejected_even_after_arms_are_frozen(tmp_path: Path):
    paths = evaluation_fixture(tmp_path, validation_has_label=True)
    with pytest.raises(ValueError, match="label-free"):
        EVAL.aligned_rows(*paths)


def test_validation_name_must_match_frozen_train_registry(tmp_path: Path):
    paths = evaluation_fixture(tmp_path)
    validation_path = paths[0] / "fold0" / "validation.jsonl"
    value = json.loads(validation_path.read_text(encoding="utf-8"))
    value["name"] = "топливо для зажигалки"
    write_jsonl(validation_path, [value])
    with pytest.raises(ValueError, match="cross-fold registry"):
        EVAL.aligned_rows(*paths)


def test_paired_contract_resource_values_are_forwarded(tmp_path: Path):
    paths = evaluation_fixture(tmp_path)
    for fold in range(5):
        for arm_index, root in enumerate(paths[2:]):
            contract_path = root / f"fold{fold}" / "output_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["runtime_minutes"] = float(fold + arm_index + 1)
            contract["peak_gpu_memory_bytes"] = float(100 + fold + arm_index)
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
    resources = EVAL.verify_paired_arms_frozen(paths[2], paths[3])
    assert len(resources["control"]["runtime_minutes"]) == 5
    assert max(resources["candidate"]["peak_gpu_memory_bytes"]) == 105.0

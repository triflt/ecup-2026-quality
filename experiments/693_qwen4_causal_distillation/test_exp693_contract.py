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
BUILD_CODE = load("build_code_bundle")
PRESET = load("build_preset")
REMOTE = load("remote_entrypoint")
RUNNER = load("run_all_folds")


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
        "schema_version": "exp691_fold_report_v1",
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
        "teacher_model_id": CONSUMER.TEACHERS["691"]["model_id"],
        "teacher_model_revision": CONSUMER.TEACHERS["691"]["revision"],
        "teacher_runtime_backend": CONSUMER.TEACHERS["691"]["backend"],
        "teacher_scope": "all",
        "folds": {str(fold): {} for fold in range(5)},
        "pooled": {
            "lighter_fuel_label_policy": {"rows": 27, "gold_negative": 27},
        },
        "gate": {
            "fold_wins": 5,
            "flammable_ap_delta": 0.1,
            "flammable_f1_delta": 0.1,
            "flammable_fn_nonincrease": True,
            "bad_byte_identical": True,
        },
        "artifact_bindings": bindings,
        "artifact_bindings_sha256": CONSUMER.canonical_sha256(bindings),
        "teacher_fold_report_self_sha256": report_hashes,
        "runtime_labels_read_after_teacher_terminal": True,
        "public_used": False,
        "sealed_rows": 0,
        "decision": "READY_FOR_TEACHER_COMPARISON",
    }
    acceptance["acceptance_sha256"] = CONSUMER.canonical_sha256(acceptance)
    acceptance_path = root / "student_consumer_acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")
    winner = {
        "schema_version": CONSUMER.WINNER_SCHEMA,
        "experiment_id": "692",
        "selected_teacher_experiment_id": "691",
        "selected_teacher_model_id": CONSUMER.TEACHERS["691"]["model_id"],
        "selected_teacher_model_revision": CONSUMER.TEACHERS["691"]["revision"],
        "selected_teacher_runtime_backend": CONSUMER.TEACHERS["691"]["backend"],
        "selected_teacher_layout": "root",
        "selected_acceptance_sha256": acceptance["acceptance_sha256"],
        "selected_acceptance_file_sha256": CONSUMER.sha256_file(acceptance_path),
        "selected_artifact_bindings_sha256": acceptance["artifact_bindings_sha256"],
        "student_methods_open": ["693", "694", "695"],
        "public_used": False,
        "sealed_rows": 0,
        "decision": "OPEN_WINNING_ALL_DATA_TEACHER_FOR_THREE_STUDENTS",
    }
    winner["winner_sha256"] = CONSUMER.canonical_sha256(winner)
    (root / "teacher_winner.json").write_text(json.dumps(winner), encoding="utf-8")
    return train, runtime_sha, acceptance_path, CONSUMER.sha256_file(acceptance_path)


def winner_args(acceptance_path: Path) -> tuple[Path, str]:
    path = acceptance_path.parent / "teacher_winner.json"
    return path, CONSUMER.sha256_file(path)


def convert_fixture_to_696(root: Path, acceptance_path: Path) -> tuple[Path, str]:
    fivefold = root / "fivefold"
    fivefold.mkdir()
    (root / "fold0").rename(fivefold / "fold0")
    report_path = fivefold / "fold0" / "report.json"
    report = json.loads(report_path.read_text())
    report.pop("report_sha256")
    report.update(
        {
            "schema_version": "exp696_fold_report_v1",
            "experiment_id": "696",
            "runtime_backend": "verified_fast_path",
        }
    )
    report["report_sha256"] = CONSUMER.canonical_sha256(report)
    report_path.write_text(json.dumps(report))
    acceptance = json.loads(acceptance_path.read_text())
    acceptance.pop("acceptance_sha256")
    acceptance.update(
        {
            "teacher_experiment_id": "696",
            "teacher_model_id": CONSUMER.TEACHERS["696"]["model_id"],
            "teacher_model_revision": CONSUMER.TEACHERS["696"]["revision"],
            "teacher_runtime_backend": CONSUMER.TEACHERS["696"]["backend"],
        }
    )
    acceptance["artifact_bindings"][0]["fold_report_file_sha256"] = CONSUMER.sha256_file(
        report_path
    )
    acceptance["artifact_bindings"][0]["fold_report_self_sha256"] = report["report_sha256"]
    acceptance["teacher_fold_report_self_sha256"][0] = report["report_sha256"]
    acceptance["artifact_bindings_sha256"] = CONSUMER.canonical_sha256(
        acceptance["artifact_bindings"]
    )
    acceptance["acceptance_sha256"] = CONSUMER.canonical_sha256(acceptance)
    acceptance_path.write_text(json.dumps(acceptance))
    winner = {
        "schema_version": CONSUMER.WINNER_SCHEMA,
        "experiment_id": "692",
        "selected_teacher_experiment_id": "696",
        "selected_teacher_model_id": CONSUMER.TEACHERS["696"]["model_id"],
        "selected_teacher_model_revision": CONSUMER.TEACHERS["696"]["revision"],
        "selected_teacher_runtime_backend": CONSUMER.TEACHERS["696"]["backend"],
        "selected_teacher_layout": "fivefold",
        "selected_acceptance_sha256": acceptance["acceptance_sha256"],
        "selected_acceptance_file_sha256": CONSUMER.sha256_file(acceptance_path),
        "selected_artifact_bindings_sha256": acceptance["artifact_bindings_sha256"],
        "student_methods_open": ["693", "694", "695"],
        "public_used": False,
        "sealed_rows": 0,
        "decision": "OPEN_WINNING_ALL_DATA_TEACHER_FOR_THREE_STUDENTS",
    }
    winner["winner_sha256"] = CONSUMER.canonical_sha256(winner)
    winner_path = root / "teacher_winner.json"
    winner_path.write_text(json.dumps(winner))
    return winner_path, CONSUMER.sha256_file(winner_path)


def test_actual_exp691_targets_merge_only_into_flammable(tmp_path: Path):
    train, runtime_sha, acceptance_path, acceptance_sha = teacher_fixture(
        tmp_path, with_acceptance=True
    )
    assert acceptance_path is not None and acceptance_sha is not None
    winner_path, winner_sha = winner_args(acceptance_path)
    enriched, binding = CONSUMER.load_fold(
        tmp_path,
        fold=0,
        train=train,
        runtime_contract_sha256=runtime_sha,
        require_evidence=False,
        acceptance_path=acceptance_path,
        expected_acceptance_file_sha256=acceptance_sha,
        winner_path=winner_path,
        expected_winner_file_sha256=winner_sha,
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
    winner_path, winner_sha = winner_args(acceptance_path)
    enriched, _ = CONSUMER.load_fold(
        tmp_path,
        fold=0,
        train=train,
        runtime_contract_sha256=runtime_sha,
        require_evidence=True,
        acceptance_path=acceptance_path,
        expected_acceptance_file_sha256=acceptance_sha,
        winner_path=winner_path,
        expected_winner_file_sha256=winner_sha,
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


def test_qwen38_winner_consumes_fivefold_layout(tmp_path: Path) -> None:
    train, runtime_sha, acceptance_path, _ = teacher_fixture(tmp_path, with_acceptance=True)
    assert acceptance_path is not None
    winner_path, winner_sha = convert_fixture_to_696(tmp_path, acceptance_path)
    enriched, binding = CONSUMER.load_fold(
        tmp_path,
        fold=0,
        train=train,
        runtime_contract_sha256=runtime_sha,
        require_evidence=True,
        acceptance_path=acceptance_path,
        expected_acceptance_file_sha256=CONSUMER.sha256_file(acceptance_path),
        winner_path=winner_path,
        expected_winner_file_sha256=winner_sha,
    )
    assert enriched[0]["teacher_score"] == 2.0
    assert binding["teacher_experiment_id"] == "696"


def test_winner_selected_acceptance_mismatch_is_rejected(tmp_path: Path) -> None:
    _, _, acceptance_path, _ = teacher_fixture(tmp_path, with_acceptance=True)
    assert acceptance_path is not None
    winner_path, _ = winner_args(acceptance_path)
    winner = json.loads(winner_path.read_text())
    winner.pop("winner_sha256")
    winner["selected_acceptance_sha256"] = "f" * 64
    winner["winner_sha256"] = CONSUMER.canonical_sha256(winner)
    winner_path.write_text(json.dumps(winner))
    with pytest.raises(ValueError, match="does not bind selected"):
        CONSUMER.verify_routed_acceptance(
            acceptance_path,
            expected_file_sha256=CONSUMER.sha256_file(acceptance_path),
            winner_path=winner_path,
            expected_winner_file_sha256=CONSUMER.sha256_file(winner_path),
        )


def test_causal_acceptance_requires_exact_file_sha(tmp_path: Path):
    train, runtime_sha, acceptance_path, _ = teacher_fixture(tmp_path, with_acceptance=True)
    assert acceptance_path is not None
    winner_path, winner_sha = winner_args(acceptance_path)
    with pytest.raises(ValueError, match="acceptance file SHA-256 mismatch"):
        CONSUMER.load_fold(
            tmp_path,
            fold=0,
            train=train,
            runtime_contract_sha256=runtime_sha,
            require_evidence=True,
            acceptance_path=acceptance_path,
            expected_acceptance_file_sha256="0" * 64,
            winner_path=winner_path,
            expected_winner_file_sha256=winner_sha,
        )


def test_rejected_exp692_acceptance_cannot_open_any_student(tmp_path: Path):
    _, _, acceptance_path, _ = teacher_fixture(tmp_path, with_acceptance=True)
    assert acceptance_path is not None
    winner_path, winner_sha = winner_args(acceptance_path)
    value = json.loads(acceptance_path.read_text(encoding="utf-8"))
    value.pop("acceptance_sha256")
    value["decision"] = "REJECT_TEACHER_TARGETS"
    value["acceptance_sha256"] = CONSUMER.canonical_sha256(value)
    acceptance_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="routed acceptance contract mismatch"):
        CONSUMER.verify_routed_acceptance(
            acceptance_path,
            expected_file_sha256=CONSUMER.sha256_file(acceptance_path),
            winner_path=winner_path,
            expected_winner_file_sha256=winner_sha,
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
        for index in range(27)
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
    report = EVAL.build_report(rows, expected_label_policy_rows=27)
    policy = report["pooled"]["label_policy_slice"]
    assert policy == {
        "rows": 27,
        "labels_zero": 27,
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
        for index in range(27)
    ]
    with pytest.raises(ValueError, match="disagrees with accepted exp692"):
        EVAL.build_report(rows, expected_label_policy_rows=27)


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
            contract = {
                "experiment_id": "693",
                "outer_fold": fold,
                "mode": "hard_bce_control" if arm == control else "causal_candidate",
                "decision": "GO_EVALUATE",
                "technical_smoke": False,
                "exp691_binding": evaluation_teacher_binding(fold, require_evidence=True),
                "artifacts": {
                    "predictions.jsonl": CONSUMER.sha256_file(arm_fold / "predictions.jsonl")
                },
            }
            contract["contract_sha256"] = CONSUMER.canonical_sha256(contract)
            (arm_fold / "output_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    return runtime, baseline, control, candidate


def evaluation_teacher_acceptance() -> dict:
    return {
        "teacher_experiment_id": "691",
        "acceptance_sha256": "a" * 64,
        "winner_gate_file_sha256": "e" * 64,
        "winner_gate": {"winner_sha256": "f" * 64},
        "artifact_bindings": [
            {
                "fold_report_file_sha256": f"{100 + fold:064x}",
                "fold_report_self_sha256": f"{200 + fold:064x}",
                "teacher_targets_sha256": f"{300 + fold:064x}",
                "evidence_sha256": f"{400 + fold:064x}",
            }
            for fold in range(5)
        ],
    }


def evaluation_teacher_binding(fold: int, *, require_evidence: bool) -> dict:
    acceptance = evaluation_teacher_acceptance()
    binding = acceptance["artifact_bindings"][fold]
    return {
        "teacher_experiment_id": "691",
        "routed_acceptance_experiment_id": "692",
        "fold": fold,
        "fold_report_file_sha256": binding["fold_report_file_sha256"],
        "fold_report_self_sha256": binding["fold_report_self_sha256"],
        "teacher_targets_sha256": binding["teacher_targets_sha256"],
        "evidence_sha256": binding["evidence_sha256"] if require_evidence else None,
        "routed_acceptance_sha256": "a" * 64,
        "routed_acceptance_file_sha256": "b" * 64,
        "winner_gate_sha256": "f" * 64,
        "winner_gate_file_sha256": "e" * 64,
        "scope": "all",
    }


def aligned_fixture_rows(paths: tuple[Path, ...]) -> list[dict]:
    return EVAL.aligned_rows(
        paths[0],
        paths[1],
        paths[3],
        candidate_experiment_id="693",
        expected_candidate_mode="causal_candidate",
        teacher_acceptance=evaluation_teacher_acceptance(),
        teacher_acceptance_file_sha256="b" * 64,
        teacher_winner_file_sha256="e" * 64,
        require_evidence=True,
    )


def test_labels_are_joined_only_after_candidate_and_teacher_lineage_are_frozen(
    tmp_path: Path,
):
    paths = evaluation_fixture(tmp_path)
    rows = aligned_fixture_rows(paths)
    assert [row["label"] for row in rows] == [0, 1, 0, 1, 0]
    (paths[3] / "fold0" / "output_contract.json").unlink()
    with pytest.raises(ValueError, match="candidate is not frozen"):
        aligned_fixture_rows(paths)


def test_outer_validation_label_is_rejected_after_candidate_is_frozen(tmp_path: Path):
    paths = evaluation_fixture(tmp_path, validation_has_label=True)
    with pytest.raises(ValueError, match="label-free"):
        aligned_fixture_rows(paths)


def test_validation_name_must_match_frozen_train_registry(tmp_path: Path):
    paths = evaluation_fixture(tmp_path)
    validation_path = paths[0] / "fold0" / "validation.jsonl"
    value = json.loads(validation_path.read_text(encoding="utf-8"))
    value["name"] = "топливо для зажигалки"
    write_jsonl(validation_path, [value])
    with pytest.raises(ValueError, match="cross-fold registry"):
        aligned_fixture_rows(paths)


def test_student_jobs_train_exactly_one_candidate_per_fold() -> None:
    assert RUNNER.MODES == ("causal_candidate",)
    hardneg_source = (HERE.parent / "694_qwen4_hardneg_curriculum/run_all_folds.py").read_text()
    rank_source = (HERE.parent / "695_qwen4_hard_anchored_listwise/run_all_folds.py").read_text()
    assert 'MODES = ("hardneg_candidate",)' in hardneg_source
    assert 'MODES = ("rank_candidate",)' in rank_source
    assert hardneg_source.count("subprocess.run(smoke_command, check=True)") == 1
    assert rank_source.count("subprocess.run(smoke_command, check=True)") == 1
    assert "if fold == 0 and not args.technical_smoke" in hardneg_source
    assert "if fold == 0 and not args.technical_smoke" in rank_source


def test_structured_memory_smoke_is_fail_closed_before_fivefold() -> None:
    contract = {
        "mode": "causal_candidate",
        "technical_smoke": True,
        "decision": "TECHNICAL_SMOKE_ONLY",
        "peak_gpu_memory_bytes": 60 * 1024**3,
        "changed_factor_smoke": {
            "eligible_grounded_rows": 1,
            "auxiliary_losses": {
                "sold_object": 1.0,
                "substance": 1.0,
                "relation": 1.0,
                "evidence_pointer": 1.0,
            },
        },
    }
    contract["contract_sha256"] = CONSUMER.canonical_sha256(contract)
    assert RUNNER.validate_memory_smoke_contract(contract) == 60 * 1024**3
    with pytest.raises(ValueError, match="memory smoke"):
        oversized = {**contract, "peak_gpu_memory_bytes": 76 * 1024**3}
        oversized.pop("contract_sha256")
        oversized["contract_sha256"] = CONSUMER.canonical_sha256(oversized)
        RUNNER.validate_memory_smoke_contract(oversized)


def test_primary_evaluation_is_frozen_production_route() -> None:
    rows = [
        {
            "fold": fold,
            "label": 1,
            "category": "БАД",
            "name": "",
            "baseline": 1,
            "control": 0,
            "candidate": 1,
            "baseline_score": 1.0,
            "control_score": -1.0,
            "candidate_score": 1.0,
            "singleton": False,
            "slices": [],
        }
        for fold in range(5)
    ]
    rows.extend(
        {
            "fold": index % 5,
            "label": 0,
            "category": TRAIN.FLAMMABLE,
            "name": f"топливо для зажигалки {index}",
            "baseline": 0,
            "control": 1,
            "candidate": 0,
            "baseline_score": -1.0,
            "control_score": 1.0,
            "candidate_score": -1.0,
            "singleton": False,
            "slices": [],
        }
        for index in range(27)
    )
    report = EVAL.build_report(rows, expected_label_policy_rows=27)
    assert report["primary_comparison"] == "candidate_routed_vs_frozen_production_route"
    assert report["pooled"]["categories"][TRAIN.FLAMMABLE]["control"]["fp"] == 0


def test_candidate_contract_resource_values_are_forwarded(tmp_path: Path):
    paths = evaluation_fixture(tmp_path)
    for fold in range(5):
        for arm_index, root in enumerate(paths[2:]):
            contract_path = root / f"fold{fold}" / "output_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["runtime_minutes"] = float(fold + arm_index + 1)
            contract["peak_gpu_memory_bytes"] = float(100 + fold + arm_index)
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
    resources = EVAL.verify_candidate_frozen(paths[3])
    assert resources["control"]["runtime_minutes"] == []
    assert max(resources["candidate"]["peak_gpu_memory_bytes"]) == 105.0


def test_evaluation_binds_ordered_five_candidate_outputs(tmp_path: Path):
    paths = evaluation_fixture(tmp_path)
    bindings = EVAL.candidate_output_bindings(
        paths[3],
        experiment_id="693",
        expected_mode="causal_candidate",
        teacher_acceptance=evaluation_teacher_acceptance(),
        teacher_acceptance_file_sha256="b" * 64,
        teacher_winner_file_sha256="e" * 64,
        require_evidence=True,
    )
    assert [(row["fold"], row["arm"]) for row in bindings] == [
        (fold, "candidate") for fold in range(5)
    ]
    assert all(len(row["predictions_sha256"]) == 64 for row in bindings)
    report = EVAL.bind_evaluation_provenance(
        {"decision": "REJECT_CANDIDATE", "folds": {}, "pooled": {}},
        frozen_method="causal",
        teacher_acceptance=evaluation_teacher_acceptance(),
        teacher_acceptance_file_sha256="b" * 64,
        runtime_bundle_sha256="c" * 64,
        baseline_bundle_sha256="d" * 64,
        candidate_root=paths[3],
        candidate_bindings=bindings,
    )
    body = dict(report)
    digest = body.pop("evaluation_sha256")
    assert digest == CONSUMER.canonical_sha256(body)
    assert report["method"] == "causal" and report["experiment_id"] == "693"
    assert len(report["candidate_output_bindings"]) == 5


def preset_args(output_dir: Path):
    _, _, acceptance_path, acceptance_sha = teacher_fixture(
        output_dir.parent / "teacher", with_acceptance=True
    )
    assert acceptance_path is not None and acceptance_sha is not None
    winner_path, winner_sha = winner_args(acceptance_path)
    return __import__("argparse").Namespace(
        project="example-project",
        region="example-region",
        image="example/image:immutable",
        h100_flavor="gpu-h100-1-80",
        time_limit="20h0m",
        preemption="forbidden",
        input_bucket="example-input-bucket",
        output_bucket="example-output-bucket",
        code_bundle_src="/example/code",
        code_bundle_file="code.tar.gz",
        code_bundle_sha256="a" * 64,
        code_revision="b" * 40,
        runtime_bundle_src="/example/runtime",
        runtime_bundle_file="runtime.tar.gz",
        runtime_bundle_sha256="c" * 64,
        baseline_bundle_src="/example/baseline",
        baseline_bundle_file="baseline.tar.gz",
        baseline_bundle_sha256="d" * 64,
        qwen27_output_src="/example/qwen27-terminal",
        acceptance_output_src="/example/exp692",
        acceptance_file="acceptance.json",
        acceptance_sha256=acceptance_sha,
        local_acceptance=acceptance_path,
        winner_output_src="/example/exp692-winner",
        winner_file="teacher_winner.json",
        winner_sha256=winner_sha,
        local_winner=winner_path,
        vendor_bundle_src="/example/vendor",
        vendor_bundle_file="vendor.zip",
        vendor_bundle_sha256="f" * 64,
        model_mrid=f"example/qwen/{PRESET.MODEL_REVISION}",
        output_prefix="/example/student-output/run-1",
        output_dir=output_dir,
    )


def test_real_preset_builder_emits_exactly_three_secret_free_job_shapes(tmp_path: Path):
    output = tmp_path / "presets"
    result = PRESET.build_all(preset_args(output))
    assert result["jobs"] == ["qwen4-causal", "qwen4-hardneg", "qwen4-rank"]
    assert sorted(path.name for path in output.iterdir()) == result["files"]
    for job_name, method in PRESET.JOBS:
        payload = (output / f"{job_name}.yaml").read_text(encoding="utf-8")
        assert f"generate_name: {job_name}" in payload
        assert "type: model_registry" in payload
        assert PRESET.MODEL_REVISION in payload
        assert f"--method {method}" in payload
        assert 'preemption: "forbidden"' in payload
        assert "--submission-limit-minutes" not in payload
        assert 'file: "acceptance.json"' in payload
        assert 'dst: "/work/input/acceptance/acceptance.json"' in payload
        assert "upload_policies" not in payload
        assert payload.count("type: s3msk") == 8
        assert "project:" not in payload


def test_preset_builder_rejects_placeholder_acceptance_before_output(tmp_path: Path):
    args = preset_args(tmp_path / "presets")
    args.acceptance_sha256 = "e" * 64
    with pytest.raises(ValueError, match="acceptance file SHA-256 mismatch"):
        PRESET.build_all(args)
    assert not args.output_dir.exists()


def test_bundle_whitelist_is_exact_and_label_free():
    assert BUILD_CODE.BUNDLE_PATHS == (
        "experiments/693_qwen4_causal_distillation",
        "experiments/694_qwen4_hardneg_curriculum",
        "experiments/695_qwen4_hard_anchored_listwise",
        "experiments/645_qwen_scale_2x3_gate/grid_contract.py",
        "experiments/645_qwen_scale_2x3_gate/train_lora.py",
        "experiments/645_qwen_scale_2x3_gate/frozen_spec.json",
        "experiments/641_qwen35_4b_class_only_lora/frozen_spec.json",
    )
    assert not any(path.startswith("validation/") for path in BUILD_CODE.BUNDLE_PATHS)


def test_remote_archive_extraction_rejects_path_escape(tmp_path: Path):
    import hashlib
    import zipfile

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as destination:
        destination.writestr("../escape", "forbidden")
    with pytest.raises(ValueError, match="unsafe archive member"):
        REMOTE.extract_archive(
            archive,
            tmp_path / "output",
            hashlib.sha256(archive.read_bytes()).hexdigest(),
        )


def test_remote_entrypoint_uses_verified_archive_root_layouts():
    source = Path(REMOTE.__file__).read_text(encoding="utf-8")
    assert 'runtime = work / "runtime_bundle" / "runtime"' in source
    assert 'baseline = work / "baseline_bundle"' in source
    assert 'vendor = work / "vendor_bundle"' in source
    assert 'baseline_bundle" / "baseline"' not in source
    assert 'vendor_bundle" / "vendor"' not in source
    assert 'vendor / "peft" / "__init__.py"' in source
    assert "--submission-limit-minutes" not in source


def test_train_contract_records_measured_cuda_peak(monkeypatch):
    calls: list[str] = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        reset_peak_memory_stats=lambda: calls.append("reset"),
        max_memory_allocated=lambda: 123456,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    TRAIN.reset_cuda_peak_memory()
    peak = TRAIN.measured_cuda_peak_memory_bytes()
    assert calls == ["reset"]
    assert isinstance(peak, int) and peak == 123456
    assert '"peak_gpu_memory_bytes": peak_gpu_memory_bytes' in __import__("inspect").getsource(
        TRAIN.run
    )


def test_fold_output_is_committed_by_atomic_rename(tmp_path: Path):
    staging = tmp_path / ".incomplete" / "control" / "fold0"
    staging.mkdir(parents=True)
    (staging / "output_contract.json").write_text("{}", encoding="utf-8")
    final = tmp_path / "control" / "fold0"
    RUNNER.commit_fold_output(staging, final)
    assert (final / "output_contract.json").is_file()
    assert not staging.exists()

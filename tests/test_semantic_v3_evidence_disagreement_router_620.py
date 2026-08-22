from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "620_semantic_v3_evidence_disagreement_router"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_LOCAL_IMPORT_NAMES = ("contract", "smoke_evidence_head", "score_hypothesis_shard")
_previous_modules = {name: sys.modules.get(name) for name in _LOCAL_IMPORT_NAMES}
try:
    contract = _load("exp620_contract_test", EXP / "contract.py")
    sys.modules["contract"] = contract
    builder = _load("exp620_builder_test", EXP / "build_evidence_manifest.py")
    null_control = _load("exp620_null_test", EXP / "evaluate_null_control.py")
    smoke = _load("exp620_smoke_test", EXP / "smoke_evidence_head.py")
    sys.modules["smoke_evidence_head"] = smoke
    scorer = _load("exp620_scorer_test", EXP / "score_hypothesis_shard.py")
    sys.modules["score_hypothesis_shard"] = scorer
    runtime_builder = _load("exp620_runtime_builder_test", EXP / "prepare_feature_runtime.py")
    router_eval = _load("exp620_router_eval_test", EXP / "evaluate_router.py")
    full_eval = _load("exp620_full_eval_test", EXP / "evaluate_full.py")
finally:
    for _name, _previous in _previous_modules.items():
        if _previous is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _previous


def _synthetic_registry(tmp_path: Path) -> tuple[Path, Path]:
    ids = [f"dev-{index}" for index in range(5)]
    categories = ["БАД", "БАД", "Легковоспламеняющиеся", "Легковоспламеняющиеся", "БАД"]
    folds = pd.DataFrame(
        {
            "id": ids + ["sealed-sentinel"],
            "category": categories + ["БАД"],
            "label": [1, 0, 1, 0, 1, 0],
            "semantic_component": [f"component-{index}" for index in range(6)],
            "component_size": [1] * 6,
            "split": ["development"] * 5 + ["sealed_holdout"],
            "development_fold": [0, 1, 2, 3, 4, -1],
        }
    )
    data = pd.DataFrame(
        {
            "id": ids,
            "name": [
                "БАД к пище с витамином C",
                "Этот товар не является БАД",
                "Газовый баллон с бутаном",
                "Горелка без газового баллона",
                "Food supplement magnesium",
            ],
            "description": [""] * 5,
            "category": categories,
            # A sentinel supervision column proves the builder uses an explicit
            # safe-column allow-list rather than loading the full frame.
            "label": ["must-not-be-read"] * 5,
        }
    )
    folds_path = tmp_path / "folds.csv"
    data_path = tmp_path / "development.csv"
    folds.to_csv(folds_path, index=False)
    data.to_csv(data_path, index=False)
    return folds_path, data_path


def test_synthetic_smoke_helpers_are_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "models"
    model = root / "nested"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert smoke.resolve_model_dir(root) == model
    assert set(smoke.metadata_checksums(model)) == {"config.json", "tokenizer_config.json"}
    smoke.validate_offsets("абв", [(0, 1), (1, 3)])
    with pytest.raises(ValueError, match="non-monotonic"):
        smoke.validate_offsets("абв", [(1, 2), (0, 1)])
    with pytest.raises(FileExistsError):
        smoke.write_report(tmp_path, {"status": "passed"})


def test_synthetic_smoke_report_contains_no_input_text(tmp_path: Path) -> None:
    output = tmp_path / "new-output"
    path = smoke.write_report(
        output,
        {
            "status": "passed",
            "synthetic_input_only": True,
            "competition_rows_loaded": 0,
            "labels_loaded": False,
            "sealed_rows_loaded": 0,
        },
    )
    payload = path.read_text(encoding="utf-8")
    assert smoke.RUSSIAN_PROBE not in payload
    assert json.loads(payload)["sealed_rows_loaded"] == 0


def test_hypothesis_scorer_reads_only_safe_columns(tmp_path: Path) -> None:
    folds_path, data_path = _synthetic_registry(tmp_path)
    rows = scorer.load_category_rows(
        data_path=data_path,
        folds_path=folds_path,
        category="БАД",
        enforce_frozen=False,
        expected_rows=5,
    )
    assert tuple(rows.columns) == scorer.SAFE_DATA_COLUMNS
    assert rows["id"].tolist() == ["dev-0", "dev-1", "dev-4"]
    assert "label" not in rows


def test_hypothesis_order_and_selection_are_frozen() -> None:
    spec = json.loads((EXP / "frozen_router_spec.json").read_text(encoding="utf-8"))
    ids, grouped = scorer.ordered_hypotheses(spec)
    assert len(ids) == 10
    assert ids[:5] == [item["id"] for item in grouped["БАД"]]
    assert ids[5:] == [item["id"] for item in grouped["Легковоспламеняющиеся"]]
    category, item = scorer.select_hypothesis(spec, "FL_EMPTY_EQUIPMENT")
    assert category == "Легковоспламеняющиеся"
    assert item["supports_verdict"] == 0


def test_hypothesis_shard_has_no_supervision(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "id": ["a", "b"],
            "category": ["БАД", "Легковоспламеняющиеся"],
            "name": ["x", "y"],
            "description": ["", ""],
        }
    )
    spec = EXP / "frozen_router_spec.json"
    frozen = json.loads(spec.read_text(encoding="utf-8"))
    category, hypothesis = scorer.select_hypothesis(frozen, "BAD_DIRECT_MARKING")
    output = tmp_path / "output"
    report = scorer.write_outputs(
        output_dir=output,
        rows=rows,
        hypothesis=hypothesis,
        category=category,
        scores=np.full(2, 0.5, dtype=np.float32),
        spec_path=spec,
        runtime={
            "forward_batches": 0,
            "scoring_seconds": 0.0,
            "peak_gpu_memory_bytes": 0,
            "zero_token_id": 0,
            "one_token_id": 1,
        },
        backbone_decision={
            "status": "accepted",
            "model_id": "synthetic/model",
            "revision": "synthetic-revision",
            "files_sha256": {"config.json": "0" * 64},
        },
    )
    with np.load(output / "hypothesis_score_shard.npz", allow_pickle=False) as bundle:
        assert "labels" not in bundle.files
        assert bundle["hypothesis_scores"].shape == (2,)
        assert bundle["ids"].dtype.kind == "U"
        assert bundle["categories"].dtype.kind == "U"
        assert str(bundle["scorer_sha256"]) == contract.sha256_file(
            EXP / "score_hypothesis_shard.py"
        )
    assert report["scorer_sha256"] == contract.sha256_file(EXP / "score_hypothesis_shard.py")
    assert report["labels_loaded"] is False
    assert report["sealed_rows_loaded"] == 0


def test_feature_runtime_is_physically_label_free(tmp_path: Path) -> None:
    folds_path, data_path = _synthetic_registry(tmp_path)
    ids = np.asarray([f"dev-{index}" for index in range(5)])
    categories = np.asarray(
        ["БАД", "БАД", "Легковоспламеняющиеся", "Легковоспламеняющиеся", "БАД"]
    )
    visual = tmp_path / "visual.npz"
    route = tmp_path / "route.npz"
    np.savez_compressed(
        visual,
        ids=ids,
        categories=categories,
        labels=np.ones(5, dtype=np.int8),
        robust_base_score=np.linspace(0.1, 0.9, 5),
        qwen3vl_score=np.linspace(-1, 1, 5),
    )
    np.savez_compressed(
        route,
        ids=ids,
        categories=categories,
        labels=np.ones(5, dtype=np.int8),
        original_route_predictions=np.asarray([0, 1, 0, 1, 0], dtype=np.int8),
    )
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        "".join(
            json.dumps(
                {
                    "id": value,
                    "candidate_for_0": None,
                    "candidate_for_1": None,
                }
            )
            + "\n"
            for value in ids
        ),
        encoding="utf-8",
    )
    evidence_audit = tmp_path / "evidence_audit.json"
    evidence_audit.write_text(
        json.dumps(
            {
                "decision": "GO",
                "output_sha256": {evidence.name: contract.sha256_file(evidence)},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "runtime"
    audit = runtime_builder.prepare_runtime(
        data_path=data_path,
        folds_path=folds_path,
        visual_bundle_path=visual,
        route_predictions_path=route,
        evidence_manifest_path=evidence,
        evidence_audit_path=evidence_audit,
        output_dir=output,
        enforce_frozen=False,
        expected_rows=5,
    )
    safe_data = pd.read_csv(output / "development_feature_data.csv")
    safe_folds = pd.read_csv(output / "development_feature_folds.csv")
    assert tuple(safe_data.columns) == scorer.SAFE_DATA_COLUMNS
    assert "label" not in safe_folds.columns
    assert audit["labels_present"] is False
    assert audit["sealed_rows_present"] == 0
    with np.load(output / "selector_components.npz", allow_pickle=False) as selector:
        assert "labels" not in selector.files
        assert selector["categories"].dtype.kind == "U"
        assert set(selector.files) == {
            "ids",
            "categories",
            "robust_base_score",
            "qwen3vl_score",
            "baseline_predictions",
        }
    scorer.validate_runtime_inputs(
        audit_path=output / "runtime_audit.json",
        data_path=output / "development_feature_data.csv",
        folds_path=output / "development_feature_folds.csv",
        selector_path=output / "selector_components.npz",
        evidence_path=output / "development_evidence_manifest.jsonl",
        enforce_frozen=False,
    )
    with pytest.raises(FileExistsError):
        runtime_builder.prepare_runtime(
            data_path=data_path,
            folds_path=folds_path,
            visual_bundle_path=visual,
            route_predictions_path=route,
            evidence_manifest_path=evidence,
            evidence_audit_path=evidence_audit,
            output_dir=output,
            enforce_frozen=False,
            expected_rows=5,
        )


def test_label_independent_selector_requires_compatible_retained_span(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "id": ["a", "b"],
            "category": ["БАД", "БАД"],
            "name": ["БАД к пище", "Витамины"],
            "description": ["", "Не является БАД"],
        }
    )
    selector = {
        "ids": np.asarray(["a", "b"]),
        "categories": np.asarray(["БАД", "БАД"]),
        "robust_base_score": np.asarray([0.8, 0.2]),
        "qwen3vl_score": np.asarray([1.0, -1.0]),
        "baseline_predictions": np.asarray([0, 1], dtype=np.int8),
    }
    evidence = [
        {
            "id": "a",
            "candidate_for_0": None,
            "candidate_for_1": {
                "concept": "BAD_EXPLICIT_MARKING",
                "exact_surface_span": "БАД",
            },
        },
        {
            "id": "b",
            "candidate_for_0": {
                "concept": "BAD_EXPLICIT_NEGATION",
                "exact_surface_span": "не является БАД",
            },
            "candidate_for_1": None,
        },
    ]
    spec = json.loads((EXP / "frozen_router_spec.json").read_text(encoding="utf-8"))
    _, hypothesis = scorer.select_hypothesis(spec, "BAD_DIRECT_MARKING")
    selected, audit = scorer.select_candidate_rows(
        rows=rows,
        selector=selector,
        evidence_rows=evidence,
        hypothesis=hypothesis,
        compatible_concepts=spec["hypothesis_concept_compatibility"]["BAD_DIRECT_MARKING"],
    )
    assert selected["id"].tolist() == ["a"]
    assert audit["span_retained_in_prompt"] == 1


def test_screen_metrics_require_both_frozen_folds_to_improve() -> None:
    labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    categories = np.asarray(["БАД", "БАД", "Легковоспламеняющиеся", "Легковоспламеняющиеся"] * 2)
    folds = np.asarray([0, 0, 0, 0, 3, 3, 3, 3], dtype=np.int8)
    baseline = np.asarray([0, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    candidate = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    result = router_eval.screen_metrics(
        labels=labels,
        categories=categories,
        folds=folds,
        baseline=baseline,
        candidate=candidate,
    )
    assert result["fold_deltas"]["0"] > 0
    assert result["fold_deltas"]["3"] == 0
    assert result["passed"] is False


def test_grouped_bootstrap_resamples_semantic_components() -> None:
    labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    categories = np.asarray(["БАД", "БАД", "Легковоспламеняющиеся", "Легковоспламеняющиеся"] * 2)
    components = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    baseline = np.asarray([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    candidate = labels.copy()
    result = full_eval.grouped_component_bootstrap(
        labels=labels,
        categories=categories,
        components=components,
        baseline=baseline,
        candidate=candidate,
        repeats=200,
        seed=7,
        batch_size=32,
    )
    assert result["unit"] == "semantic_component"
    assert result["components"] == 4
    assert result["probability_delta_positive"] == 1.0


def test_full_audit_refuses_to_release_when_screen_failed(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    screen.mkdir()
    (screen / "screen_metrics.json").write_text(
        json.dumps({"status": "rejected_at_screen"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="frozen screen passed"):
        full_eval.evaluate(
            route_predictions_path=tmp_path / "unused.npz",
            screen_dir=screen,
            development_data_path=tmp_path / "unused.csv",
            output_dir=tmp_path / "output",
            bootstrap_repeats=10,
        )


def test_frozen_spec_has_no_teacher_or_concrete_backbone() -> None:
    spec = json.loads((EXP / "frozen_router_spec.json").read_text(encoding="utf-8"))
    decision = json.loads(
        (EXP / "backbone_compatibility_decision.template.json").read_text(encoding="utf-8")
    )
    hypotheses = [item for values in spec["hypotheses"].values() for item in values]
    assert len(hypotheses) == 10
    assert len({item["id"] for item in hypotheses}) == 10
    assert spec["teacher"]["used"] is False
    assert spec["teacher"]["pseudo_labels_used"] is False
    assert spec["router"]["uses_old_190_260_400_oof_as_features"] is False
    assert decision["status"] == "pending"
    assert decision["model_id"] is None
    assert decision["revision"] is None


def test_label_free_manifest_has_exact_offsets_and_no_supervision(tmp_path: Path) -> None:
    folds_path, data_path = _synthetic_registry(tmp_path)
    output_dir = tmp_path / "evidence"
    audit = builder.build_manifest(
        development_data_path=data_path,
        folds_path=folds_path,
        spec_path=EXP / "frozen_router_spec.json",
        output_dir=output_dir,
        enforce_frozen=False,
        expected_rows=5,
    )
    rows = [
        json.loads(line)
        for line in (output_dir / "development_evidence_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert audit["labels_loaded"] is False
    assert audit["sealed_rows_in_inputs"] == 0
    assert audit["sealed_rows_in_outputs"] == 0
    assert audit["exact_offset_failures"] == 0
    assert audit["forbidden_flip_evidence"] == 0
    assert len(rows) == 5
    assert [row["id"] for row in rows] == [f"dev-{index}" for index in range(5)]
    for row in rows:
        assert not {"label", "labels", "gold", "target", "targets"} & set(row)
        assert "name" not in row and "description" not in row
        for verdict in (0, 1):
            candidate = row[f"candidate_for_{verdict}"]
            if candidate is not None:
                assert candidate["exact_surface_span"]
                assert candidate["surface_start"] < candidate["surface_end"]
                assert candidate["concept"] not in {
                    "BAD_TEXT_MARKING_NOT_FOUND",
                    "FL_QUALIFYING_ITEM_NOT_FOUND",
                }


def test_manifest_rejects_any_non_development_id(tmp_path: Path) -> None:
    folds_path, data_path = _synthetic_registry(tmp_path)
    data = pd.read_csv(data_path, dtype={"id": str})
    data.loc[len(data)] = {
        "id": "sealed-sentinel",
        "category": "БАД",
        "name": "irrelevant",
        "description": "",
        "label": "must-not-be-read",
    }
    data.to_csv(data_path, index=False)
    with pytest.raises(ValueError, match="exactly development IDs"):
        builder.build_manifest(
            development_data_path=data_path,
            folds_path=folds_path,
            spec_path=EXP / "frozen_router_spec.json",
            output_dir=tmp_path / "evidence",
            enforce_frozen=False,
            expected_rows=5,
        )


def test_pending_backbone_decision_is_rejected_before_feature_read(tmp_path: Path) -> None:
    registry = pd.DataFrame({"id": ["dev-0"]})
    with pytest.raises(RuntimeError, match="not accepted"):
        contract.validate_feature_bundle(
            tmp_path / "does-not-exist.npz",
            registry=registry,
            backbone_decision_path=EXP / "backbone_compatibility_decision.template.json",
        )


def test_null_control_refuses_incomplete_600_without_creating_output(tmp_path: Path) -> None:
    metrics_600 = tmp_path / "metrics600.json"
    metrics_600.write_text(
        json.dumps(
            {
                "experiment_id": "600",
                "status": "training_running",
                "sealed_holdout_used": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "null"
    with pytest.raises(RuntimeError, match="not complete"):
        null_control.evaluate_null_control(
            folds_path=tmp_path / "absent-folds.csv",
            metrics_600_path=metrics_600,
            metrics_601_path=tmp_path / "absent-601.json",
            qwen35_bundle_path=tmp_path / "absent-600.npz",
            qwen35_contract_path=tmp_path / "absent-600-contract.json",
            visual_bundle_path=tmp_path / "absent-601.npz",
            visual_contract_path=tmp_path / "absent-601-contract.json",
            evidence_manifest_path=tmp_path / "absent.jsonl",
            evidence_audit_path=tmp_path / "absent-audit.json",
            output_dir=output,
        )
    assert not output.exists()


def test_complete_synthetic_null_control_is_bit_exact_and_label_free(tmp_path: Path) -> None:
    folds_path, data_path = _synthetic_registry(tmp_path)
    evidence_dir = tmp_path / "evidence"
    evidence_audit = builder.build_manifest(
        development_data_path=data_path,
        folds_path=folds_path,
        spec_path=EXP / "frozen_router_spec.json",
        output_dir=evidence_dir,
        enforce_frozen=False,
        expected_rows=5,
    )
    # Coverage is diagnostic in this tiny fixture; the production 15% gate remains
    # frozen in the real spec. The null test only needs a provenance-complete manifest.
    evidence_audit["decision"] = "GO"
    (evidence_dir / "evidence_manifest_audit.json").write_text(
        json.dumps(evidence_audit), encoding="utf-8"
    )

    registry = contract.load_development_registry(folds_path, enforce_frozen=False, expected_rows=5)
    ids = np.asarray(registry["id"].astype(str).tolist(), dtype=str)
    categories = np.asarray(registry["category"].astype(str).tolist(), dtype=str)
    folds = registry["development_fold"].to_numpy(np.int8)
    components = np.asarray(registry["semantic_component"].astype(str).tolist(), dtype=str)
    ranks = np.linspace(0.05, 0.95, 5, dtype=np.float32)

    visual_path = tmp_path / "visual.npz"
    np.savez_compressed(
        visual_path,
        ids=ids,
        categories=categories,
        folds=folds,
        semantic_components=components,
        robust_base_rank=ranks,
        qwen3vl_rank=ranks[::-1],
    )
    visual_contract = tmp_path / "visual-contract.json"
    visual_contract.write_text(
        json.dumps(
            {
                "status": "complete",
                "sealed_rows_in_outputs": 0,
                "output_sha256": {visual_path.name: contract.sha256_file(visual_path)},
            }
        ),
        encoding="utf-8",
    )
    qwen_path = tmp_path / "qwen.npz"
    np.savez_compressed(
        qwen_path,
        ids=ids,
        categories=categories,
        folds=folds,
        semantic_components=components,
        qwen35_original_rank=ranks,
        qwen35_specialist_rank=ranks[::-1],
    )
    qwen_contract = tmp_path / "qwen-contract.json"
    qwen_contract.write_text(
        json.dumps(
            {
                "status": "complete",
                "validation_version": "semantic_family_v3",
                "development_rows": 5,
                "sealed_rows_in_outputs": 0,
                "completed_folds": [0, 1, 2, 3, 4],
                "components": ["original", "specialist"],
                "output_sha256": {qwen_path.name: contract.sha256_file(qwen_path)},
            }
        ),
        encoding="utf-8",
    )
    metrics_paths = []
    for experiment_id in ("600", "601"):
        path = tmp_path / f"metrics-{experiment_id}.json"
        path.write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "status": "complete",
                    "sealed_holdout_used": False,
                }
            ),
            encoding="utf-8",
        )
        metrics_paths.append(path)

    output_dir = tmp_path / "null"
    audit = null_control.evaluate_null_control(
        folds_path=folds_path,
        metrics_600_path=metrics_paths[0],
        metrics_601_path=metrics_paths[1],
        qwen35_bundle_path=qwen_path,
        qwen35_contract_path=qwen_contract,
        visual_bundle_path=visual_path,
        visual_contract_path=visual_contract,
        evidence_manifest_path=evidence_dir / "development_evidence_manifest.jsonl",
        evidence_audit_path=evidence_dir / "evidence_manifest_audit.json",
        output_dir=output_dir,
        enforce_frozen=False,
        expected_rows=5,
    )
    assert audit["changed_predictions"] == 0
    assert audit["labels_loaded"] is False
    with np.load(output_dir / "null_router_output.npz", allow_pickle=False) as output:
        assert np.array_equal(output["baseline_predictions"], output["candidate_predictions"])
        assert "labels" not in output.files

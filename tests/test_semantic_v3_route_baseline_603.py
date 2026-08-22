from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "603_semantic_v3_route_baseline"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load("exp603_evaluator_test", EXP / "evaluate.py")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_prediction_dir(
    directory: Path,
    *,
    experiment: str,
    fold: int,
    ids: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    logits: np.ndarray,
    component: str | None = None,
    seed: int | None = None,
) -> None:
    directory.mkdir(parents=True)
    predictions = directory / "lora_holdout_predictions.csv"
    pd.DataFrame(
        {
            "id": ids,
            "category": categories,
            "label": labels,
            "fold": fold,
            "lora_score": logits,
        }
    ).to_csv(predictions, index=False)
    (directory / "adapter.zip").write_bytes(f"adapter-{experiment}-{fold}".encode())
    selection_name = (
        "selection_audit.runtime.json"
        if experiment == "600"
        else "seed_selection_audit.runtime.json"
    )
    _write_json(directory / selection_name, {"decision": "GO"})
    common = {
        "experiment_id": experiment,
        "outer_fold": fold,
        "decision": "GO",
        "sealed_rows_in_predictions": 0,
        "prediction_rows": len(ids),
        "predictions_sha256": evaluator.sha256_file(predictions),
        "adapter_sha256": evaluator.sha256_file(directory / "adapter.zip"),
        "selection_audit_sha256": evaluator.sha256_file(directory / selection_name),
    }
    if experiment == "600":
        report_path = directory / "lora_holdout_report.json"
        _write_json(report_path, {"fold": fold})
        common.update(
            {
                "protocol_version": "semantic_family_v3",
                "component": component,
                "sealed_rows_used_for_threshold": 0,
                "sealed_rows_used_for_evaluation": 0,
                "parent_report_sha256": evaluator.sha256_file(report_path),
                "protocol_audit_sha256": "0" * 64,
                "prediction_ids_sha256": evaluator.canonical_sha256(
                    ids.astype(str).tolist()
                ),
            }
        )
        contract_name = "output_contract.runtime.json"
    else:
        common.update(
            {
                "protocol_version": "semantic_family_v3_seed_variance_v1",
                "seed": seed,
                "protocol_input_audit_sha256": "0" * 64,
            }
        )
        contract_name = "seed_output_contract.runtime.json"
    common["contract_sha256"] = evaluator.canonical_sha256(common)
    _write_json(directory / contract_name, common)


def _fixture(tmp_path: Path) -> dict:
    ids = np.asarray([f"dev-{index}" for index in range(10)], dtype=str)
    categories = np.asarray(["БАД", "Легковоспламеняющиеся"] * 5, dtype=str)
    folds = np.repeat(np.arange(5, dtype=np.int8), 2)
    labels = np.asarray([1, 0, 1, 1, 0, 0, 1, 0, 0, 1], dtype=np.int8)
    components = np.asarray([f"component-{index}" for index in range(10)], dtype=str)
    robust = np.asarray([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.7, 0.2, 0.4, 0.8], np.float32)
    visual_rank = np.asarray([0.7, 0.2, 0.9, 0.6, 0.1, 0.4, 0.8, 0.3, 0.2, 0.9], np.float32)

    visual_path = tmp_path / "visual.npz"
    np.savez_compressed(
        visual_path,
        ids=ids,
        labels=labels,
        categories=categories,
        folds=folds,
        semantic_components=components,
        robust_base_rank=robust,
        qwen3vl_rank=visual_rank,
    )
    visual_contract = tmp_path / "visual-contract.json"
    _write_json(
        visual_contract,
        {
            "version": "semantic_v3_visual_base_components_v1",
            "status": "complete",
            "development_rows": 10,
            "sealed_rows_in_outputs": 0,
            "output_sha256": {visual_path.name: evaluator.sha256_file(visual_path)},
        },
    )
    folds_path = tmp_path / "folds.csv"
    pd.DataFrame(
        {
            "id": ids,
            "category": categories,
            "label": labels,
            "semantic_component": components,
            "component_size": 1,
            "split": "development",
            "development_fold": folds,
        }
    ).to_csv(folds_path, index=False)

    original_logits = np.asarray([3, -3, 2, 1, -1, -2, 3, -2, -1, 2], np.float64)
    specialist_logits = np.asarray([3, -4, 2, 3, -1, -3, 3, -4, -1, 3], np.float64)
    seed_logits = {
        31415: original_logits + np.asarray([0, 1, -1, 1, 0, 1, -1, 0, 1, -1]),
        271828: original_logits + np.asarray([1, 0, 0, -1, 1, 0, 0, 1, -1, 0]),
        161803: original_logits + np.asarray([-1, 0, 1, 0, -1, 0, 1, -1, 0, 1]),
    }
    specs600: list[list[str]] = []
    specs602: list[list[str]] = []
    for component, logits in (
        ("original", original_logits),
        ("specialist", specialist_logits),
    ):
        for fold in range(5):
            mask = folds == fold
            directory = tmp_path / "exp600" / component / f"fold{fold}"
            _write_prediction_dir(
                directory,
                experiment="600",
                component=component,
                fold=fold,
                ids=ids[mask],
                labels=labels[mask],
                categories=categories[mask],
                logits=logits[mask],
            )
            specs600.append([component, str(fold), str(directory)])
    for seed, logits in seed_logits.items():
        for fold in range(5):
            mask = folds == fold
            directory = tmp_path / "exp602" / str(seed) / f"fold{fold}"
            _write_prediction_dir(
                directory,
                experiment="602",
                seed=seed,
                fold=fold,
                ids=ids[mask],
                labels=labels[mask],
                categories=categories[mask],
                logits=logits[mask],
            )
            specs602.append([str(seed), str(fold), str(directory)])

    protocol = json.loads((EXP / "frozen_protocol.json").read_text(encoding="utf-8"))
    protocol["development_rows"] = 10
    visual = {
        "ids": ids,
        "labels": labels,
        "categories": categories,
        "folds": folds,
        "semantic_components": components,
        "robust_base_rank": robust,
        "qwen3vl_rank": visual_rank,
    }
    original_rank = evaluator.fold_category_ranks(
        evaluator.sigmoid(original_logits), folds, categories
    )
    specialist_rank = evaluator.fold_category_ranks(
        evaluator.sigmoid(specialist_logits), folds, categories
    )
    mean_probability = np.mean(
        np.stack(
            [evaluator.sigmoid(original_logits)]
            + [evaluator.sigmoid(seed_logits[seed]) for seed in evaluator.NEW_SEEDS]
        ),
        axis=0,
    )
    mean_rank = evaluator.fold_category_ranks(mean_probability, folds, categories)
    expected = {}
    for name, bad_rank, flammable_rank in (
        ("original_route_baseline", original_rank, original_rank),
        ("category_routed_specialist_400", original_rank, specialist_rank),
        ("fixed_four_seed_probability_mean_route", mean_rank, mean_rank),
    ):
        _, report = evaluator.evaluate_route(
            name=name,
            bad_rank=bad_rank,
            flammable_rank=flammable_rank,
            visual=visual,
            route=protocol["route"],
        )
        expected[name] = report["macro_f1"]
    protocol["expected_macro_f1"] = expected
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)
    return {
        "visual": visual_path,
        "visual_contract": visual_contract,
        "folds": folds_path,
        "protocol": protocol_path,
        "specs600": specs600,
        "specs602": specs602,
    }


def _evaluate_fixture(tmp_path: Path, fixture: dict) -> tuple[dict, Path, Path]:
    output = tmp_path / "summary.json"
    local = tmp_path / "predictions.npz"
    summary = evaluator.evaluate(
        visual_bundle_path=fixture["visual"],
        visual_contract_path=fixture["visual_contract"],
        folds_path=fixture["folds"],
        exp600_specifications=fixture["specs600"],
        exp602_specifications=fixture["specs602"],
        protocol_path=fixture["protocol"],
        output_path=output,
        local_predictions_path=local,
        enforce_frozen=False,
    )
    return summary, output, local


def test_frozen_protocol_has_exactly_three_predeclared_comparisons() -> None:
    protocol = json.loads((EXP / "frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["comparisons"] == [
        "original_route_baseline",
        "category_routed_specialist_400",
        "fixed_four_seed_probability_mean_route",
    ]
    assert protocol["additional_variants_allowed"] is False
    assert protocol["seeds"] == [42, 31415, 271828, 161803]
    assert set(protocol["four_seed_weights"].values()) == {0.25}
    assert protocol["sealed_holdout_allowed"] is False


def test_complete_grid_reproduces_only_frozen_comparisons(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary, output, local = _evaluate_fixture(tmp_path, fixture)
    assert output.is_file() and local.is_file()
    assert summary["status"] == "complete"
    assert summary["sealed_holdout_used"] is False
    assert summary["gpu_jobs_launched"] == 0
    assert list(summary["results"]) == summary["comparisons_frozen"]
    assert all(item["passed"] for item in summary["reference_checks"].values())
    serialized = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    public = json.loads(serialized)
    forbidden_keys = {"job_name", "job_names", "local_path", "source_directory"}

    def all_keys(value):
        if isinstance(value, dict):
            yield from value
            for nested in value.values():
                yield from all_keys(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from all_keys(nested)

    assert forbidden_keys.isdisjoint(all_keys(public))
    with np.load(local, allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "ids",
            "labels",
            "categories",
            "folds",
            "semantic_components",
            "original_route_predictions",
            "category_routed_specialist_predictions",
            "fixed_four_seed_predictions",
            "fixed_four_seed_probability",
        }


def test_incomplete_grid_refuses_before_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["specs602"].pop()
    output = tmp_path / "summary.json"
    with pytest.raises(ValueError, match="incomplete experiment-602 grid"):
        evaluator.evaluate(
            visual_bundle_path=fixture["visual"],
            visual_contract_path=fixture["visual_contract"],
            folds_path=fixture["folds"],
            exp600_specifications=fixture["specs600"],
            exp602_specifications=fixture["specs602"],
            protocol_path=fixture["protocol"],
            output_path=output,
            enforce_frozen=False,
        )
    assert not output.exists()


def test_prediction_checksum_tamper_refuses_before_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = Path(fixture["specs600"][0][2]) / "lora_holdout_predictions.csv"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    output = tmp_path / "summary.json"
    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluator.evaluate(
            visual_bundle_path=fixture["visual"],
            visual_contract_path=fixture["visual_contract"],
            folds_path=fixture["folds"],
            exp600_specifications=fixture["specs600"],
            exp602_specifications=fixture["specs602"],
            protocol_path=fixture["protocol"],
            output_path=output,
            enforce_frozen=False,
        )
    assert not output.exists()


def test_reference_mismatch_refuses_without_publishing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    protocol = json.loads(Path(fixture["protocol"]).read_text(encoding="utf-8"))
    protocol["expected_macro_f1"]["original_route_baseline"] += 0.01
    _write_json(Path(fixture["protocol"]), protocol)
    output = tmp_path / "summary.json"
    with pytest.raises(AssertionError, match="reference mismatch"):
        evaluator.evaluate(
            visual_bundle_path=fixture["visual"],
            visual_contract_path=fixture["visual_contract"],
            folds_path=fixture["folds"],
            exp600_specifications=fixture["specs600"],
            exp602_specifications=fixture["specs602"],
            protocol_path=fixture["protocol"],
            output_path=output,
            enforce_frozen=False,
        )
    assert not output.exists()

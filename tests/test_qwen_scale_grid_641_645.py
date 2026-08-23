from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "experiments/645_qwen_scale_2x3_gate"
if str(GRID) not in sys.path:
    sys.path.insert(0, str(GRID))

import evaluate
import freeze_audit
import grid_contract
import runtime_builder
import train_lora

PROMPT_640_PATH = ROOT / "experiments/640_qwen38_scale_prompt_grid/run_prompt_scores.py"
PROMPT_SPEC = importlib.util.spec_from_file_location("exp640_prompt_for_grid_test", PROMPT_640_PATH)
assert PROMPT_SPEC is not None and PROMPT_SPEC.loader is not None
prompt_640 = importlib.util.module_from_spec(PROMPT_SPEC)
PROMPT_SPEC.loader.exec_module(prompt_640)


def test_grid_reuses_exact_prompt_and_frozen_effective_batch() -> None:
    row = SimpleNamespace(category="БАД", name="Товар", description="Описание")
    assert grid_contract.base_prompt(row) == prompt_640.prompt(row)
    assert grid_contract.CELL_SPECS["641"].micro_batch_size == 4
    assert grid_contract.CELL_SPECS["642"].micro_batch_size == 1
    assert {
        spec.micro_batch_size * spec.gradient_accumulation
        for spec in grid_contract.CELL_SPECS.values()
    } == {16}
    assert grid_contract.grid_contract_payload()["threshold"] == 0.0
    for prefix in ("641", "642", "643", "644", "645"):
        directory = next((ROOT / "experiments").glob(f"{prefix}_*"))
        frozen = json.loads((directory / "frozen_spec.json").read_text(encoding="utf-8"))
        assert frozen["grid_contract_sha256"] == grid_contract.GRID_CONTRACT_SHA256
    args = train_lora.parser_for("642").parse_args(
        [
            "--fold",
            "0",
            "--runtime-dir",
            "runtime",
            "--images",
            "images",
            "--model-root",
            "model",
            "--model-revision",
            grid_contract.MODEL_REVISIONS["Qwen/Qwen3.8-27B"],
            "--vendor",
            "vendor",
            "--output-dir",
            "output",
            "--technical-smoke",
        ]
    )
    assert args.technical_smoke is True
    source = (GRID / "train_lora.py").read_text(encoding="utf-8")
    assert 'use_kernels=True' in source
    assert 'optimized training kernels were requested but not activated' in source


def test_grounding_is_exact_and_never_generates_coordinates() -> None:
    row = {
        "name": "Комплект без топлива",
        "description": "Описание",
        "ocr_images": [
            {
                "image_index": 2,
                "detections": [
                    {
                        "text": "ОГНЕОПАСНО",
                        "polygon": [[1, 2], [3, 2], [3, 4], [1, 4]],
                    }
                ],
            }
        ],
    }
    text = grid_contract.resolve_grounding(row, "без топлива")
    assert text["grounded"] is True and text["source"] == "text"
    ocr = grid_contract.resolve_grounding(row, "ОГНЕОПАСНО")
    assert ocr == {
        "grounded": True,
        "source": "ocr",
        "image_index": 2,
        "region_index": 0,
        "polygon": [[1, 2], [3, 2], [3, 4], [1, 4]],
    }
    assert grid_contract.resolve_grounding(row, "огнеопасно")["grounded"] is False
    assert "polygon" not in grid_contract.structured_target(
        verdict=1,
        quote="ОГНЕОПАСНО",
        concept="COMPOSITION",
        order="evidence_first",
    )


def test_structured_generation_requires_declared_order_and_closed_evidence() -> None:
    target = grid_contract.structured_target(
        verdict=1,
        quote="БАД",
        concept="OBJECT_OF_SALE",
        order="evidence_first",
    )
    parsed = grid_contract.parse_structured_generation(target, expected_order="evidence_first")
    assert parsed["format_valid"] is True
    assert parsed["generated_verdict"] == 1
    assert (
        grid_contract.parse_structured_generation(target, expected_order="class_first")[
            "format_valid"
        ]
        is False
    )
    invalid = '{"quote":"выдумка","concept":"OPEN_CLASS","verdict":1}'
    assert (
        grid_contract.parse_structured_generation(invalid, expected_order="evidence_first")[
            "format_valid"
        ]
        is False
    )


def test_selector_never_uses_outer_fold_and_is_repeatable() -> None:
    labels = np.asarray([0, 1] * 1200, dtype=np.int8)
    categories = np.asarray(["БАД"] * 1200 + ["Легковоспламеняющиеся"] * 1200)
    folds = np.asarray([index % 5 for index in range(2400)], dtype=np.int8)
    scores = np.linspace(0.0, 1.0, 2400, dtype=np.float32)
    first = runtime_builder.select_training_indices(
        labels=labels,
        categories=categories,
        folds=folds,
        fused_scores=scores,
        outer_fold=3,
    )
    second = runtime_builder.select_training_indices(
        labels=labels,
        categories=categories,
        folds=folds,
        fused_scores=scores,
        outer_fold=3,
    )
    assert first == second
    assert first
    assert all(folds[index] != 3 for index in first)


def _standard_row(
    *,
    registry_row: dict,
    track: str,
    prediction: int,
    score: float,
) -> dict:
    model_id, objective, order = evaluate.TRACKS[track]
    evidence_track = objective == "grounded_evidence"
    return {
        "global_index": int(registry_row["global_index"]),
        "id": str(registry_row["id"]),
        "fold": int(registry_row["development_fold"]),
        "category": str(registry_row["category"]),
        "score": score,
        "prediction": prediction,
        "model_id": model_id,
        "model_revision": grid_contract.MODEL_REVISIONS[model_id],
        "objective": objective,
        "target_order": order,
        "prompt_version": grid_contract.PROMPT_VERSION,
        "preprocessing_version": grid_contract.PREPROCESSING_VERSION,
        "raw_generation": "{}" if evidence_track else "",
        "generated_verdict": prediction if evidence_track else -1,
        "format_valid": True,
        "quote": "маркер" if evidence_track else grid_contract.NO_EVIDENCE,
        "concept": "OBJECT_OF_SALE" if evidence_track else grid_contract.NO_EVIDENCE,
        "grounded": True,
        "grounding_source": "text" if evidence_track else "none",
        "image_index": -1,
        "region_index": -1,
    }


def _grid_fixture(
    tmp_path: Path,
) -> tuple[Path, list[Path], dict[str, list[Path]], dict[str, list[Path]]]:
    rows = []
    global_index = 0
    for fold in grid_contract.SCREEN_FOLDS:
        for category in grid_contract.CATEGORIES:
            for local_index, label in enumerate((1, 0, 1, 0)):
                rows.append(
                    {
                        "global_index": global_index,
                        "id": f"{global_index:04d}",
                        "category": category,
                        "label": label,
                        "semantic_component": f"c-{global_index}",
                        "split": "development",
                        "development_fold": fold,
                        "local_index": local_index,
                    }
                )
                global_index += 1
    registry = pd.DataFrame(rows)
    registry_path = tmp_path / "folds.csv"
    registry.drop(columns=["global_index", "local_index"]).to_csv(registry_path, index=False)
    runtime_paths = []
    for fold in grid_contract.SCREEN_FOLDS:
        path = tmp_path / f"runtime-{fold}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                if row["development_fold"] == fold:
                    runtime = {
                        "global_index": row["global_index"],
                        "id": row["id"],
                        "fold": fold,
                        "semantic_component": row["semantic_component"],
                        "category": row["category"],
                        "name": "маркер",
                        "description": "описание",
                        "image_url": "https://example.invalid/image.jpg",
                        "ocr_images": [],
                    }
                    stream.write(json.dumps(runtime, ensure_ascii=False) + "\n")
        runtime_paths.append(path)

    track_paths: dict[str, list[Path]] = {}
    contract_paths: dict[str, list[Path]] = {}
    for track in evaluate.TRACKS:
        directory = tmp_path / track
        directory.mkdir()
        path = directory / "predictions.jsonl"
        output_rows = []
        large = track.endswith("27b")
        for row in rows:
            label = int(row["label"])
            # Every small track makes the first positive in each category/fold negative;
            # every 27B track is perfect, so all scale gates are deterministic.
            wrong = not large and int(row["local_index"]) == 0
            prediction = 1 - label if wrong else label
            score = 1.0 if prediction else -1.0
            output_rows.append(
                _standard_row(
                    registry_row=row,
                    track=track,
                    prediction=prediction,
                    score=score,
                )
            )
        with path.open("w", encoding="utf-8") as stream:
            for row in output_rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        model_id, objective, _order = evaluate.TRACKS[track]
        contract = {
            "schema_version": 1,
            "experiment_id": "640" if objective == "prompting" else "fixture",
            "model_id": model_id,
            "model_revision": grid_contract.MODEL_REVISIONS[model_id],
            "objective": objective,
            "grid_contract_sha256": grid_contract.GRID_CONTRACT_SHA256,
            "model_input_view_sha256": "same-input-view",
            "threshold": 0.0,
            "threshold_tuned": False,
            "sealed_rows_used": 0,
            "artifacts": {path.name: grid_contract.sha256_file(path)},
            "decision": "GO_EVALUATE",
        }
        contract["contract_sha256"] = grid_contract.canonical_sha256(contract)
        contract_path = directory / "output_contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        track_paths[track] = [path]
        contract_paths[track] = [contract_path]
    return registry_path, runtime_paths, track_paths, contract_paths


def test_screen_evaluator_reports_full_grid_and_distinct_reasoning_orders(tmp_path: Path) -> None:
    registry, runtimes, tracks, contracts = _grid_fixture(tmp_path)
    result = evaluate.evaluate(
        registry_path=registry,
        runtime_paths=runtimes,
        track_paths=tracks,
        contract_paths=contracts,
        output_path=tmp_path / "result.json",
        folds_to_evaluate=grid_contract.SCREEN_FOLDS,
    )
    assert result["decision"] == "GO_FULL_FOLDS_1_2_4"
    assert set(result["tracks"]) == set(evaluate.TRACKS)
    assert result["comparisons"]["evidence_first_scale_4b_to_27b"]["winning_folds"] == 2
    assert "4b_class_to_class_first_aux" in result["comparisons"]
    assert result["distillation_allowed"] is False
    assert result["sealed_rows_loaded"] == 0
    assert result["public_used"] is False


def test_evaluator_rechecks_grounding_instead_of_trusting_output(tmp_path: Path) -> None:
    registry, runtimes, tracks, contracts = _grid_fixture(tmp_path)
    target = tracks["evidence_first_27b"][0]
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    rows[0]["quote"] = "выдуманная цитата"
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    contract_path = contracts["evidence_first_27b"][0]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["artifacts"][target.name] = grid_contract.sha256_file(target)
    contract.pop("contract_sha256")
    contract["contract_sha256"] = grid_contract.canonical_sha256(contract)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="grounding claim mismatch"):
        evaluate.evaluate(
            registry_path=registry,
            runtime_paths=runtimes,
            track_paths=tracks,
            contract_paths=contracts,
            output_path=tmp_path / "result.json",
            folds_to_evaluate=grid_contract.SCREEN_FOLDS,
        )


def test_manual_audit_freezer_is_label_blind_and_balanced(tmp_path: Path) -> None:
    rows = []
    index = 0
    for fold in grid_contract.FULL_FOLDS:
        for category in grid_contract.CATEGORIES:
            for _ in range(21):
                rows.append(
                    {
                        "id": str(index),
                        "category": category,
                        "semantic_component": f"component-{index}",
                        "split": "development",
                        "development_fold": fold,
                        "label": index % 2,
                    }
                )
                index += 1
    registry = tmp_path / "registry.csv"
    pd.DataFrame(rows).to_csv(registry, index=False)
    result = freeze_audit.freeze(
        registry_path=registry,
        output_path=tmp_path / "audit.csv",
        report_path=tmp_path / "audit.json",
    )
    assert result["rows"] == 200
    assert result["labels_read"] == 0
    assert result["sealed_rows"] == 0


def test_manual_audit_gate_requires_relevance_support_and_scope(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {"id": [str(index) for index in range(200)], "track": "evidence_first_27b"}
    )
    ratings = manifest.copy()
    ratings["relevant"] = 1
    ratings["unsupported"] = 0
    ratings["sold_object_correct"] = 1
    ratings["negation_correct"] = 1
    ratings["completeness_correct"] = 1
    manifest_path = tmp_path / "manifest.csv"
    ratings_path = tmp_path / "ratings.csv"
    manifest.to_csv(manifest_path, index=False)
    ratings.to_csv(ratings_path, index=False)
    result = evaluate._manual_audit(
        manifest_path,
        ratings_path,
        expected_track="evidence_first_27b",
    )
    assert result["passed"] is True
    ratings.loc[:2, "unsupported"] = 1
    ratings.to_csv(ratings_path, index=False)
    result = evaluate._manual_audit(
        manifest_path,
        ratings_path,
        expected_track="evidence_first_27b",
    )
    assert result["passed"] is False

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFIT = load(
    "refit_contract_629",
    ROOT / "experiments/629_semantic_v3_full_data_refit/full_data_contract.py",
)
PRODUCTION = load(
    "production_contract_630",
    ROOT / "experiments/630_semantic_v3_production_submission/production_contract.py",
)


def dump(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def recipe() -> dict:
    value = {
        "schema_version": "final_recipe_manifest_v1",
        "selected_recipe_count": 1,
        "sealed_holdout_used": False,
        "component_weights": {"text": 1.0},
        "route_weights": {"БАД": 1.0, "Легковоспламеняющиеся": 1.0},
        "seed": 31415,
    }
    value.update({field: "7" * 64 for field in REFIT.REQUIRED_RECIPE_SHA_FIELDS})
    return value


def make_plan(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    recipe_path = dump(tmp_path / "recipe.json", recipe())
    recipe_sha = REFIT.sha256_file(recipe_path)
    accepted_path = dump(
        tmp_path / "accepted_628.json",
        {
            "schema_version": "sealed_acceptance_result_v1",
            "experiment_id": 628,
            "status": "accepted",
            "decision": "ACCEPT",
            "sealed_evaluation_count": 1,
            "frozen_recipe_manifest_sha256": recipe_sha,
            "sealed_acceptance_policy_sha256": "a" * 64,
            "sealed_evaluator_sha256": "b" * 64,
            "sealed_input_identity_sha256": "c" * 64,
        },
    )
    data_path = dump(
        tmp_path / "full_data.json",
        {
            "schema_version": "permitted_full_data_manifest_v1",
            "sealed_training_rows": 0,
            "data_registry_sha256": "d" * 64,
            "data_selection_sha256": "e" * 64,
            "components": [
                {
                    "component": "text",
                    "source_experiment": 623,
                    "training_rows": 100,
                    "sealed_training_rows": 0,
                    "training_input_sha256": "f" * 64,
                    "training_config_sha256": "1" * 64,
                }
            ],
        },
    )
    plan = REFIT.build_fit_plan(
        recipe_path=recipe_path,
        accepted_628_path=accepted_path,
        full_data_path=data_path,
    )
    return plan, recipe_path, accepted_path, data_path


def valid_refit(plan: dict, artifact_root: Path | None = None) -> dict:
    job = plan["jobs"][0]
    if artifact_root is None:
        artifact_hash = "2" * 64
        runtime_hash = "3" * 64
        provenance_hash = "4" * 64
    else:
        (artifact_root / "model.bin").write_bytes(b"model-output")
        (artifact_root / "runtime.json").write_text("{}", encoding="utf-8")
        (artifact_root / "provenance.json").write_text("{}", encoding="utf-8")
        artifact_hash = REFIT.sha256_file(artifact_root / "model.bin")
        runtime_hash = REFIT.sha256_file(artifact_root / "runtime.json")
        provenance_hash = REFIT.sha256_file(artifact_root / "provenance.json")
    return {
        "schema_version": "full_data_refit_manifest_v1",
        "experiment_id": 629,
        "status": "accepted",
        "frozen_recipe_manifest_sha256": plan["frozen_recipe_manifest_sha256"],
        "accepted_628_result_sha256": plan["accepted_628_result_sha256"],
        "permitted_full_data_manifest_sha256": plan["permitted_full_data_manifest_sha256"],
        "sealed_training_rows": 0,
        "component_fits": [
            {
                "component": "text",
                "source_experiment": 623,
                "fit_count": 1,
                "fit_index": 1,
                "gpu_count": 1,
                "sealed_training_rows": 0,
                "training_input_sha256": job["training_input_sha256"],
                "training_config_sha256": job["training_config_sha256"],
                "artifact_sha256": artifact_hash,
                "runtime_evidence_sha256": runtime_hash,
                "provenance_sha256": provenance_hash,
                "artifact_size_bytes": 12,
                "integrity_passed": True,
                "artifact_path": "model.bin",
                "runtime_evidence_path": "runtime.json",
                "provenance_path": "provenance.json",
            }
        ],
    }


def test_629_plan_requires_accepted_628_and_no_sealed_rows(tmp_path: Path) -> None:
    plan, recipe_path, accepted_path, data_path = make_plan(tmp_path)
    assert plan["fits_per_component"] == 1
    assert plan["jobs"] == [
        {
            "component": "text",
            "source_experiment": 623,
            "fit_index": 1,
            "gpu_count": 1,
            "seed": 31415,
            "training_input_sha256": "f" * 64,
            "training_config_sha256": "1" * 64,
        }
    ]
    parent = json.loads(accepted_path.read_text(encoding="utf-8"))
    parent["decision"] = "REJECT"
    dump(accepted_path, parent)
    with pytest.raises(ValueError, match="not accepted"):
        REFIT.build_fit_plan(
            recipe_path=recipe_path,
            accepted_628_path=accepted_path,
            full_data_path=data_path,
        )


def test_629_refit_rejects_second_fit_or_wrong_provenance(tmp_path: Path) -> None:
    plan, *_ = make_plan(tmp_path)
    manifest = valid_refit(plan, tmp_path)
    REFIT.verify_refit_manifest(manifest, plan=plan, artifact_root=tmp_path)
    manifest["component_fits"][0]["fit_count"] = 2
    with pytest.raises(ValueError, match="exactly once"):
        REFIT.verify_refit_manifest(manifest, plan=plan, artifact_root=tmp_path)
    manifest = valid_refit(plan, tmp_path)
    manifest["component_fits"][0]["training_input_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="changed frozen"):
        REFIT.verify_refit_manifest(manifest, plan=plan, artifact_root=tmp_path)


def test_629_refuses_any_sealed_training_row(tmp_path: Path) -> None:
    _, recipe_path, accepted_path, data_path = make_plan(tmp_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    data["components"][0]["sealed_training_rows"] = 1
    dump(data_path, data)
    with pytest.raises(ValueError, match="sealed rows"):
        REFIT.build_fit_plan(
            recipe_path=recipe_path,
            accepted_628_path=accepted_path,
            full_data_path=data_path,
        )


def official() -> dict:
    return {
        "schema_version": "official_submission_identity_v1",
        "rows": [
            {"id": "a", "category": "БАД"},
            {"id": "b", "category": "Легковоспламеняющиеся"},
        ],
    }


def predictions() -> list[dict]:
    card = "Газовая горелка без баллона"
    quote = "без баллона"
    return [
        {
            "id": "a",
            "category": "БАД",
            "score": 0.1,
            "verdict": 0,
            "card": "Витамин C в таблетках",
            "evidence": "Витамин C",
            "char_start": 0,
            "char_end": 9,
            "concept": "COMPOSITION",
            "explanation": "Состав товара указан прямо: «Витамин C».",
        },
        {
            "id": "b",
            "category": "Легковоспламеняющиеся",
            "score": 0.2,
            "verdict": 0,
            "card": card,
            "evidence": quote,
            "char_start": card.index(quote),
            "char_end": card.index(quote) + len(quote),
            "concept": "NEGATION",
            "explanation": "Комплектность описана точной фразой «без баллона».",
        },
    ]


def runtime() -> dict:
    return {
        "schema_version": "production_runtime_evidence_v1",
        "measured_public_seconds": 80.0,
        "public_limit_seconds": 100.0,
        "public_margin_seconds": 20.0,
        "measured_private_seconds": 160.0,
        "private_limit_seconds": 200.0,
        "private_margin_seconds": 40.0,
        "hardware_contract_sha256": "5" * 64,
        "container_contract_sha256": "6" * 64,
    }


def test_630_rejects_wrong_ids_generic_and_absence_reasoning() -> None:
    rows = predictions()
    PRODUCTION.verify_predictions(rows, official())
    rows[0]["id"] = "wrong"
    with pytest.raises(ValueError, match="ID set"):
        PRODUCTION.verify_predictions(rows, official())
    rows = predictions()
    rows[0]["explanation"] = "Витамин C: найдены признаки нарушения."
    with pytest.raises(ValueError, match="generic"):
        PRODUCTION.verify_predictions(rows, official())
    rows = predictions()
    rows[0]["explanation"] = "Витамин C, а другой признак не найден."
    with pytest.raises(ValueError, match="unsupported visual or absence"):
        PRODUCTION.verify_predictions(rows, official())


def test_630_uses_exact_623_closed_concept_vocabulary() -> None:
    rows = predictions()
    rows[1]["concept"] = "FUEL_OR_IGNITION"
    PRODUCTION.verify_predictions(rows, official())
    rows[1]["concept"] = "FUEL"
    with pytest.raises(ValueError, match="closed vocabulary"):
        PRODUCTION.verify_predictions(rows, official())


def test_630_archive_is_allowlisted_bound_and_integrity_checked(tmp_path: Path) -> None:
    plan, *_ = make_plan(tmp_path)
    refit_path = dump(tmp_path / "refit.json", valid_refit(plan))
    official_path = dump(tmp_path / "official.json", official())
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions()) + "\n",
        encoding="utf-8",
    )
    runtime_path = dump(tmp_path / "runtime.json", runtime())
    payload_path = tmp_path / "model.bin"
    payload_path.write_bytes(b"model")
    payloads = {"model/model.bin": payload_path}
    manifest = PRODUCTION.build_manifest(
        accepted_refit_path=refit_path,
        official_path=official_path,
        predictions_path=predictions_path,
        runtime_path=runtime_path,
        payloads=payloads,
    )
    archive_path = tmp_path / "candidate.zip"
    PRODUCTION.assemble_archive(
        manifest=manifest,
        predictions_path=predictions_path,
        payloads=payloads,
        output=archive_path,
    )
    receipt = PRODUCTION.verify_archive(
        archive_path,
        accepted_refit_path=refit_path,
        official_path=official_path,
        runtime_path=runtime_path,
    )
    assert receipt["decision"] == "GO"
    assert receipt["row_count"] == 2


def test_630_rejects_nonpositive_or_inconsistent_runtime_margin() -> None:
    value = runtime()
    value["private_margin_seconds"] = 0.0
    with pytest.raises(ValueError, match="runtime margin"):
        PRODUCTION.verify_runtime(value)

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/600_semantic_v3_qwen35_baselines"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load("exp600_protocol_test", EXP / "protocol.py")
trainer = _load("exp600_trainer_test", EXP / "train_component.py")


def _synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    ids = [f"dev-{index}" for index in range(5)] + ["sealed-0"]
    data = pd.DataFrame(
        {
            "id": ids,
            "category": ["БАД", "БАД", "Легковоспламеняющиеся", "БАД", "БАД", "БАД"],
            "label": [0, 1, 0, 1, 0, 1],
            "name": [f"item {index}" for index in range(6)],
            "description": [f"description {index}" for index in range(6)],
            "comment": ["comment"] * 6,
        }
    )
    folds = pd.DataFrame(
        {
            "id": ids,
            "category": data["category"],
            "label": data["label"],
            "semantic_component": [f"component-{index}" for index in range(6)],
            "component_size": [1] * 6,
            "split": ["development"] * 5 + ["sealed_holdout"],
            "development_fold": [0, 1, 2, 3, 4, -1],
        }
    )
    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"
    selector_path = tmp_path / "development_selector_oof.npz"
    provenance_path = tmp_path / "development_selector_provenance.json"
    data.to_csv(data_path, index=False)
    folds.to_csv(folds_path, index=False)
    np.savez_compressed(
        selector_path,
        ids=np.asarray(ids[:5]),
        fold_ids=np.arange(5, dtype=np.int8),
        fused_scores=np.linspace(0.1, 0.9, 5, dtype=np.float32),
    )
    provenance = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "scope": "development_only",
        "rows": 5,
        "ids_sha256": protocol.id_sequence_sha256(ids[:5]),
        "selector_npz_sha256": protocol.sha256_file(selector_path),
        "sealed_rows_used_for_fit": False,
        "sealed_labels_used": False,
        "sealed_rows_used_for_selection": False,
        "sealed_rows_used_for_thresholds": False,
        "selector_source_protocol": protocol.SELECTOR_SOURCE_PROTOCOL,
        "frozen_before_qwen_training": True,
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    return data_path, folds_path, selector_path, provenance_path


def _prepare(tmp_path: Path, *, component: str = "original", fold: int = 2):
    data, folds, selector, provenance = _synthetic_inputs(tmp_path)
    output = tmp_path / "protocol_inputs"
    prepared = protocol.prepare_development_inputs(
        data_path=data,
        folds_path=folds,
        selector_path=selector,
        selector_provenance_path=provenance,
        output_dir=output,
        component=component,
        outer_fold=fold,
        enforce_frozen_hash=False,
    )
    return prepared, folds


def test_frozen_protocol_manifest_matches_real_semantic_v3() -> None:
    manifest = json.loads(
        (EXP / "analysis/frozen_protocol_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["folds_sha256"] == protocol.FOLDS_SHA256
    assert manifest["data_sha256"] == protocol.DATA_SHA256
    assert manifest["image_manifest_sha256"] == protocol.IMAGE_MANIFEST_SHA256
    assert manifest["parent_sha256"] == trainer.PARENT_SHA256
    assert manifest["source_rows"] == 12_971
    assert manifest["development_rows"] == 11_118
    assert manifest["sealed_holdout_rows"] == 1_853
    assert manifest["required_jobs"] == 10
    assert {(job["component"], job["fold"]) for job in manifest["jobs"]} == {
        (component, fold)
        for component in protocol.COMPONENTS
        for fold in protocol.DEVELOPMENT_FOLDS
    }
    assert all(job["sealed_rows"] == 0 for job in manifest["jobs"])


def test_physical_filter_mapping_and_audit_exclude_sealed(tmp_path: Path) -> None:
    prepared, _ = _prepare(tmp_path)
    physical = pd.read_csv(prepared["data"], dtype={"id": str})
    mapping = pd.read_csv(prepared["mapping"], dtype={"id": str})
    selector = np.load(prepared["selector"], allow_pickle=True)
    audit = prepared["audit_payload"]

    assert physical["id"].tolist() == [f"dev-{index}" for index in range(5)]
    assert "sealed-0" not in set(physical["id"])
    assert selector["ids"].astype(str).tolist() == physical["id"].tolist()
    assert selector["fold_ids"].tolist() == [0, 1, 2, 3, 4]
    assert mapping["original_index"].tolist() == [0, 1, 2, 3, 4]
    assert mapping["outer_role"].tolist().count("validation") == 1
    assert audit["sealed_rows_in_physical_dataset"] == 0
    assert audit["sealed_rows_in_selector_npz"] == 0
    assert audit["sealed_rows_in_mapping"] == 0
    assert audit["sealed_holdout_used_for_train"] is False
    assert audit["sealed_holdout_used_for_evaluation"] is False


def test_selector_with_sealed_row_is_rejected(tmp_path: Path) -> None:
    data, folds, selector, provenance = _synthetic_inputs(tmp_path)
    np.savez_compressed(
        selector,
        ids=np.asarray([f"dev-{index}" for index in range(5)] + ["sealed-0"]),
        fold_ids=np.asarray([0, 1, 2, 3, 4, -1], dtype=np.int8),
        fused_scores=np.linspace(0.1, 0.9, 6, dtype=np.float32),
    )
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["selector_npz_sha256"] = protocol.sha256_file(selector)
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly development ids"):
        protocol.prepare_development_inputs(
            data_path=data,
            folds_path=folds,
            selector_path=selector,
            selector_provenance_path=provenance,
            output_dir=tmp_path / "output",
            component="original",
            outer_fold=0,
            enforce_frozen_hash=False,
        )


def test_provenance_that_used_sealed_labels_is_rejected(tmp_path: Path) -> None:
    data, folds, selector, provenance = _synthetic_inputs(tmp_path)
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["sealed_labels_used"] = True
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        protocol.prepare_development_inputs(
            data_path=data,
            folds_path=folds,
            selector_path=selector,
            selector_provenance_path=provenance,
            output_dir=tmp_path / "output",
            component="specialist",
            outer_fold=4,
            enforce_frozen_hash=False,
        )


def test_legacy_selector_source_protocol_is_rejected(tmp_path: Path) -> None:
    data, folds, selector, provenance = _synthetic_inputs(tmp_path)
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload.pop("selector_source_protocol")
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        protocol.prepare_development_inputs(
            data_path=data,
            folds_path=folds,
            selector_path=selector,
            selector_provenance_path=provenance,
            output_dir=tmp_path / "output",
            component="original",
            outer_fold=0,
            enforce_frozen_hash=False,
        )


@pytest.mark.parametrize("component", protocol.COMPONENTS)
@pytest.mark.parametrize("fold", protocol.DEVELOPMENT_FOLDS)
def test_environment_locks_exact_parent_recipe(component: str, fold: int) -> None:
    environment = trainer.configure_environment(
        component=component,
        fold=fold,
        environment={},
    )
    assert environment["SEED"] == "42"
    assert environment["HOLDOUT_FOLD"] == str(fold)
    assert environment["FULL_TRAIN"] == "0"
    assert environment["TRAINING_MODE"] == "hard"
    assert environment["MODEL_CLASS"] == "multimodal"
    assert environment["USE_CHAT_BATCH"] == "1"
    assert environment["DESCRIPTION_LIMIT"] == "1800"
    if component == "specialist":
        assert environment["FAMILY_BALANCE_FLAMMABLE"] == "0"
        assert environment["FAMILY_DIVERSE_BAD_POSITIVES"] == "1"
        assert environment["FAMILY_DIVERSE_FLAMMABLE_NEGATIVES"] == "0"


def test_output_contract_accepts_only_outer_development_predictions(tmp_path: Path) -> None:
    prepared, _ = _prepare(tmp_path)
    output = tmp_path / "candidate"
    output.mkdir()
    target_protocol = output / "protocol_inputs"
    prepared["data"].parent.rename(target_protocol)
    mapping = pd.read_csv(target_protocol / "id_mapping.csv", dtype={"id": str})
    expected = mapping.loc[mapping["outer_role"].eq("validation"), "id"].tolist()
    pd.DataFrame({"id": expected, "fold": [2], "prediction": [0]}).to_csv(
        output / "lora_holdout_predictions.csv", index=False
    )
    (output / "lora_holdout_report.json").write_text(
        json.dumps({"holdout_fold": 2, "train_records": 4, "download_failures": 0}),
        encoding="utf-8",
    )
    (output / "selection_audit.runtime.json").write_text(
        json.dumps({"training_records": 4, "decision": "GO"}), encoding="utf-8"
    )
    (output / "adapter.zip").write_bytes(b"synthetic adapter")

    contract = trainer.validate_output_contract(
        output_dir=output,
        component="original",
        fold=2,
    )
    assert contract["sealed_rows_in_predictions"] == 0
    assert contract["sealed_rows_used_for_evaluation"] == 0
    assert contract["decision"] == "GO"


def test_null_parity_is_exact_and_experiment_is_not_launched() -> None:
    parity = json.loads((EXP / "analysis/null_selector_parity.json").read_text(encoding="utf-8"))
    metrics = json.loads((EXP / "results/metrics.json").read_text(encoding="utf-8"))
    assert parity["decision"] == "PASS"
    assert parity["original_all_exact"] is True
    assert parity["specialist_all_exact"] is True
    assert metrics["status"] == "prepared_blocked"
    assert metrics["launched"] is False
    assert metrics["completed_jobs"] == 0
    assert metrics["sealed_holdout_used"] is False


def test_runtime_presets_cover_ten_neutral_one_gpu_jobs() -> None:
    presets = sorted((EXP / ".local/runtime").glob("*.yml"))
    assert len(presets) == 10
    observed: set[tuple[str, int]] = set()
    for path in presets:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))["job"]
        args = [str(value) for value in config["args"]]
        component = args[args.index("--component") + 1]
        fold = int(args[args.index("--fold") + 1])
        observed.add((component, fold))
        assert config["flavor"] == "h100-1x"
        assert config["generate_name"] == (
            f"sv3-q35o-f{fold}" if component == "original" else f"sv3-q35s-f{fold}"
        )
        assert "ecup" not in config["generate_name"].lower()
        assert args[args.index("--runtime-dir") + 1] == "/work/input/runtime"
        assert "--data" not in args
        assert "--folds" not in args
        assert "--image-manifest" not in args
        file_sources = [
            str(item.get("src", "")) for item in config["input"] if item.get("type") == "files"
        ]
        assert "research/data.csv" not in file_sources
        assert "validation/semantic_family_v3/folds.csv" not in file_sources
        assert "research/lora_image_manifest_complete.tsv.gz" not in file_sources
    assert observed == {
        (component, fold)
        for component in protocol.COMPONENTS
        for fold in protocol.DEVELOPMENT_FOLDS
    }


def test_public_files_contain_no_private_infrastructure_references() -> None:
    public_paths = [
        EXP / "README.md",
        EXP / "experiment.toml",
        EXP / "artifacts/README.md",
        EXP / "results/metrics.json",
        EXP / "analysis/frozen_protocol_manifest.json",
        EXP / "analysis/null_selector_parity.json",
    ]
    forbidden = ("private-compute-host", "secretToken", "task_id", "job_id")
    for path in public_paths:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(token.lower() in text for token in forbidden), path

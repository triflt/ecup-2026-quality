from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/654_qwen36_27b_class_only_lora"


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, EXP / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILD = load("runtime_654", "build_runtime.py")
VERIFY = load("verify_654", "verify_artifact.py")
PRESET = load("preset_654", "build_private_preset.py")
EVALUATE = load("evaluate_654", "evaluate_screen.py")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_runtime_rejects_non_screen_fold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="screen folds"):
        BUILD.build(tmp_path / "source", tmp_path / "output", fold=1)


def test_artifact_verifier_accepts_contract_and_rejects_tamper(tmp_path: Path) -> None:
    predictions = [
        {
            "global_index": 1,
            "id": "a",
            "fold": 0,
            "category": "БАД",
            "score": 0.25,
            "prediction": 1,
        },
        {
            "global_index": 2,
            "id": "b",
            "fold": 0,
            "category": "БАД",
            "score": -0.25,
            "prediction": 0,
        },
    ]
    payload = "".join(json.dumps(row) + "\n" for row in predictions).encode()
    adapter_config = b"{}"
    adapter_model = b"weights"
    report = {
        "experiment_id": "654",
        "model_revision": VERIFY.MODEL_REVISION,
        "outer_fold": 0,
        "train_occurrences": 4892,
        "validation_rows": 2,
        "optimizer_steps": 306,
        "threshold": 0.0,
        "threshold_tuned": False,
        "validation_labels_written": 0,
        "sealed_rows": 0,
        "public_used": False,
        "cpu_or_disk_offload": False,
        "decision": "READY_FOR_FROZEN_EVALUATION",
        "predictions_sha256": VERIFY.sha256_bytes(payload),
        "adapter_manifest": {
            "adapter_config.json": VERIFY.sha256_bytes(adapter_config),
            "adapter_model.safetensors": VERIFY.sha256_bytes(adapter_model),
        },
    }
    archive = tmp_path / "artifact.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("report.json", json.dumps(report))
        output.writestr("predictions.jsonl", payload)
        output.writestr("adapter/adapter_config.json", adapter_config)
        output.writestr("adapter/adapter_model.safetensors", adapter_model)
    assert VERIFY.verify(archive, fold=0, expected_rows=2)["decision"] == "PASS"
    predictions[0]["prediction"] = 0
    bad_payload = "".join(json.dumps(row) + "\n" for row in predictions).encode()
    report["predictions_sha256"] = VERIFY.sha256_bytes(bad_payload)
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("report.json", json.dumps(report))
        output.writestr("predictions.jsonl", bad_payload)
        output.writestr("adapter/adapter_config.json", adapter_config)
        output.writestr("adapter/adapter_model.safetensors", adapter_model)
    with pytest.raises(ValueError, match="zero threshold"):
        VERIFY.verify(archive, fold=0, expected_rows=2)


def test_private_preset_builder_uses_frozen_fold_and_four_gpu_flavor(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    four = tmp_path / "four.yml"
    base.write_text(
        """job:
  flavor: one
  region: region
  image: image
  preemption: false
  work_dir: /work
  env:
    TOKENIZERS_PARALLELISM: "false"
    PYTORCH_ALLOC_CONF: expandable_segments:True
  input:
    - {type: model_registry, src: model, dst: /hf_models/}
""",
        encoding="utf-8",
    )
    four.write_text("job:\n  flavor: four\n", encoding="utf-8")
    payload = PRESET.build(
        SimpleNamespace(
            output=tmp_path / "out.yml",
            fold=3,
            bundle=Path("experiments/654/.local/bundle.tar.gz"),
            bundle_url_file=None,
            base_qwen36=base,
            base_four_gpu=four,
        )
    )
    assert "flavor: four" in payload
    assert "--fold 3" in payload
    assert "fold3" in payload
    assert payload.count("type: model_registry") == 1
    assert "time_limit: 8h0m0s" in payload


def test_private_preset_builder_supports_presigned_bundle_delivery(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    four = tmp_path / "four.yml"
    url_file = tmp_path / "url.txt"
    base.write_text(
        """job:
  flavor: one
  region: region
  image: image
  preemption: false
  work_dir: /work
  env:
    TOKENIZERS_PARALLELISM: "false"
    PYTORCH_ALLOC_CONF: expandable_segments:True
  input:
    - {type: model_registry, src: model, dst: /hf_models/}
""",
        encoding="utf-8",
    )
    four.write_text("job:\n  flavor: four\n", encoding="utf-8")
    url_file.write_text("https://example.invalid/bundle?token=private\n", encoding="utf-8")
    payload = PRESET.build(
        SimpleNamespace(
            output=tmp_path / "out.yml",
            fold=0,
            bundle=None,
            bundle_url_file=url_file,
            base_qwen36=base,
            base_four_gpu=four,
        )
    )
    assert "BUNDLE_URL:" in payload
    assert "urlretrieve" in payload
    assert "dst: /work/qwen36_class_bundle.tar.gz" not in payload
    assert payload.count("type: model_registry") == 1


def test_private_preset_builder_rejects_closed_fold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="screen folds"):
        PRESET.build(SimpleNamespace(output=tmp_path / "out", fold=1))


def test_score_loader_rejects_supervision(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    write_jsonl(
        path,
        [
            {
                "global_index": 0,
                "id": "a",
                "fold": 0,
                "category": "БАД",
                "score": 1.0,
                "prediction": 1,
                "label": 1,
            }
        ],
    )
    with pytest.raises(ValueError, match="supervision"):
        EVALUATE.load_scores([path], [0])

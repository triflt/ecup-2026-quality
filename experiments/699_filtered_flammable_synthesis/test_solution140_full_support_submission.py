from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load(
    "exp699_solution140_full_support_submission",
    "build_solution140_full_support_submission.py",
)

SOURCE_RUN = '''from __future__ import annotations
import argparse
import gc
import gzip
import html
import importlib.util
import json
import os
QWEN35_ADAPTER_PATH = Path(
    os.environ.get("QWEN35_LORA_ADAPTER_PATH", ROOT / "adapter_qwen35")
)
def open_lora_image(path: str | None) -> Image.Image:
    image = Image.open(path).convert("RGB") if path else Image.new("RGB", (32, 32), "white")
    # Training normalized every first image to this bound before both Qwen
    # processors. It also keeps Qwen3.5 visual tokens inside the 1,536-token
    # context instead of truncating multimodal special tokens.
    image.thumbnail((448, 448), Image.Resampling.LANCZOS)
    return image
def compute_lora_scores(
    model, processor, frame: pd.DataFrame, *, use_chat_batch: bool = False
) -> np.ndarray:
        images = [open_lora_image(path) for path in paths]
    qwen35_model, qwen35_processor = load_lora_model(
        QWEN35_MODEL_PATH, QWEN35_ADAPTER_PATH, "multimodal"
    )
    qwen35_scores = compute_lora_scores(
        qwen35_model, qwen35_processor, frame, use_chat_batch=True
    )
'''


def write_json(path: Path, value: dict, self_field: str) -> str:
    value = dict(value)
    value[self_field] = builder.canonical_sha256(value)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return value[self_field]


def adapter(root: Path, marker: bytes) -> dict[str, str]:
    root.mkdir(parents=True)
    payloads = {
        "README.md": b"adapter\n" + marker,
        "adapter_config.json": b'{}\n' + marker,
        "adapter_model.safetensors": b"weights-" + marker,
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    return {name: builder.sha256_file(root / name) for name in payloads}


def packet(
    tmp_path: Path, *, cap: int, preprocessing: str
) -> tuple[Path, Path, Path, str, str, str, str]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text(SOURCE_RUN, encoding="utf-8")
    (source / "metadata.json").write_text("{}\n", encoding="utf-8")
    (source / "untouched.bin").write_bytes(b"untouched")
    adapter(source / "adapter_qwen35", b"original")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    synth_ids = "1" * 64
    synth_payload = "2" * 64
    runtime_contract = write_json(
        runtime / "runtime_audit.json",
        {
            "schema": "exp699_solution140_full_support_refit_runtime_v1",
            "decision": "GO_FULL_REFIT_SOLUTION140_FULL_SUPPORT",
            "outer_fold": -1,
            "source": "v2",
            "mode": "positive_only_append",
            "cap": cap,
            "preprocessing_key": preprocessing,
            "original_real_occurrences": 6790,
            "original_real_unique": 5998,
            "train_occurrences": 6790 + cap,
            "synthetic_occurrences": cap,
            "removed_real_occurrences": 0,
            "expected_optimizer_steps": 425,
            "validation_labels_written": 0,
            "sealed_rows_written": 0,
            "public_rows_used": 0,
            "selected_synth_ids_sha256": synth_ids,
            "selected_synth_payload_sha256": synth_payload,
            **builder.EXPECTED_REAL_SELECTION,
        },
        "contract_sha256",
    )
    output = tmp_path / "output"
    output.mkdir()
    adapter_hashes = adapter(output / "adapter", b"refit")
    with zipfile.ZipFile(output / "adapter.zip", "w") as archive:
        for name in sorted(adapter_hashes):
            archive.write(output / "adapter" / name, name)
    (output / "predictions.jsonl").write_bytes(b"")
    write_json(
        output / "output_contract.json",
        {
            "experiment_id": "699",
            "architecture": "qwen35_4b",
            "fold": -1,
            "runtime_contract_sha256": runtime_contract,
            "source": "v2",
            "mode": "positive_only_append",
            "cap": cap,
            "preprocessing_key": preprocessing,
            "epochs": 1,
            "train_occurrences": 6790 + cap,
            "training_occurrences_seen": 6790 + cap,
            "synthetic_occurrences": cap,
            "optimizer_steps": 425,
            "validation_rows": 0,
            "validation_labels_read": 0,
            "sealed_rows_used": 0,
            "public_rows_used": 0,
            "technical_smoke": {"finite": True, "loss": 0.2, "grad_norm": 3.0},
            "last_grad_norm": 2.0,
            "training_runtime_minutes": 50.0,
            "artifacts": {
                "predictions.jsonl": builder.sha256_file(output / "predictions.jsonl"),
                "adapter.zip": builder.sha256_file(output / "adapter.zip"),
            },
            "decision": "GO_PACKAGE",
        },
        "contract_sha256",
    )
    return (
        source,
        runtime,
        output,
        runtime_contract,
        synth_ids,
        synth_payload,
        builder.sha256_file(source / "run.py"),
    )


def build_candidate(tmp_path: Path, *, cap: int, preprocessing: str, route: str):
    source, runtime, output, runtime_contract, synth_ids, synth_payload, run_sha = packet(
        tmp_path, cap=cap, preprocessing=preprocessing
    )
    destination = tmp_path / "candidate"
    archive = tmp_path / "candidate.zip"
    report = builder.build(
        source=source,
        runtime_dir=runtime,
        refit_output=output,
        destination=destination,
        archive=archive,
        route=route,
        preprocessing=preprocessing,
        synth_cap=cap,
        expected_source_run_sha256=run_sha,
        expected_runtime_contract=runtime_contract,
        expected_synth_ids_sha256=synth_ids,
        expected_synth_payload_sha256=synth_payload,
    )
    return report, source, destination, archive


def test_all_thumbnail_changes_only_qwen35_adapter(tmp_path: Path) -> None:
    report, source, destination, archive = build_candidate(
        tmp_path, cap=10, preprocessing="thumbnail448", route="all"
    )
    assert report["decision"] == "GO_PACKAGE_SMOKE"
    assert report["candidate_run_sha256"] == report["source_run_sha256"]
    assert (destination / "untouched.bin").read_bytes() == b"untouched"
    assert builder.sha256_file(destination / "adapter_qwen35/adapter_model.safetensors") != (
        builder.sha256_file(source / "adapter_qwen35/adapter_model.safetensors")
    )
    assert archive.is_file()


def test_all_area_cap_patches_only_preprocessing_and_adapter(tmp_path: Path) -> None:
    report, _, destination, _ = build_candidate(
        tmp_path, cap=10, preprocessing="area_cap262144", route="all"
    )
    run = (destination / "run.py").read_text(encoding="utf-8")
    assert "max_pixels=262144" in run
    assert report["unrelated_solution140_members_changed"] == 0


def test_flammable_route_preserves_original_bad_adapter(tmp_path: Path) -> None:
    report, source, destination, _ = build_candidate(
        tmp_path, cap=10, preprocessing="thumbnail448", route="flammable_only"
    )
    assert report["category_route"] == {
        "БАД": "original_solution140_qwen35_adapter",
        "Легковоспламеняющиеся": "refit_adapter",
    }
    for name in builder.ADAPTER_MEMBERS:
        assert builder.sha256_file(destination / "adapter_qwen35" / name) == (
            builder.sha256_file(source / "adapter_qwen35" / name)
        )
        assert (destination / "adapter_qwen35_flammable" / name).is_file()
    run = (destination / "run.py").read_text(encoding="utf-8")
    assert "QWEN35_FLAMMABLE_ADAPTER_PATH" in run
    assert "qwen35_scores = compute_lora_scores" in run
    assert "qwen35_flammable_scores = compute_lora_scores" in run
    assert "qwen35_scores[qwen35_flammable_mask]" in run
    assert report["unrelated_solution140_members_changed"] == 0


def test_runtime_synth_binding_is_fail_closed(tmp_path: Path) -> None:
    source, runtime, output, contract, _synth_ids, synth_payload, run_sha = packet(
        tmp_path, cap=10, preprocessing="thumbnail448"
    )
    with pytest.raises(ValueError, match="runtime gate mismatch"):
        builder.build(
            source=source,
            runtime_dir=runtime,
            refit_output=output,
            destination=tmp_path / "candidate",
            archive=tmp_path / "candidate.zip",
            route="all",
            preprocessing="thumbnail448",
            synth_cap=10,
            expected_source_run_sha256=run_sha,
            expected_runtime_contract=contract,
            expected_synth_ids_sha256="f" * 64,
            expected_synth_payload_sha256=synth_payload,
        )

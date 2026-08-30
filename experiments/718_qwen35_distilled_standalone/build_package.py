from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

from select_candidate import canonical_sha256, sha256, verify_self_hash

MAX_BYTES = 5 * 1024**3
IMAGE_PREPROCESSING = "solution140_first_image_thumbnail_448_lanczos_v1"


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--full-refit-contract", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--run-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite standalone package")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    verify_self_hash(selection, "standalone selection")
    expected_selection = {
        "schema_version": "exp718_standalone_selection_v1",
        "experiment_id": "718",
        "authorized": True,
        "decision": "STANDALONE_4B_AUTHORIZED",
        "image_preprocessing": IMAGE_PREPROCESSING,
        "deployment_score": "category_batch_percentile_rank",
        "deployment_threshold_rule": "median_of_five_outer_train_thresholds",
        "teacher_required_at_inference": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
    }
    mismatch = {
        key: {"expected": value, "actual": selection.get(key)}
        for key, value in expected_selection.items()
        if selection.get(key) != value
    }
    if mismatch:
        raise ValueError(f"standalone selection mismatch: {mismatch}")
    if (
        set(selection.get("deployment_thresholds", {}))
        != {"БАД", "Легковоспламеняющиеся"}
        or not selection.get("gates")
        or not all(selection["gates"].values())
        or float(selection.get("delta_vs_incumbent_140", float("-inf"))) < 0.001
    ):
        raise ValueError("standalone selection gates or thresholds are invalid")
    full = json.loads(args.full_refit_contract.read_text(encoding="utf-8"))
    verify_self_hash(full, "full-refit contract")
    expected_full = {
        "schema_version": "exp715_full_refit_v1",
        "experiment_id": "715",
        "source_experiment_id": "698",
        "full_data": True,
        "technical_smoke": False,
        "teacher_required_at_inference": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
        "image_preprocessing": IMAGE_PREPROCESSING,
    }
    full_mismatch = {
        key: {"expected": value, "actual": full.get(key)}
        for key, value in expected_full.items()
        if full.get(key) != value
    }
    if full_mismatch:
        raise ValueError(f"full-refit contract mismatch: {full_mismatch}")
    if (
        selection.get("full_refit_contract_sha256") != sha256(args.full_refit_contract)
        or selection.get("selected_mode") != full.get("selected_mode")
        or selection.get("adapter_model_sha256") != full.get("adapter_model_sha256")
        or full.get("image_preprocessing") != IMAGE_PREPROCESSING
    ):
        raise ValueError("selection/full-refit binding mismatch")
    adapter_files = {
        path.name: path.read_bytes()
        for path in sorted(args.adapter_dir.iterdir())
        if path.is_file()
    }
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(adapter_files):
        raise ValueError("full-refit adapter lacks required files")
    if bytes_sha256(adapter_files["adapter_model.safetensors"]) != full["adapter_model_sha256"]:
        raise ValueError("adapter model checksum mismatch")
    if len(adapter_files["adapter_model.safetensors"]) != int(full["adapter_bytes"]):
        raise ValueError("adapter model size mismatch")
    config = json.loads(adapter_files["adapter_config.json"])
    if (
        config.get("r") != 16
        or config.get("lora_alpha") != 32
        or config.get("use_rslora") is not True
        or set(config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj"}
    ):
        raise ValueError("adapter configuration mismatch")
    run_bytes = args.run_source.read_bytes()
    forbidden = (b"Qwen3-VL", b"PaddleOCR", b"QWEN_EMBED_MODEL_PATH", b"ANNOTATOR_PRIOR")
    if any(token in run_bytes for token in forbidden):
        raise ValueError("standalone runner references a forbidden extra stage")
    runtime = {
        "schema_version": "exp718_runtime_config_v1",
        "experiment_id": "718",
        "architecture": "qwen35_only",
        "base_model": "Qwen/Qwen3.5-4B",
        "selected_mode": selection["selected_mode"],
        "score": selection["deployment_score"],
        "threshold_rule": selection["deployment_threshold_rule"],
        "thresholds": selection["deployment_thresholds"],
        "image_preprocessing": selection["image_preprocessing"],
        "selection_contract_sha256": sha256(args.selection),
        "adapter_model_sha256": full["adapter_model_sha256"],
        "teacher_required_at_inference": False,
    }
    runtime["contract_sha256"] = canonical_sha256(runtime)
    runtime_bytes = (json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    metadata = {
        "image": "odsai/ecup26-quality-baseline:1.0",
        "entry_point": "python -u run.py",
        "description": "Experiment 718 distilled standalone Qwen3.5-4B",
    }
    metadata_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    members = {
        "run.py": run_bytes,
        "standalone_config.json": runtime_bytes,
        "metadata.json": metadata_bytes,
        **{f"adapter_qwen35/{name}": value for name, value in adapter_files.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for name, value in sorted(members.items()):
                archive.writestr(zip_info(name), value)
        temporary.replace(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    if args.output.stat().st_size >= MAX_BYTES:
        raise ValueError("standalone package exceeds 5 GiB")
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(members):
            raise ValueError("standalone ZIP integrity or member mismatch")
        for name, value in members.items():
            if archive.read(name) != value:
                raise ValueError(f"standalone ZIP member mismatch: {name}")
    report = {
        "schema_version": "exp718_standalone_package_v1",
        "experiment_id": "718",
        "architecture": "qwen35_only",
        "selection_contract_sha256": sha256(args.selection),
        "full_refit_contract_sha256": sha256(args.full_refit_contract),
        "run_sha256": sha256(args.run_source),
        "adapter_model_sha256": full["adapter_model_sha256"],
        "image_preprocessing": IMAGE_PREPROCESSING,
        "teacher_in_submission": False,
        "base_model_in_submission": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
        "extra_model_stages": [],
        "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
        "under_5_gib": args.output.stat().st_size < MAX_BYTES,
        "zip_integrity": "PASS",
        "runtime_smoke": "PENDING",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
